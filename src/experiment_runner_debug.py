from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

from input_loader import ModelDef, PersonaDef
from test_loader import load_test, TestDefinition
from results_io import write_metadata_json, write_raw_csv, write_scored_csv
from llm_router import call_model


@dataclass
class PersonaRunConfig:
    persona: PersonaDef
    runs: int
    memory_within_persona: str  # "fresh" ή "continuous"


@dataclass
class ExperimentConfig:
    experiment_id: str
    test_name: str
    test_file: Path
    models: List[ModelDef]
    personas: List[PersonaRunConfig]
    memory_between_personas: str  # "reset" ή "carry_over"


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _infer_scale_from_test(test_def: TestDefinition) -> Tuple[int, int]:
    """
    Αν το TestDefinition έχει scale_min / scale_max, τα χρησιμοποιούμε.
    Αλλιώς υποθέτουμε 1–5.
    """
    scale_min = getattr(test_def, "scale_min", None)
    scale_max = getattr(test_def, "scale_max", None)
    if isinstance(scale_min, int) and isinstance(scale_max, int):
        return scale_min, scale_max
    return 1, 5


def _parse_likert_answer(text: str, min_val: int, max_val: int) -> int:
    """
    Robust parsing:
    - Βρίσκει ακέραιους αριθμούς μέσα στο κείμενο (όχι "πρώτο digit").
    - Επιλέγει έναν που είναι εντός [min_val, max_val].
    - Αν βρει πολλούς, προτιμά τον ΤΕΛΕΥΤΑΙΟ (συχνά είναι η τελική επιλογή).
    - Αν δεν βρει κανέναν, επιστρέφει midpoint.
    """
    text = (text or "").strip()

    # find integers like 1, 5, 10 (if ever needed)
    ints = [int(m.group(0)) for m in re.finditer(r"-?\d+", text)]

    in_range = [x for x in ints if min_val <= x <= max_val]
    if in_range:
        return in_range[-1]  # prefer last in-range integer

    # fallback: midpoint
    return (min_val + max_val) // 2


def _compute_scored_rows(
    test_def: TestDefinition,
    raw_rows: List[Dict],
) -> List[Dict]:
    """
    Υπολογίζει scores ανά trait για κάθε (model, persona, run, test_name).
    """
    scored_rows: List[Dict] = []
    if not raw_rows:
        return scored_rows

    scale_min, scale_max = _infer_scale_from_test(test_def)

    grouped: Dict[tuple, List[Dict]] = {}
    for row in raw_rows:
        key = (
            row["model"],
            row["provider"],
            row["persona_id"],
            row["run_index"],
            row["test_name"],
        )
        grouped.setdefault(key, []).append(row)

    for (model, provider, persona_id, run_index, test_name), rows in grouped.items():
        trait_values: Dict[str, List[float]] = {}

        for r in rows:
            trait = r["trait"]
            ans = int(r["answer"])
            rev = r["reverse"]

            is_rev = (
                rev is True
                or rev == "true"
                or rev == "True"
                or rev == 1
                or rev == "1"
            )

            if is_rev:
                adj = scale_min + scale_max - ans
            else:
                adj = ans

            trait_values.setdefault(trait, []).append(adj)

        for trait, vals in trait_values.items():
            mean_val = sum(vals) / len(vals) if vals else 0.0
            scored_rows.append(
                {
                    "model": model,
                    "provider": provider,
                    "persona_id": persona_id,
                    "run_index": run_index,
                    "test_name": test_name,
                    "score_name": trait,
                    "score_kind": "trait",
                    "score_value": round(mean_val, 3),
                    "score_normalized": "",
                    "summary_label": "",
                }
            )

    return scored_rows


def run_experiment(config: ExperimentConfig) -> None:
    test_def: TestDefinition = load_test(config.test_file)
    scale_min, scale_max = _infer_scale_from_test(test_def)

    metadata = {
        "experiment_id": config.experiment_id,
        "test": config.test_name,
        "models": [
            {"id": m.id, "provider": m.provider, "api_name": m.api_name}
            for m in config.models
        ],
        "personas": [
            {
                "id": p.persona.id,
                "prompt_prefix": p.persona.prompt_prefix,
                "runs": p.runs,
                "memory_within_persona": p.memory_within_persona,
            }
            for p in config.personas
        ],
        "memory_between_personas": config.memory_between_personas,
        "scale_min": scale_min,
        "scale_max": scale_max,
    }
    write_metadata_json(metadata)

    raw_rows: List[Dict] = []

    print("=== Running BiasMind experiment (DEBUG) ===")
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Test: {config.test_name} ({len(test_def.items)} items)")
    print(f"Memory between personas: {config.memory_between_personas}")
    print(f"Scale: {scale_min}–{scale_max}")
    print()

    for model in config.models:
        print(f"\n=== MODEL: {model.id} (provider={model.provider}) ===")
        previous_persona_id: Optional[str] = None
        context_messages: List[Dict] = []

        for persona_cfg in config.personas:
            persona = persona_cfg.persona

            if previous_persona_id is None:
                between_reset = True
            else:
                between_reset = (config.memory_between_personas == "reset")

            print(
                f"\n-- Persona: {persona.id} "
                f"(runs={persona_cfg.runs}, "
                f"memory_within={persona_cfg.memory_within_persona})"
            )

            for run_index in range(1, persona_cfg.runs + 1):
                if run_index == 1:
                    reset_context = between_reset
                else:
                    reset_context = (persona_cfg.memory_within_persona == "fresh")

                print(f"  Run {run_index}: reset_context={reset_context}")

                if reset_context:
                    context_messages = []

                system_prompt = (
                    f"{persona.prompt_prefix} "
                    "You are answering a psychometric questionnaire. "
                    f"Always answer ONLY with a single integer number from {scale_min} to {scale_max}."
                )

                # Αν θες λιγότερα logs, κάνε προσωρινά: test_def.items[:2]
                for item in test_def.items:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item.text},
                    ]

                    reply_text = call_model(model, messages, temperature=0.2)

                    # DEBUG prints (πριν γράψουμε το raw row)
                    print("\n=== DEBUG ===")
                    print("PERSONA:", persona.id, "RUN:", run_index, "ITEM:", item.id)
                    print("REPLY repr:", repr(reply_text))
                    print("REPLY:", reply_text)
                    print("DIGITS:", [ch for ch in (reply_text or "") if ch.isdigit()])

                    answer_val = _parse_likert_answer(reply_text, scale_min, scale_max)
                    print("PARSED:", answer_val)
                    print("=== END DEBUG ===\n")

                    timestamp = _now_iso()

                    raw_rows.append(
                        {
                            "model": model.id,
                            "provider": model.provider,
                            "persona_id": persona.id,
                            "run_index": run_index,
                            "test_name": config.test_name,
                            "question_id": item.id,
                            "question_text": item.text,
                            "trait": item.trait,
                            "reverse": item.reverse,
                            "answer": answer_val,
                            "timestamp_run": timestamp,
                        }
                    )

                    context_messages.append({"role": "assistant", "content": reply_text})

            previous_persona_id = persona.id

    write_raw_csv(config.experiment_id, raw_rows)

    scored_rows = _compute_scored_rows(test_def, raw_rows)
    write_scored_csv(config.experiment_id, scored_rows)

    print(f"\n✅ Experiment finished. RAW + SCORED saved for {config.experiment_id}")
