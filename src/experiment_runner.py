from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

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


def _parse_likert_answer(text: str, min_val: int = 1, max_val: int = 5) -> int:
    """
    Παίρνει την απάντηση του μοντέλου (κείμενο) και προσπαθεί να βγάλει
    έναν ακέραιο από min_val έως max_val.
    Αν δεν τα καταφέρει, επιστρέφει την μέση τιμή.
    """
    text = text.strip()
    digits = [ch for ch in text if ch.isdigit()]
    if digits:
        try:
            val = int(digits[0])
            if min_val <= val <= max_val:
                return val
        except ValueError:
            pass
    return (min_val + max_val) // 2


def _infer_scale_from_test(test_def: TestDefinition) -> tuple[int, int]:
    """
    Αν το TestDefinition έχει scale_min / scale_max, τα χρησιμοποιούμε.
    Αλλιώς υποθέτουμε 1–5.
    """
    scale_min = getattr(test_def, "scale_min", None)
    scale_max = getattr(test_def, "scale_max", None)
    if isinstance(scale_min, int) and isinstance(scale_max, int):
        return scale_min, scale_max
    return 1, 5


def _compute_scored_rows(
    test_def: TestDefinition,
    raw_rows: List[Dict],
) -> List[Dict]:
    """
    Υπολογίζει scores ανά trait για κάθε (model, persona, run, test_name).
    Επιστρέφει λίστα από rows συμβατά με το write_scored_csv.
    """
    scored_rows: List[Dict] = []

    if not raw_rows:
        return scored_rows

    scale_min, scale_max = _infer_scale_from_test(test_def)

    # group by (model, provider, persona_id, run_index, test_name)
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
        # trait -> list of adjusted answers
        trait_values: Dict[str, List[float]] = {}

        for r in rows:
            trait = r["trait"]
            ans = int(r["answer"])
            rev = r["reverse"]
            # handle boolean or string
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

        # compute mean per trait
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
    """
    Πλήρες experiment:
    - φορτώνει test
    - γράφει metadata_<experiment_id>.json
    - κάνει loop: model → persona → runs → items
    - για κάθε item καλεί LLM μέσω call_model
    - γράφει raw_<experiment_id>.csv
    - υπολογίζει trait-level scores και γράφει scored_<experiment_id>.csv
    """
    test_def: TestDefinition = load_test(config.test_file)

    # METADATA
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
    }
    write_metadata_json(metadata)

    raw_rows: List[Dict] = []

    print("=== Running BiasMind experiment ===")
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Test: {config.test_name} ({len(test_def.items)} items)")
    print(f"Memory between personas: {config.memory_between_personas}")
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
                    "Always answer ONLY with a single integer number from 1 to 5."
                )

                for item in test_def.items:
                    messages = []
                    messages.append(
                        {"role": "system", "content": system_prompt}
                    )
                    messages.append(
                        {"role": "user", "content": item.text}
                    )

                    reply_text = call_model(model, messages, temperature=0.2)
                    answer_val = _parse_likert_answer(reply_text, 1, 5)
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

                    context_messages.append(
                        {"role": "assistant", "content": reply_text}
                    )

            previous_persona_id = persona.id

    # RAW CSV
    write_raw_csv(config.experiment_id, raw_rows)

    # SCORED CSV
    scored_rows = _compute_scored_rows(test_def, raw_rows)
    write_scored_csv(config.experiment_id, scored_rows)

    print(f"\n✅ Experiment finished. RAW + SCORED saved for {config.experiment_id}")
