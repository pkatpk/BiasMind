# experiment_runner.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, UTC
import math
import os
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


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _infer_scale_from_test(test_def: TestDefinition) -> Tuple[int, int]:
    scale_min = getattr(test_def, "scale_min", None)
    scale_max = getattr(test_def, "scale_max", None)
    if isinstance(scale_min, int) and isinstance(scale_max, int):
        return scale_min, scale_max
    return 1, 5


def _parse_likert_answer(text: str, min_val: int, max_val: int) -> int:
    """
    Robust Likert parsing:
    - βρίσκει όλους τους ακέραιους στο text
    - κρατά τον τελευταίο εντός scale
    - fallback στο midpoint
    """
    text = (text or "").strip()
    ints = [int(m.group(0)) for m in re.finditer(r"-?\d+", text)]
    in_range = [x for x in ints if min_val <= x <= max_val]

    if in_range:
        return in_range[-1]

    return (min_val + max_val) // 2


def _compute_scored_rows(
    test_def: TestDefinition,
    raw_rows: List[Dict],
) -> List[Dict]:
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


def _compute_summary_rows(scored_rows: List[Dict]) -> List[Dict]:
    """
    Summary ανά:
    model, provider, persona_id, test_name, trait
    με:
    n_runs, mean, std, min, max, sem
    """
    summary_rows: List[Dict] = []
    if not scored_rows:
        return summary_rows

    grouped: Dict[tuple, List[float]] = {}
    for row in scored_rows:
        key = (
            row["model"],
            row["provider"],
            row["persona_id"],
            row["test_name"],
            row["score_name"],
        )
        grouped.setdefault(key, []).append(float(row["score_value"]))

    for (model, provider, persona_id, test_name, trait), vals in grouped.items():
        n = len(vals)
        mean_val = sum(vals) / n if n else 0.0

        if n > 1:
            var = sum((x - mean_val) ** 2 for x in vals) / (n - 1)
            std_val = math.sqrt(var)
        else:
            std_val = 0.0

        sem_val = (std_val / math.sqrt(n)) if n > 0 else 0.0

        summary_rows.append(
            {
                "model": model,
                "provider": provider,
                "persona_id": persona_id,
                "test_name": test_name,
                "trait": trait,
                "n_runs": n,
                "mean": round(mean_val, 6),
                "std": round(std_val, 6),
                "min": round(min(vals), 6),
                "max": round(max(vals), 6),
                "sem": round(sem_val, 6),
            }
        )

    return summary_rows


def run_experiment(config: ExperimentConfig) -> None:
    """
    Memory policy:
    - fresh:
        κάθε item ανεξάρτητο, χωρίς history
    - continuous:
        όλα τα items ενός run είναι μία συνομιλία
    - μετά το τέλος κάθε run γίνεται πάντα reset
    - δεν υπάρχει memory between runs / personas

    Debug:
    - BIASMIND_DEBUG_CTX=1
    """
    debug_ctx = (os.getenv("BIASMIND_DEBUG_CTX") or "").strip().lower() in ("1", "true", "yes", "on")

    test_def: TestDefinition = load_test(config.test_file)
    scale_min, scale_max = _infer_scale_from_test(test_def)

    metadata = {
        "experiment_id": config.experiment_id,
        "created_at": _now_iso(),
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
        "memory_policy": {
            "within_run": ["fresh", "continuous"],
            "between_runs": "always_reset",
            "between_personas": "always_reset",
        },
        "scale_min": scale_min,
        "scale_max": scale_max,
    }
    write_metadata_json(metadata)

    raw_rows: List[Dict] = []

    print("=== Running BiasMind experiment ===")
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Test: {config.test_name} ({len(test_def.items)} items)")
    print(f"Scale: {scale_min}–{scale_max}")

    for model in config.models:
        print(f"\n=== MODEL: {model.id} (provider={model.provider}) ===")

        for persona_cfg in config.personas:
            persona = persona_cfg.persona
            memory_mode = persona_cfg.memory_within_persona

            if memory_mode not in ("fresh", "continuous"):
                raise ValueError(
                    f"Unsupported memory mode '{memory_mode}' for persona '{persona.id}'. "
                    "Expected 'fresh' or 'continuous'."
                )

            print(f"-- Persona: {persona.id} (runs={persona_cfg.runs}, memory={memory_mode})")

            for run_index in range(1, persona_cfg.runs + 1):
                # πάντα νέα συνομιλία στην αρχή κάθε run
                run_context: List[Dict] = []

                system_prompt = (
                    f"{persona.prompt_prefix} "
                    "You are answering a psychometric questionnaire. "
                    f"Use the following scale: {scale_min} = Strongly DISAGREE, {scale_max} = Strongly AGREE. "
                    f"Always answer ONLY with a single integer number from {scale_min} to {scale_max}."
                )

                for item in test_def.items:
                    if memory_mode == "continuous":
                        messages = (
                            [{"role": "system", "content": system_prompt}]
                            + run_context
                            + [{"role": "user", "content": item.text}]
                        )
                    else:
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": item.text},
                        ]

                    if debug_ctx:
                        print("\n" + "-" * 80)
                        print(f"[CTX DEBUG] model={model.id} persona={persona.id} run={run_index} qid={item.id}")
                        print(f"[CTX DEBUG] memory_mode={memory_mode}")
                        print(f"[CTX DEBUG] history_messages={len(run_context) if memory_mode == 'continuous' else 0}")
                        if memory_mode == "continuous" and len(run_context) >= 2:
                            print("[CTX DEBUG] last user:", run_context[-2]["content"][:200])
                            print("[CTX DEBUG] last assistant:", run_context[-1]["content"][:200])
                        else:
                            print("[CTX DEBUG] (no prior turns)")
                        print("[CTX DEBUG] current item:", item.text[:200])
                        print("-" * 80 + "\n")

                    reply_text = call_model(model, messages, temperature=0.2)
                    answer_val = _parse_likert_answer(reply_text, scale_min, scale_max)

                    raw_rows.append(
                        {
                            "experiment_id": config.experiment_id,
                            "timestamp_run": _now_iso(),
                            "model": model.id,
                            "provider": model.provider,
                            "persona_id": persona.id,
                            "run_index": run_index,
                            "test_name": config.test_name,
                            "question_id": item.id,
                            "question_text": item.text,
                            "trait": getattr(item, "trait", ""),
                            "reverse": getattr(item, "reverse", False),
                            "answer": answer_val,
                        }
                    )

                    if memory_mode == "continuous":
                        run_context.extend(
                            [
                                {"role": "user", "content": item.text},
                                {"role": "assistant", "content": str(answer_val)},
                            ]
                        )

                # reset στο τέλος του run
                run_context = []

    scored_rows = _compute_scored_rows(test_def, raw_rows)
    summary_rows = _compute_summary_rows(scored_rows)

    write_raw_csv(config.experiment_id, raw_rows)
    write_scored_csv(config.experiment_id, scored_rows, summary_rows)

    print(f"\n✅ Experiment finished. RAW + SCORED saved for {config.experiment_id}")