from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import re

from input_loader import ModelDef, PersonaDef
from test_loader import load_test
from results_io import write_metadata_json, write_raw_csv, write_scored_csv
from llm_router import call_model


@dataclass
class PersonaRunConfig:
    persona: PersonaDef
    runs: int
    memory_within_persona: str


@dataclass
class ExperimentConfig:
    experiment_id: str
    test_name: str
    test_file: Path
    models: List[ModelDef]
    personas: List[PersonaRunConfig]
    memory_between_personas: str


def _now_iso():
    return datetime.utcnow().isoformat(timespec="seconds")


def _infer_scale_from_test(test_def):
    scale_min = getattr(test_def, "scale_min", None)
    scale_max = getattr(test_def, "scale_max", None)

    if scale_min is not None and scale_max is not None:
        return scale_min, scale_max

    return 1, 5


def _parse_likert_answer(text, min_val, max_val):
    text = text.strip()

    ints = [int(m.group(0)) for m in re.finditer(r"-?\d+", text)]
    in_range = [x for x in ints if min_val <= x <= max_val]

    if in_range:
        return in_range[-1]

    return (min_val + max_val) // 2


def _compute_scored_rows(test_def, raw_rows):
    scored_rows = []

    if not raw_rows:
        return scored_rows

    scale_min, scale_max = _infer_scale_from_test(test_def)

    grouped = {}

    for row in raw_rows:
        key = (
            row["model"],
            row["provider"],
            row["persona_id"],
            row["run_index"],
            row["test_name"],
        )

        grouped.setdefault(key, []).append(row)

    for key, rows in grouped.items():

        trait_values = {}

        for r in rows:
            trait = r["trait"]
            ans = int(r["answer"])
            rev = r["reverse"]

            if rev:
                adj = scale_min + scale_max - ans
            else:
                adj = ans

            trait_values.setdefault(trait, []).append(adj)

        for trait, vals in trait_values.items():

            mean_val = sum(vals) / len(vals)

            scored_rows.append(
                {
                    "model": key[0],
                    "provider": key[1],
                    "persona_id": key[2],
                    "run_index": key[3],
                    "test_name": key[4],
                    "score_name": trait,
                    "score_kind": "trait",
                    "score_value": round(mean_val, 3),
                }
            )

    return scored_rows


def run_experiment(config: ExperimentConfig):

    test_def = load_test(config.test_file)
    scale_min, scale_max = _infer_scale_from_test(test_def)

    metadata = {
        "experiment_id": config.experiment_id,
        "test": config.test_name,
    }

    write_metadata_json(metadata)

    raw_rows = []

    for model in config.models:

        print(f"\nMODEL: {model.id}")

        for persona_cfg in config.personas:

            persona = persona_cfg.persona

            for run_index in range(1, persona_cfg.runs + 1):

                print(f"Persona {persona.id} run {run_index}")

                system_prompt = (
                    f"{persona.prompt_prefix} "
                    "You are answering a psychometric questionnaire. "
                    f"Use the following scale: {scale_min} = Strongly DISAGREE, {scale_max} = Strongly AGREE. "
                    f"Always answer ONLY with a single integer number from {scale_min} to {scale_max}."
                )

                # conversation memory inside run
                run_context = [
                    {"role": "system", "content": system_prompt}
                ]

                for item in test_def.items:

                    run_context.append(
                        {"role": "user", "content": item.text}
                    )

                    reply_text = call_model(
                        model,
                        run_context,
                        temperature=0.2,
                    )

                    answer_val = _parse_likert_answer(
                        reply_text,
                        scale_min,
                        scale_max,
                    )

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
                            "timestamp_run": _now_iso(),
                        }
                    )

                    run_context.append(
                        {"role": "assistant", "content": reply_text}
                    )

    write_raw_csv(config.experiment_id, raw_rows)

    scored_rows = _compute_scored_rows(test_def, raw_rows)

    write_scored_csv(config.experiment_id, scored_rows)

    print("\nExperiment finished")