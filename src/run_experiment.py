# src/run_experiment.py
import argparse
from datetime import datetime, UTC
from pathlib import Path
from typing import Tuple

from input_loader import load_models, load_personas, ModelDef, PersonaDef
from experiment_runner import ExperimentConfig, PersonaRunConfig, run_experiment


def _generate_experiment_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def _parse_persona_spec(spec: str, personas_defs: list[PersonaDef]) -> PersonaRunConfig:
    """
    spec μορφής:
      "neutral:50"

    όπου:
      id = personas/<id>.json
      runs = πόσα runs
    """
    persona_map = {p.id: p for p in personas_defs}

    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Persona spec '{spec}' πρέπει να είναι της μορφής id:runs "
            f"(π.χ. neutral:50)"
        )

    pid, runs_str = parts
    pid = pid.strip()

    try:
        runs = int(runs_str)
    except ValueError as exc:
        raise ValueError(f"Τα runs για persona '{pid}' πρέπει να είναι ακέραιος.") from exc

    if runs <= 0:
        raise ValueError(f"Τα runs για persona '{pid}' πρέπει να είναι > 0.")

    if pid not in persona_map:
        raise ValueError(f"Persona id '{pid}' δεν βρέθηκε στα data/personas/*.json")

    return PersonaRunConfig(
        persona=persona_map[pid],
        runs=runs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BiasMind experiment runner")

    parser.add_argument(
        "--experiment-id",
        help="ID του experiment (αν δεν δοθεί, θα φτιαχτεί timestamp-based).",
    )

    parser.add_argument(
        "--test-file",
        required=True,
        help="Διαδρομή στο test JSON (π.χ. data/tests/rwa22_en.json).",
    )

    parser.add_argument(
        "--test-name",
        help="Όνομα του test. Αν δεν δοθεί, θα χρησιμοποιηθεί το όνομα του αρχείου.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="ID μοντέλου, π.χ. tinyllama-chat.",
    )

    parser.add_argument(
        "--persona",
        required=True,
        help="Persona spec της μορφής id:runs, π.χ. neutral:50",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment_id = args.experiment_id or _generate_experiment_id()

    test_file = Path(args.test_file)
    if not test_file.exists():
        raise FileNotFoundError(f"Test file δεν βρέθηκε: {test_file}")

    test_name = args.test_name or test_file.stem

    model_id = args.model.strip()
    models: list[ModelDef] = load_models([model_id])

    persona_id = args.persona.split(":")[0].strip()
    personas_defs: list[PersonaDef] = load_personas([persona_id])

    persona_cfg: PersonaRunConfig = _parse_persona_spec(args.persona, personas_defs)

    config = ExperimentConfig(
        experiment_id=experiment_id,
        test_name=test_name,
        test_file=test_file,
        models=models,
        persona=persona_cfg,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()