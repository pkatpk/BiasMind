import argparse
from datetime import datetime
from pathlib import Path
from typing import List

from input_loader import load_models, load_personas, ModelDef, PersonaDef
from experiment_runner import ExperimentConfig, PersonaRunConfig, run_experiment


def _generate_experiment_id() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def _parse_persona_specs(
    specs: List[str],
    personas_defs: List[PersonaDef],
) -> List[PersonaRunConfig]:
    """
    specs: λίστα από strings τύπου:
      "neutral:1:fresh"
      "farmer:2:continuous"

    όπου:
      id = personas/<id>.json
      runs = πόσα runs
      memory_within = "fresh" ή "continuous"
    """
    persona_map = {p.id: p for p in personas_defs}
    result: List[PersonaRunConfig] = []

    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Persona spec '{spec}' πρέπει να είναι της μορφής id:runs:memory_within "
                f"(π.χ. neutral:2:fresh)"
            )
        pid, runs_str, mem = parts
        pid = pid.strip()
        runs = int(runs_str)

        if pid not in persona_map:
            raise ValueError(f"Persona id '{pid}' δεν βρέθηκε στα data/personas/*.json")

        if mem not in ("fresh", "continuous"):
            raise ValueError(
                f"memory_within_persona για '{pid}' πρέπει να είναι 'fresh' ή 'continuous', όχι '{mem}'"
            )

        result.append(
            PersonaRunConfig(
                persona=persona_map[pid],
                runs=runs,
                memory_within_persona=mem,
            )
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BiasMind experiment runner")

    parser.add_argument(
        "--experiment-id",
        help="ID του experiment (αν δεν δοθεί, θα φτιαχτεί timestamp-based).",
    )

    parser.add_argument(
        "--test-file",
        required=True,
        help="Διαδρομή στο test JSON (π.χ. data/tests/test_bfi10.json).",
    )

    parser.add_argument(
        "--test-name",
        help="Όνομα του test (π.χ. BFI-10). Αν δεν δοθεί, θα χρησιμοποιηθεί το όνομα του αρχείου.",
    )

    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="ID μοντέλου, π.χ. tinyllama-chat. Μπορεί να δοθεί πολλές φορές.",
    )

    parser.add_argument(
        "--persona",
        action="append",
        required=True,
        help=(
            "Persona spec της μορφής id:runs:memory_within, π.χ. "
            "neutral:1:fresh ή farmer:2:continuous. "
            "Μπορεί να δοθεί πολλές φορές."
        ),
    )

    parser.add_argument(
        "--memory-between",
        choices=["reset", "carry_over"],
        default="reset",
        help="Συμπεριφορά μνήμης όταν αλλάζει persona.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment_id = args.experiment_id or _generate_experiment_id()

    test_file = Path(args.test_file)
    if not test_file.exists():
        raise FileNotFoundError(f"Test file δεν βρέθηκε: {test_file}")

    test_name = args.test_name or test_file.stem

    # Φορτώνουμε τα μοντέλα
    model_ids = [m.strip() for m in args.model]
    models: List[ModelDef] = load_models(model_ids)

    # Φορτώνουμε τις personas (μόνο τα PersonaDef)
    persona_ids = [spec.split(":")[0].strip() for spec in args.persona]
    personas_defs: List[PersonaDef] = load_personas(persona_ids)

    # Συνδυάζουμε personas + runs + memory_within
    persona_cfgs: List[PersonaRunConfig] = _parse_persona_specs(args.persona, personas_defs)

    config = ExperimentConfig(
        experiment_id=experiment_id,
        test_name=test_name,
        test_file=test_file,
        models=models,
        personas=persona_cfgs,
        memory_between_personas=args.memory_between,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
