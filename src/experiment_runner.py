from dataclasses import dataclass
from pathlib import Path
from typing import List

from input_loader import ModelDef, PersonaDef
from test_loader import load_test, TestDefinition
from results_io import write_metadata_json


@dataclass
class PersonaRunConfig:
    """
    Persona + πόσα runs + πώς δουλεύει η μνήμη ΜΕΣΑ σε αυτή την persona.
    """
    persona: PersonaDef
    runs: int
    memory_within_persona: str  # "fresh" ή "continuous"


@dataclass
class ExperimentConfig:
    """
    Ορισμός ενός πειράματος που θα τρέξει το BiasMind.
    """
    experiment_id: str
    test_name: str
    test_file: Path
    models: List[ModelDef]
    personas: List[PersonaRunConfig]
    memory_between_personas: str  # "reset" ή "carry_over"


def run_experiment(config: ExperimentConfig) -> None:
    """
    Demo της ροής του πειράματος.
    - Φορτώνει το test
    - Γράφει metadata_<experiment_id>.json
    - Κάνει loop: model → persona → runs
    - Υπολογίζει αν πρέπει να καθαρίσει το context
    - Προς το παρόν ΜΟΝΟ τυπώνει τι θα γινόταν (χωρίς πραγματικό LLM ακόμη)
    """
    # Φόρτωση test definition
    test_def: TestDefinition = load_test(config.test_file)

    # --- METADATA: γράφουμε metadata_<experiment_id>.json ---
    metadata = {
        "experiment_id": config.experiment_id,
        "test": config.test_name,
        "models": [
            {
                "id": m.id,
                "provider": m.provider,
                "api_name": m.api_name,
            }
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
    # -------------------------------------------------------

    print("=== Running BiasMind experiment ===")
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Test: {config.test_name} ({len(test_def.items)} items)")
    print(f"Memory between personas: {config.memory_between_personas}")
    print()

    for model in config.models:
        print(f"\n=== MODEL: {model.id} (provider={model.provider}) ===")
        previous_persona_id: str | None = None

        for persona_cfg in config.personas:
            persona = persona_cfg.persona

            # Απόφαση για reset όταν αλλάζει persona
            if previous_persona_id is None:
                # πρώτη persona για αυτό το μοντέλο
                between_reset = True
            else:
                between_reset = (config.memory_between_personas == "reset")

            print(
                f"\n-- Persona: {persona.id} "
                f"(runs={persona_cfg.runs}, "
                f"memory_within={persona_cfg.memory_within_persona})"
            )

            for run_index in range(1, persona_cfg.runs + 1):
                # Απόφαση για reset context μέσα στην persona
                if run_index == 1:
                    # Πρώτο run αυτής της persona για το συγκεκριμένο μοντέλο
                    reset_context = between_reset
                else:
                    # Επόμενα runs της ίδιας persona
                    reset_context = (persona_cfg.memory_within_persona == "fresh")

                print(
                    f"  Run {run_index}: "
                    f"reset_context={reset_context}"
                )

                # ΕΔΩ αργότερα:
                # - φτιάχνουμε prompt με βάση persona + item
                # - κάνουμε call στο LLM
                # - συλλέγουμε raw απαντήσεις
                # - κάνουμε scoring
                # - γράφουμε raw/scored CSV

            previous_persona_id = persona.id
