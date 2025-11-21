from dataclasses import dataclass
from pathlib import Path
from typing import List

from input_loader import ModelDef, PersonaDef
from test_loader import load_test, TestDefinition


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
    ΠΡΟΣ ΤΟ ΠΑΡΟΝ: μόνο demo της ροής.
    - Φορτώνει το test
    - Κάνει loop: model → persona → runs
    - Υπολογίζει αν πρέπει να καθαρίσει το context
    - Τυπώνει τι θα γινόταν (χωρίς πραγματικό LLM ακόμη)
    """
    test_def: TestDefinition = load_test(config.test_file)

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

            # Μεταξύ personas
            if previous_persona_id is None:
                between_reset = True  # πρώτη persona για αυτό το μοντέλο
            else:
                between_reset = (config.memory_between_personas == "reset")

            print(f"\n-- Persona: {persona.id} (runs={persona_cfg.runs}, "
                  f"memory_within={persona_cfg.memory_within_persona})")

            for run_index in range(1, persona_cfg.runs + 1):
                # Μέσα στην persona, ανά run
                if run_index == 1:
                    reset_context = between_reset
                else:
                    reset_context = (persona_cfg.memory_within_persona == "fresh")

                print(
                    f"  Run {run_index}: "
                    f"reset_context={reset_context}"
                )
                # ΕΔΩ αργότερα:
                # - φτιάχνουμε prompt
                # - κάνουμε call στο LLM
                # - γράφουμε raw + scored + metadata

            previous_persona_id = persona.id
