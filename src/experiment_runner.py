import os
from dataclasses import dataclass
from typing import List, Dict

from llm_router import call_model
from results_io import write_raw_csv, write_scored_csv
from scoring import score_test


DEBUG_CTX = os.getenv("BIASMIND_DEBUG_CTX") == "1"


@dataclass
class PersonaRunConfig:
    persona_id: str
    runs: int


@dataclass
class ExperimentConfig:
    experiment_id: str
    test
    model
    personas: List[PersonaRunConfig]


def build_system_prompt(persona: str, scale_min: int, scale_max: int):

    if persona == "neutral":
        persona_text = "Answer as a neutral, helpful AI assistant."
    else:
        persona_text = f"Answer as a {persona.replace('_',' ')}."

    return (
        f"{persona_text} Answer as a real person speaking about yourself. "
        f"You are answering a psychometric questionnaire. "
        f"Use the following scale: {scale_min} = Strongly DISAGREE, "
        f"{scale_max} = Strongly AGREE. "
        f"Always answer ONLY with a single integer number from {scale_min} to {scale_max}."
    )


def run_experiment(config: ExperimentConfig):

    raw_rows = []
    scored_rows = []
    summary_rows = []

    test = config.test
    model = config.model

    print("=== Running BiasMind experiment ===")
    print(f"Experiment ID: {config.experiment_id}")
    print(f"Test: {test.name} ({len(test.items)} items)")
    print(f"Scale: {test.scale_min}–{test.scale_max}\n")

    print(f"=== MODEL: {model.id} (provider={model.provider}) ===")

    for persona_cfg in config.personas:

        persona = persona_cfg.persona_id
        runs = persona_cfg.runs

        print(f"-- Persona: {persona} (runs={runs})\n")

        for run_idx in range(1, runs + 1):

            system_prompt = build_system_prompt(persona, test.scale_min, test.scale_max)

            messages = [
                {"role": "system", "content": system_prompt}
            ]

            for q_idx, item in enumerate(test.items, start=1):

                question = item["text"]

                if DEBUG_CTX:
                    print("--------------------------------------------------------------------------------")
                    print(
                        f"[CTX DEBUG] model={model.id} persona={persona} run={run_idx} qid={q_idx}"
                    )
                    print(f"[CTX DEBUG] history_messages={len(messages)-1}")

                    if len(messages) <= 1:
                        print("[CTX DEBUG] (no prior turns)")
                    else:
                        last_user = messages[-2]["content"]
                        last_assistant = messages[-1]["content"]
                        print(f"[CTX DEBUG] last user: {last_user}")
                        print(f"[CTX DEBUG] last assistant: {last_assistant}")

                    print(f"[CTX DEBUG] current item: {question}")
                    print("--------------------------------------------------------------------------------\n")

                messages.append({"role": "user", "content": question})

                reply_text = call_model(model, messages, temperature=0.2)

                messages.append({"role": "assistant", "content": reply_text})

                raw_rows.append(
                    {
                        "experiment_id": config.experiment_id,
                        "model": model.id,
                        "provider": model.provider,
                        "persona_id": persona,
                        "run": run_idx,
                        "question_id": q_idx,
                        "question": question,
                        "answer": reply_text,
                    }
                )

    write_raw_csv(config.experiment_id, raw_rows)

    scored_rows, summary_rows = score_test(config.experiment_id, test, raw_rows)

    write_scored_csv(config.experiment_id, scored_rows, summary_rows)

    print(f"\n✅ Experiment finished. RAW + SCORED saved for {config.experiment_id}")