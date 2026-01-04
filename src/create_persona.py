import json
import os
import argparse


def create_persona(persona_id: str, prompt_prefix: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    persona = {
        "id": persona_id,
        "prompt_prefix": prompt_prefix
    }

    output_path = os.path.join(output_dir, f"{persona_id}.json")

    if os.path.exists(output_path):
        raise FileExistsError(f"Persona '{persona_id}' already exists.")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(persona, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new persona JSON")
    parser.add_argument("--id", required=True, help="Persona id (e.g. manager)")
    parser.add_argument("--prompt", required=True, help="Persona prompt prefix")
    parser.add_argument(
        "--output-dir",
        default="data/personas",
        help="Directory to store persona JSON files"
    )

    args = parser.parse_args()

    path = create_persona(
        persona_id=args.id,
        prompt_prefix=args.prompt,
        output_dir=args.output_dir
    )

    print(f"Persona created at: {path}")
