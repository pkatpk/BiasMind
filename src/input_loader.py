import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


# ----- Βασικοί τύποι -----

@dataclass
class ModelDef:
    id: str
    provider: str
    api_name: str


@dataclass
class PersonaDef:
    id: str
    prompt_prefix: str


# ----- Loaders για μοντέλα -----

def load_model(model_id: str, base_dir: str | Path = "data/models") -> ModelDef:
    """
    Φορτώνει ένα μοντέλο από αρχείο JSON στο data/models/<model_id>.json
    """
    base_dir = Path(base_dir)
    path = base_dir / f"{model_id}.json"

    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return ModelDef(
        id=data["id"],
        provider=data["provider"],
        api_name=data["api_name"],
    )


def load_models(model_ids: List[str], base_dir: str | Path = "data/models") -> List[ModelDef]:
    """
    Φορτώνει μια λίστα από μοντέλα με βάση τα ids.
    """
    return [load_model(mid, base_dir=base_dir) for mid in model_ids]


# ----- Loaders για personas -----

def load_persona(persona_id: str, base_dir: str | Path = "data/personas") -> PersonaDef:
    """
    Φορτώνει μια persona από αρχείο JSON στο data/personas/<persona_id>.json
    """
    base_dir = Path(base_dir)
    path = base_dir / f"{persona_id}.json"

    if not path.exists():
        raise FileNotFoundError(f"Persona config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return PersonaDef(
        id=data["id"],
        prompt_prefix=data["prompt_prefix"],
    )


def load_personas(persona_ids: List[str], base_dir: str | Path = "data/personas") -> List[PersonaDef]:
    """
    Φορτώνει μια λίστα από personas με βάση τα ids.
    """
    return [load_persona(pid, base_dir=base_dir) for pid in persona_ids]
