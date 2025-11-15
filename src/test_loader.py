import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class Item:
    id: int
    text: str
    trait: str
    reverse: bool = False


@dataclass
class ScoringRule:
    items: List[int]
    formula: str = "mean"  # π.χ. "mean", "sum" κλπ.


@dataclass
class TestDefinition:
    test_name: str
    description: Optional[str]
    reference: Optional[str]
    language: Optional[str]
    scale_min: int
    scale_max: int
    traits: List[str]
    items: List[Item]
    scoring: Dict[str, ScoringRule]


def load_test(path: str | Path) -> TestDefinition:
    """
    Φορτώνει τον ορισμό ενός τεστ από JSON με schema τύπου BFI-10:
    - test_name, description, reference, language
    - scale_min, scale_max
    - traits
    - items: λίστα με {id, text, trait, reverse}
    - scoring: rules ανά trait
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = [
        Item(
            id=item["id"],
            text=item["text"],
            trait=item["trait"],
            reverse=bool(item.get("reverse", False)),
        )
        for item in data["items"]
    ]

    scoring: Dict[str, ScoringRule] = {
        trait_name: ScoringRule(
            items=rule["items"],
            formula=rule.get("formula", "mean"),
        )
        for trait_name, rule in data.get("scoring", {}).items()
    }

    return TestDefinition(
        test_name=data["test_name"],
        description=data.get("description"),
        reference=data.get("reference"),
        language=data.get("language"),
        scale_min=int(data["scale_min"]),
        scale_max=int(data["scale_max"]),
        traits=list(data.get("traits", [])),
        items=items,
        scoring=scoring,
    )
