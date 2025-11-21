from pathlib import Path
from typing import List, Dict
import csv
import json


# Στήλες για το RAW CSV
RAW_COLUMNS = [
    "experiment_id",
    "model",
    "provider",
    "persona_id",
    "run_index",
    "test_name",
    "question_id",
    "question_text",
    "trait",
    "reverse",
    "answer",
    "timestamp_run",
]


def write_raw_csv(
    experiment_id: str,
    rows: List[Dict],
    base_dir: str | Path = "results/raw",
) -> Path:
    """
    Γράφει όλες τις raw απαντήσεις σε:
      results/raw/raw_<experiment_id>.csv

    rows: λίστα από dicts με κλειδιά που ταιριάζουν στις RAW_COLUMNS.
    Ό,τι κλειδί λείπει θα μείνει κενό.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    path = base_dir / f"raw_{experiment_id}.csv"

    empty_row = {col: "" for col in RAW_COLUMNS}

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        writer.writeheader()

        for row in rows:
            merged = empty_row | row
            merged["experiment_id"] = experiment_id
            writer.writerow(merged)

    return path


# Στήλες για το SCORED CSV
SCORED_COLUMNS = [
    "experiment_id",
    "model",
    "provider",
    "persona_id",
    "run_index",
    "test_name",
    "score_name",
    "score_kind",
    "score_value",
    "score_normalized",
    "summary_label",
]


def write_scored_csv(
    experiment_id: str,
    rows: List[Dict],
    base_dir: str | Path = "results/scored",
) -> Path:
    """
    Γράφει τα scored αποτελέσματα σε:
      results/scored/scored_<experiment_id>.csv

    Κάθε row = ένα score (π.χ. Openness, Economic, overall).
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    path = base_dir / f"scored_{experiment_id}.csv"

    empty_row = {col: "" for col in SCORED_COLUMNS}

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORED_COLUMNS)
        writer.writeheader()

        for row in rows:
            merged = empty_row | row
            merged["experiment_id"] = experiment_id
            writer.writerow(merged)

    return path


def write_metadata_json(
    experiment_meta: Dict,
    base_dir: str | Path = "results/metadata",
) -> Path:
    """
    Γράφει το metadata_<experiment_id>.json.
    Περιμένουμε στο dict να υπάρχει key 'experiment_id'.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = experiment_meta["experiment_id"]
    path = base_dir / f"metadata_{experiment_id}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(experiment_meta, f, ensure_ascii=False, indent=2)

    return path
