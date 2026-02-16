# src/analyze_experiment.py
# Usage: python src/analyze_experiment.py --experiment-id 20260215T074904 --results-dir results
import argparse
import math
from pathlib import Path

import pandas as pd


def summarize(scored_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(scored_csv)

    # expected columns in your scored csv
    required = {
        "model", "provider", "persona_id", "run_index", "test_name",
        "score_name", "score_value"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in scored CSV: {sorted(missing)}")

    group_cols = ["model", "provider", "persona_id", "test_name", "score_name"]
    out = (
        df.groupby(group_cols, dropna=False)["score_value"]
          .agg(
              n_runs="count",
              mean="mean",
              std=lambda x: x.std(ddof=1),
              min="min",
              max="max",
          )
          .reset_index()
    )
    out["sem"] = out["std"] / (out["n_runs"] ** 0.5)

    out = out.rename(columns={"score_name": "trait"})
    out = out.sort_values(["model", "persona_id", "test_name", "trait"]).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-id", required=True, help="e.g. 20260215T074904")
    ap.add_argument("--results-dir", default="results", help="base results dir (default: results)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    scored_csv = results_dir / "scored" / f"scored_{args.experiment_id}.csv"

    if not scored_csv.exists():
        raise FileNotFoundError(f"Could not find: {scored_csv}")

    summary_df = summarize(scored_csv)

    # Display to console (no file output)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 140)

    print(f"\n=== SUMMARY for experiment {args.experiment_id} ===")
    print(f"Source: {scored_csv}")
    print(summary_df.to_string(index=False))
    print("=== END SUMMARY ===\n")


if __name__ == "__main__":
    main()
