"""List MLflow runs from the project's SQLite store (no server needed).

Use this to verify runs exist and get direct UI links. Run after training.

Usage:
    .venv\\Scripts\\python.exe scripts\\list_mlflow_runs.py
    .venv\\Scripts\\python.exe scripts\\list_mlflow_runs.py --limit 20
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


def main() -> int:
    parser = argparse.ArgumentParser(description="List MLflow runs from project store.")
    parser.add_argument("--limit", type=int, default=10, help="Max runs to show per experiment (default 10)")
    parser.add_argument("--experiment", type=str, default=None, help="Experiment name or id (default: all)")
    args = parser.parse_args()

    if not MLFLOW_DB.exists():
        print(f"No {MLFLOW_DB.name} found. Run training first (scripts/run_training.py).")
        return 1

    uri = "sqlite:///" + str(MLFLOW_DB.resolve()).replace("\\", "/")
    os.environ["MLFLOW_TRACKING_URI"] = uri

    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        exps = client.search_experiments()
    except Exception as e:
        print(f"Error reading store: {e}")
        return 1

    if not exps:
        print("No experiments found.")
        return 0

    for exp in exps:
        if args.experiment and exp.name != args.experiment and exp.experiment_id != args.experiment:
            continue
        print(f"\nExperiment: {exp.name} (id={exp.experiment_id})")
        try:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=args.limit)
        except Exception as e:
            print(f"  Error: {e}")
            continue
        if not runs:
            print("  No runs.")
            continue
        for r in runs:
            metrics = {k: f"{v:.4f}" for k, v in (r.data.metrics or {}).items()}
            params = (r.data.params or {})
            print(f"  Run: {r.info.run_id[:8]}...  name={r.info.run_name or '-'}  metrics={metrics}  params={list(params.keys())[:5]}")
        print(f"  Total runs in experiment: use MLflow UI to see all.")
        print(f"  UI (switch to 'Classical ML' to see Runs): http://127.0.0.1:5000/experiments/{exp.experiment_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
