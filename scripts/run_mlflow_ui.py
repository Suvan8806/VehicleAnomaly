"""Launch MLflow UI with a SQLite backend so Runs and Overview work.

MLflow 3.x with a file backend only shows Traces; using SQLite restores the
Runs table and Overview. Existing runs in mlruns/ are migrated into mlflow.db
once (if the migrate command is available).

Usage:
    .venv\\Scripts\\python.exe scripts\\run_mlflow_ui.py
    .venv\\Scripts\\python.exe scripts\\run_mlflow_ui.py --port 5001
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS = PROJECT_ROOT / "mlruns"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


def sqlite_uri(path: Path) -> str:
    """SQLite URI with forward slashes for Windows."""
    return "sqlite:///" + str(path.resolve()).replace("\\", "/")


def _mlruns_has_runs() -> bool:
    if not MLRUNS.exists():
        return False
    for exp_dir in MLRUNS.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        for child in exp_dir.iterdir():
            if child.is_dir() and len(child.name) == 32 and all(c in "0123456789abcdef" for c in child.name.lower()):
                return True
    return False


def _sqlite_run_count(uri: str) -> int:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=uri)
        exps = client.search_experiments()
        n = 0
        for e in exps:
            n += len(client.search_runs(experiment_ids=[e.experiment_id], max_results=1))
        return n
    except Exception:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch MLflow UI for VehicleAnomalyNet runs.")
    parser.add_argument("--port", type=int, default=5000, help="Port for MLflow UI (default 5000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host (default 127.0.0.1)")
    parser.add_argument("--no-migrate", action="store_true", help="Skip migrating mlruns/ to SQLite")
    args = parser.parse_args()

    backend_uri = sqlite_uri(MLFLOW_DB)

    # Migrate mlruns -> sqlite when mlruns has runs but sqlite is empty (so UI shows runs)
    if not args.no_migrate and _mlruns_has_runs():
        sqlite_has_runs = MLFLOW_DB.exists() and _sqlite_run_count(backend_uri) > 0
        if not sqlite_has_runs:
            if MLFLOW_DB.exists():
                print(f"Removing empty {MLFLOW_DB.name} to migrate from mlruns/ ...")
                MLFLOW_DB.unlink()
            source_path = str(MLRUNS.resolve())
            target_uri = sqlite_uri(MLFLOW_DB)
            print(f"Migrating mlruns/ -> {MLFLOW_DB.name} ...")
            migrate = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mlflow",
                    "migrate-filestore",
                    "--source", source_path,
                    "--target", target_uri,
                ],
                capture_output=True,
                text=True,
            )
            if migrate.returncode == 0:
                print("Migration done.")
            else:
                print("Migration failed (MLflow 3.10+ required). Run: python scripts/migrate_mlruns_to_sqlite.py")
                if migrate.stderr:
                    print(migrate.stderr.strip()[:400])

    cmd = [
        sys.executable,
        "-m",
        "mlflow",
        "ui",
        "--backend-store-uri",
        backend_uri,
        "--port",
        str(args.port),
        "--host",
        args.host,
    ]
    base = f"http://{args.host}:{args.port}"
    print(f"Starting MLflow UI (SQLite): {MLFLOW_DB.name}")
    print(f"Open: {base}")
    print("")
    print("To see your training RUNS (not Traces):")
    print("  In the top navigation bar, switch from 'GenAI' to 'Classical ML' (or 'ML').")
    print("  Then open Experiments -> Default; the Runs table will appear.")
    print(f"  Direct link to try: {base}/experiments/0")
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
