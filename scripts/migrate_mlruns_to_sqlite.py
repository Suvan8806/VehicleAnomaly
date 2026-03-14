"""Migrate existing MLflow runs from mlruns/ (file store) to mlflow.db (SQLite).

Run this once so the MLflow UI (which uses SQLite) shows your training runs.
Stop the MLflow UI first if it is running.

Usage:
    .venv\\Scripts\\python.exe scripts\\migrate_mlruns_to_sqlite.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MLRUNS = PROJECT_ROOT / "mlruns"
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"


def sqlite_uri(path: Path) -> str:
    return "sqlite:///" + str(path.resolve()).replace("\\", "/")


def mlruns_has_runs() -> bool:
    """True if mlruns/ contains at least one experiment with runs."""
    if not MLRUNS.exists():
        return False
    for exp_dir in MLRUNS.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        meta = exp_dir / "meta.yaml"
        if meta.exists():
            for child in exp_dir.iterdir():
                if child.is_dir() and len(child.name) == 32 and all(c in "0123456789abcdef" for c in child.name.lower()):
                    return True
    return False


def main() -> int:
    if not MLRUNS.exists():
        print("mlruns/ not found. Nothing to migrate.")
        return 0

    if not mlruns_has_runs():
        print("mlruns/ has no runs. Run training first: scripts/run_training.py")
        return 0

    if MLFLOW_DB.exists():
        print(f"Removing existing {MLFLOW_DB.name} so migration can copy from mlruns/ ...")
        try:
            MLFLOW_DB.unlink()
        except OSError as e:
            if getattr(e, "winerror", None) == 32 or "being used" in str(e).lower():
                print("ERROR: Stop the MLflow UI (Ctrl+C in the terminal where it runs), then run this script again.")
            raise SystemExit(1) from e

    # Use plain path for source (file:// with %20 breaks on paths containing spaces)
    source_path = str(MLRUNS.resolve())
    target_uri = sqlite_uri(MLFLOW_DB)
    print(f"Migrating mlruns/ -> {MLFLOW_DB.name} ...")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "migrate-filestore",
            "--source", source_path,
            "--target", target_uri,
        ],
        capture_output=False,
    )
    if r.returncode != 0:
        print("Migration failed. Ensure MLflow UI is stopped and MLflow >= 3.10.")
        return 1
    print("Done. Start the UI: python scripts/run_mlflow_ui.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
