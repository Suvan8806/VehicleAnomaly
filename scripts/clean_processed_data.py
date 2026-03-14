"""Clean processed data so the pipeline can run from a fresh slate.

Removes data/processed/train, data/processed/val, data/processed/test,
and data/processed/metadata.csv (or paths from config). Raw data in data/raw/
is left unchanged.

Usage (from project root):
    .venv\\Scripts\\python.exe scripts\\clean_processed_data.py
    .venv\\Scripts\\python.exe scripts\\clean_processed_data.py --config config.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove processed data (train/val/test + metadata) for a clean pipeline run."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml for processed_dir and metadata_path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        data_cfg = config.get("data", {})
        processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
        metadata_path = Path(data_cfg.get("metadata_path", "data/processed/metadata.csv"))
    else:
        processed_dir = Path("data/processed")
        metadata_path = Path("data/processed/metadata.csv")

    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir
    if not metadata_path.is_absolute():
        metadata_path = PROJECT_ROOT / metadata_path

    to_remove: list[tuple[Path, str]] = []
    for sub in ("train", "val", "test"):
        p = processed_dir / sub
        if p.exists():
            to_remove.append((p, "dir"))
    if metadata_path.exists():
        to_remove.append((metadata_path, "file"))

    if not to_remove:
        print("Nothing to clean. Processed paths are already empty or missing.")
        return 0

    if args.dry_run:
        print("Would remove:")
        for p, kind in to_remove:
            print(f"  {p} ({kind})")
        return 0

    for p, kind in to_remove:
        try:
            if kind == "dir":
                shutil.rmtree(p)
                print(f"Removed: {p}")
            else:
                p.unlink()
                print(f"Removed: {p}")
        except OSError as e:
            print(f"Error removing {p}: {e}", file=sys.stderr)
            return 1

    print("Done. Run scripts/run_pipeline.py to regenerate data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
