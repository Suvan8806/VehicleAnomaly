"""Run the ground truth data generation pipeline.

Loads config, instantiates GroundTruthPipeline, runs generate(), saves
metadata CSV to config path, and prints summary (total samples, class balance, split sizes).

Usage (from project root):
    .venv\\Scripts\\python.exe scripts\\run_pipeline.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Project root = parent of scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config(path: Path) -> dict:
    """Load YAML config."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VehicleAnomalyNet ground truth data generation pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (relative to project root or absolute).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split assignment.",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("config['data'] must be a dict")

    raw_dir = data_cfg.get("raw_dir", "data/raw")
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    metadata_path = data_cfg.get("metadata_path", "data/processed/metadata.csv")
    snr_levels = data_cfg.get("snr_levels", [0])
    augment_factor = int(data_cfg.get("augment_factor", 3))

    raw_dir_resolved = PROJECT_ROOT / raw_dir
    processed_dir_resolved = PROJECT_ROOT / processed_dir
    metadata_path_resolved = PROJECT_ROOT / metadata_path

    if not raw_dir_resolved.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir_resolved}. "
            "Run scripts/download_data.py first and extract MIMII fan + slider."
        )

    import random
    random.seed(args.seed)

    from vehicleanomalynet.pipeline import GroundTruthPipeline

    pipeline = GroundTruthPipeline(
        raw_dir=str(raw_dir_resolved),
        output_dir=str(processed_dir_resolved),
        config=config,
    )
    df = pipeline.generate(
        snr_levels=snr_levels,
        augment_factor=augment_factor,
    )

    metadata_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_path_resolved, index=False)

    # Summary
    n_total = len(df)
    n_train = (df["split"] == "train").sum()
    n_val = (df["split"] == "val").sum()
    n_test = (df["split"] == "test").sum()
    n_normal = (df["label"] == 0).sum()
    n_anomalous = (df["label"] == 1).sum()

    print("\n=== Pipeline summary ===")
    print(f"Total samples: {n_total}")
    print(f"  train: {n_train}  val: {n_val}  test: {n_test}")
    print(f"  normal: {n_normal}  anomalous: {n_anomalous}")
    if "machine_type" in df.columns:
        for mt in df["machine_type"].unique():
            n_mt = (df["machine_type"] == mt).sum()
            print(f"  {mt}: {n_mt}")
    print(f"Metadata saved: {metadata_path_resolved}")
    print()


if __name__ == "__main__":
    main()
