"""CLI entry point for training VehicleAnomalyNet.

Usage (from project root):
    .venv\\Scripts\\python.exe scripts\\run_training.py --config config.yaml --run-name baseline_v2
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Use same SQLite backend as run_mlflow_ui.py so runs show in the UI (Runs / Overview tabs)
_db = PROJECT_ROOT / "mlflow.db"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///" + str(_db.resolve()).replace("\\", "/"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vehicleanomalynet.dataset import get_dataloaders
from vehicleanomalynet.model import VehicleAnomalyNet
from vehicleanomalynet.train import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VehicleAnomalyNet.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--run-name", type=str, default="baseline_v2")
    parser.add_argument("--metadata", type=str, default=None, help="Override metadata CSV path (e.g. for CV).")
    parser.add_argument("--output-checkpoint", type=str, default=None, help="Override path to save best checkpoint (e.g. for CV).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    metadata_path = args.metadata or config["data"]["metadata_path"]
    if not Path(metadata_path).is_absolute():
        metadata_path = str(PROJECT_ROOT / metadata_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}. Run scripts/run_pipeline.py first."
        )

    dataloaders = get_dataloaders(metadata_path, config)
    model = VehicleAnomalyNet(config)
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        config=config,
        run_name=args.run_name,
    )

    best_model_path = args.output_checkpoint or config["training"].get("best_model_path", "checkpoints/best_model.pt")
    if not Path(best_model_path).is_absolute():
        best_model_path = str(PROJECT_ROOT / best_model_path)
    trainer.best_model_path = Path(best_model_path)
    trainer.best_model_path.parent.mkdir(parents=True, exist_ok=True)

    best = trainer.fit()
    print("Best val AUC-ROC:", best.get("best_val_auc_roc"))
    print("Checkpoint saved:", trainer.best_model_path)


if __name__ == "__main__":
    main()
