"""Run evaluation on the test set and save results + plots.

Usage (from project root):
    .venv\\Scripts\\python.exe scripts\\run_evaluation.py --config config.yaml
    .venv\\Scripts\\python.exe scripts\\run_evaluation.py --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vehicleanomalynet.dataset import get_dataloaders
from vehicleanomalynet.evaluate import evaluate, plot_results
from vehicleanomalynet.model import VehicleAnomalyNet


def _get_device(device_cfg: str) -> torch.device:
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VehicleAnomalyNet on test set.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (default: config training.best_model_path)")
    parser.add_argument("--metadata", type=str, default=None, help="Override metadata CSV path (e.g. for CV).")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    metadata_path = args.metadata or config["data"]["metadata_path"]
    if not Path(metadata_path).is_absolute():
        metadata_path = str(PROJECT_ROOT / metadata_path)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}. Run scripts/run_pipeline.py first.")

    checkpoint = args.checkpoint or config["training"].get("best_model_path", "checkpoints/best_model.pt")
    if not Path(checkpoint).is_absolute():
        checkpoint = str(PROJECT_ROOT / checkpoint)
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run training first.")

    results_dir = config["evaluation"].get("results_dir", "results/")
    if not Path(results_dir).is_absolute():
        results_dir = str(PROJECT_ROOT / results_dir)
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    device = _get_device(str(config["training"].get("device", "auto")))
    dataloaders = get_dataloaders(metadata_path, config)
    model = VehicleAnomalyNet(config)
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()

    eval_dict = evaluate(model, dataloaders["test"], device)
    plot_results(eval_dict, results_dir)

    # Build JSON-serializable metrics
    metrics = {
        "auc_roc": float(eval_dict["auc_roc"]),
        "f1_macro": float(eval_dict["f1_macro"]),
        "precision": float(eval_dict["precision"]),
        "recall": float(eval_dict["recall"]),
        "per_machine_auc": {k: float(v) for k, v in eval_dict["per_machine_auc"].items()},
        "confusion_matrix": eval_dict["confusion_matrix"].tolist(),
    }
    metrics_path = Path(results_dir) / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {metrics_path}")

    # Print results table (ASCII for Windows console)
    auc = eval_dict["auc_roc"]
    f1 = eval_dict["f1_macro"]
    prec = eval_dict["precision"]
    rec = eval_dict["recall"]
    per_machine = eval_dict["per_machine_auc"]
    print("")
    print("+-----------------------------------------+")
    print("|         VehicleAnomalyNet Results        |")
    print("+----------------------+------------------+")
    print(f"| Test AUC-ROC         | {auc:.3f}            |")
    print(f"| Test F1 (macro)      | {f1:.3f}            |")
    print(f"| Test Precision       | {prec:.3f}            |")
    print(f"| Test Recall          | {rec:.3f}            |")
    print("+----------------------+------------------+")
    for mt, val in sorted(per_machine.items()):
        label = f"{mt.capitalize()} AUC-ROC"
        print(f"| {label:<20} | {val:.3f}            |")
    print("+----------------------+------------------+")
    print("")
    if auc < 0.75:
        print("Note: Test AUC-ROC < 0.75 — consider debugging or more training.")
    else:
        print("Test AUC-ROC >= 0.75 — acceptance criterion met.")


if __name__ == "__main__":
    main()
