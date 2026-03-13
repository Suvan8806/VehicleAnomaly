"""Model smoke test for VehicleAnomalyNet (Milestone 4).

Run from project root with:
    .venv\\Scripts\\python.exe scripts\\model_smoketest.py
"""

from __future__ import annotations

import yaml
import torch

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vehicleanomalynet.model import VehicleAnomalyNet
from vehicleanomalynet.losses import DualTaskLoss


def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = VehicleAnomalyNet(cfg)
    n_mels = cfg["features"]["n_mels"]
    T = cfg["features"]["target_frames"]
    B = 8

    x = torch.randn(B, 1, n_mels, T)
    anomaly_logit, fault_logit = model(x)

    print("anomaly_logit shape:", tuple(anomaly_logit.shape))
    print("fault_logit shape:", tuple(fault_logit.shape))

    anomaly_label = torch.randint(0, 2, (B,))
    fault_label = torch.randint(0, cfg["model"]["n_fault_classes"] - 1, (B,))
    fault_label[anomaly_label == 0] = -1

    criterion = DualTaskLoss(
        anomaly_weight=cfg["training"]["loss_weights"]["anomaly"],
        fault_weight=cfg["training"]["loss_weights"]["fault"],
    )
    loss_dict = criterion(anomaly_logit, fault_logit, anomaly_label, fault_label)
    print("Losses:", {k: float(v.detach()) for k, v in loss_dict.items()})

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", n_params)


if __name__ == "__main__":
    main()

