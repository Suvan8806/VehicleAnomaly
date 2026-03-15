"""Loss functions for dual-task anomaly and fault classification."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class DualTaskLoss(nn.Module):
    """
    Combined anomaly detection + fault classification loss.
    Ignores fault_head loss for normal samples (they have no fault type).
    """

    def __init__(
        self,
        anomaly_weight: float,
        fault_weight: float,
        anomaly_pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.anomaly_weight = float(anomaly_weight)
        self.fault_weight = float(fault_weight)
        if anomaly_pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([float(anomaly_pos_weight)], dtype=torch.float32)
            )
        else:
            self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        anomaly_logit: torch.Tensor,
        fault_logit: torch.Tensor,
        anomaly_label: torch.Tensor,
        fault_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            anomaly_logit: (B, 1)
            fault_logit: (B, C_fault)  (no normal class; only faults)
            anomaly_label: (B,) in {0,1}
            fault_label: (B,) with -1 for normal, [0..C_fault-1] for faults.
        """
        anomaly_label = anomaly_label.float().view(-1, 1)
        loss_anom = self.bce(anomaly_logit, anomaly_label)

        # CrossEntropyLoss with ignore_index=-1 automatically masks normals
        loss_fault = self.ce(fault_logit, fault_label.long())

        total = self.anomaly_weight * loss_anom + self.fault_weight * loss_fault
        return {"total": total, "anomaly": loss_anom, "fault": loss_fault}

