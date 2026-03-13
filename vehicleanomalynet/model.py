"""VehicleAnomalyNet model definition."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VehicleAnomalyNet(nn.Module):
    """CNN + BiGRU with dual heads for anomaly + fault classification."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        feat_cfg = config["features"]

        cnn_channels = model_cfg["cnn_channels"]  # e.g. [32, 64, 128]
        self.input_channels = 1
        self.n_mels = int(feat_cfg["n_mels"])
        self.target_frames = int(feat_cfg["target_frames"])

        # CNN backbone
        layers = []
        in_ch = self.input_channels
        for out_ch in cnn_channels:
            layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # After 3 MaxPool2d(2,2), time and freq are divided by 8.
        self.gru_hidden = int(model_cfg["gru_hidden"])
        self.gru_layers = int(model_cfg["gru_layers"])
        self.bidirectional = bool(model_cfg["gru_bidirectional"])
        self.dropout_p = float(model_cfg["dropout"])
        self.n_fault_classes = int(model_cfg["n_fault_classes"])

        # Determine GRU input size: channels * freq_bins_after_cnn
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_mels, self.target_frames)
            feat = self.cnn(dummy)
            b, c, f, t = feat.shape
        self.cnn_out_channels = c
        self.cnn_out_freq = f

        gru_input_size = c * f
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=self.gru_hidden,
            num_layers=self.gru_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.dropout = nn.Dropout(self.dropout_p)

        gru_out_dim = self.gru_hidden * (2 if self.bidirectional else 1)
        self.anomaly_head = nn.Linear(gru_out_dim, 1)
        self.fault_head = nn.Linear(gru_out_dim, self.n_fault_classes - 1)

    def _forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cnn_features, gru_sequence) without heads.

        Args:
            x: (B, 1, n_mels, T)
        """
        feat = self.cnn(x)  # (B, C, F, T')
        b, c, f, t = feat.shape
        seq = feat.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)  # (B, T', C*F)
        gru_out, _ = self.gru(seq)  # (B, T', H*)
        return feat, gru_out

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Return pooled embedding before heads: (B, H*)."""
        _, gru_out = self._forward_features(x)
        emb = gru_out.mean(dim=1)  # (B, H*)
        emb = self.dropout(emb)
        return emb

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: (B, 1, n_mels, T)

        Returns:
            anomaly_logit: (B, 1)
            fault_logits: (B, n_fault_classes-1)
        """
        emb = self.get_embedding(x)
        anomaly_logit = self.anomaly_head(emb)  # (B, 1)
        fault_logits = self.fault_head(emb)  # (B, n_fault_classes-1)
        return anomaly_logit, fault_logits

