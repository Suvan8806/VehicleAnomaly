"""Dataset and DataLoader utilities for the MIMII dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import pandas as pd
import torch
import torchaudio  # used for feature extraction ops
import librosa
import soundfile as sf
from torch.utils.data import DataLoader, Dataset

from vehicleanomalynet.features import extract_log_mel, extract_mfcc_delta, pad_or_truncate


SplitName = Literal["train", "val", "test"]


FAULTS_FAN = ("rotating_imbalance", "contamination", "voltage_change")
FAULTS_SLIDER = ("rail_damage", "no_grease", "loose_screws")


def _fault_to_index(fault_type: str) -> int:
    """Map fault string to [0..5]. Normal is excluded (handled with -1)."""
    if fault_type in FAULTS_FAN:
        return FAULTS_FAN.index(fault_type)
    if fault_type in FAULTS_SLIDER:
        return len(FAULTS_FAN) + FAULTS_SLIDER.index(fault_type)
    raise ValueError(f"Unknown fault_type: {fault_type}")


@dataclass(frozen=True)
class Sample:
    features: torch.Tensor
    anomaly_label: int
    fault_label: int
    filepath: str


class MIMIIDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, config: dict, split: SplitName) -> None:
        if "split" not in metadata_df.columns:
            raise ValueError("metadata_df must contain 'split' column.")
        self.df = metadata_df[metadata_df["split"] == split].reset_index(drop=True)
        self.config = config
        self.split = split

        data_cfg = config["data"]
        feat_cfg = config["features"]

        self.sample_rate = int(data_cfg["sample_rate"])
        self.feature_type = str(feat_cfg.get("feature_type", "log_mel"))
        self.target_frames = int(feat_cfg["target_frames"])

        # log-mel params
        self.n_mels = int(feat_cfg["n_mels"])
        self.n_fft = int(feat_cfg["n_fft"])
        self.hop_length = int(feat_cfg["hop_length"])
        self.f_min = float(feat_cfg["f_min"])
        self.f_max = float(feat_cfg["f_max"])

        # mfcc params
        self.n_mfcc = int(feat_cfg["n_mfcc"])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        filepath = str(row["filepath"])

        y, sr_file = sf.read(filepath, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y[:, 0]
        if sr_file != self.sample_rate:
            y = librosa.resample(y, orig_sr=sr_file, target_sr=self.sample_rate)
        waveform = torch.from_numpy(y).unsqueeze(0)  # (1, T)

        if self.feature_type == "log_mel":
            feat = extract_log_mel(
                waveform=waveform,
                sr=self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                f_min=self.f_min,
                f_max=self.f_max,
            )
        elif self.feature_type == "mfcc":
            feat = extract_mfcc_delta(waveform=waveform, sr=self.sample_rate, n_mfcc=self.n_mfcc)
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")

        feat = pad_or_truncate(feat, target_frames=self.target_frames)
        # Add channel dim to match model expectation: (1, C, T) where C is n_mels or 3*n_mfcc
        features = feat.unsqueeze(0).to(torch.float32)

        anomaly_label = int(row["label"])
        if anomaly_label == 0:
            fault_label = -1
        else:
            fault_type = str(row["fault_type"])
            fault_label = _fault_to_index(fault_type)

        machine_type = str(row["machine_type"]) if "machine_type" in row else "unknown"
        return {
            "features": features,
            "anomaly_label": int(anomaly_label),
            "fault_label": int(fault_label),
            "filepath": filepath,
            "machine_type": machine_type,
        }


def get_dataloaders(metadata_path: str, config: dict) -> Dict[str, DataLoader]:
    """Returns {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata_path not found: {metadata_path}")
    df = pd.read_csv(path)

    batch_size = int(config["training"]["batch_size"])

    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        ds = MIMIIDataset(df, config=config, split=split)  # type: ignore[arg-type]
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=False,
        )
    return loaders


def _smoke_test(metadata_path: str, config_path: str = "config.yaml", n: int = 10) -> None:
    """Minimal smoke test for Milestone 3 acceptance checks."""
    with open(config_path, "r", encoding="utf-8") as f:
        import yaml

        cfg = yaml.safe_load(f)

    loaders = get_dataloaders(metadata_path=metadata_path, config=cfg)
    ds = loaders["train"].dataset
    idxs = np.random.choice(len(ds), size=min(n, len(ds)), replace=False)

    shapes = []
    labels = []
    for i in idxs:
        item = ds[int(i)]
        x = item["features"]
        shapes.append(tuple(x.shape))
        labels.append(item["anomaly_label"])
        if not torch.isfinite(x).all():
            raise ValueError("Found non-finite values in features.")

    print("Feature shapes (sample):", shapes[:3])
    print("Unique shapes:", sorted(set(shapes)))
    if len(set(shapes)) != 1:
        raise ValueError("Not all feature tensors have identical shape.")
    print("Anomaly label distribution (sample):", {int(k): int(v) for k, v in pd.Series(labels).value_counts().items()})


if __name__ == "__main__":
    # Run with: .venv\\Scripts\\python.exe -m vehicleanomalynet.dataset data/processed/metadata.csv
    import argparse

    parser = argparse.ArgumentParser(description="Dataset smoke test for VehicleAnomalyNet.")
    parser.add_argument("metadata_path", type=str, help="Path to metadata.csv")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    _smoke_test(metadata_path=args.metadata_path, config_path=args.config, n=10)

