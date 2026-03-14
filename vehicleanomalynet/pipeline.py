"""Ground truth data generation pipeline for MIMII-style acoustic datasets.

Produces labeled, augmented datasets with stratified train/val/test splits
by machine_id. All hyperparameters come from config (no magic numbers).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


# Canonical fault types per PRD (for mapping path-based labels)
FAN_FAULT_TYPES = ("rotating_imbalance", "contamination", "voltage_change")
SLIDER_FAULT_TYPES = ("rail_damage", "no_grease", "loose_screws")


def _rms(x: np.ndarray, eps: float = 1e-12) -> float:
    """Root-mean-square of a 1D array."""
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps))


def _get_data_config(config: dict) -> dict:
    """Extract data section; require key fields."""
    data = config.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("config['data'] must be a dict")
    for key in ("raw_dir", "processed_dir", "sample_rate", "segment_length_s"):
        if key not in data:
            raise ValueError(f"config['data'] must contain '{key}'")
    return data


class GroundTruthPipeline:
    """
    Automated ground truth data generation pipeline.
    Produces high-quality labeled datasets for acoustic perception problems.
    """

    def __init__(self, raw_dir: str, output_dir: str, config: dict) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        self._data_cfg = _get_data_config(config)

    def mix_at_snr(
        self,
        signal: np.ndarray,
        noise: np.ndarray,
        target_snr_db: float,
    ) -> np.ndarray:
        """Mix clean machine audio with background noise at exact target SNR (RMS-based)."""
        signal = np.asarray(signal, dtype=np.float64).flatten()
        noise = np.asarray(noise, dtype=np.float64).flatten()
        if len(noise) != len(signal):
            # Trim or repeat noise to match signal length
            if len(noise) > len(signal):
                noise = noise[: len(signal)]
            else:
                n_repeat = (len(signal) // len(noise)) + 1
                noise = np.tile(noise, n_repeat)[: len(signal)]

        r_s = _rms(signal)
        r_n = _rms(noise)
        # target_snr_db = 10 * log10(signal_power / noise_power) => noise_power = signal_power / 10^(snr/10)
        # In amplitude: r_n' = r_s / 10^(snr_db/20)
        r_n_prime = r_s / (10 ** (target_snr_db / 20.0))
        scale = r_n_prime / (r_n + 1e-12)
        mixed = signal + noise * scale
        return mixed.astype(np.float32)

    def augment_waveform(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Apply time-stretch (±5%), gain jitter (±3 dB), and mild Gaussian noise."""
        waveform = np.asarray(waveform, dtype=np.float64).flatten()
        # Time-stretch ±5%
        rate = 1.0 + random.uniform(-0.05, 0.05)
        y = librosa.effects.time_stretch(waveform, rate=rate)
        # Gain jitter ±3 dB: 10^(db/20)
        db = random.uniform(-3.0, 3.0)
        gain = 10 ** (db / 20.0)
        y = y * gain
        # Mild Gaussian noise (low amplitude so it doesn't dominate)
        y = y + np.random.randn(len(y)).astype(np.float64) * 0.005
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def segment_audio(
        self,
        waveform: np.ndarray,
        sr: int,
        segment_len_s: float,
    ) -> List[np.ndarray]:
        """Segment long recordings into fixed-length clips (non-overlapping)."""
        waveform = np.asarray(waveform, dtype=np.float32).flatten()
        segment_samples = int(segment_len_s * sr)
        segments: List[np.ndarray] = []
        for start in range(0, len(waveform), segment_samples):
            chunk = waveform[start : start + segment_samples]
            if len(chunk) < segment_samples:
                break  # discard remainder
            segments.append(chunk)
        return segments

    def generate(
        self,
        snr_levels: List[float],
        augment_factor: int,
    ) -> pd.DataFrame:
        """
        Main entry point. Returns metadata DataFrame and saves processed WAVs.

        Returns DataFrame with columns:
        [filepath, label, machine_type, machine_id, snr_db, fault_type, split]

        label: 0=normal, 1=anomalous
        split: train/val/test (stratified by machine_id; no machine in more than one split).
        """
        data_cfg = self._data_cfg
        machine_types: List[str] = data_cfg.get("machine_types", ["fan", "slider"])
        if not isinstance(machine_types, list):
            machine_types = list(machine_types)
        sample_rate = int(data_cfg["sample_rate"])
        segment_len_s = float(data_cfg["segment_length_s"])
        train_ratio = float(data_cfg.get("train_split", 0.70))
        val_ratio = float(data_cfg.get("val_split", 0.15))
        test_ratio = float(data_cfg.get("test_split", 0.15))
        max_samples = data_cfg.get("max_generated_samples", 3000)
        if not isinstance(max_samples, int) or max_samples <= 0:
            raise ValueError("config['data']['max_generated_samples'] must be a positive int.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val", "test"):
            (self.output_dir / split_name).mkdir(parents=True, exist_ok=True)

        # Collect all (machine_type, machine_id) pairs and assign splits by machine_id
        machine_ids_by_type: Dict[str, List[str]] = {}
        for mt in machine_types:
            type_root = self.raw_dir / mt
            if not type_root.exists():
                continue
            ids = sorted(
                {
                    p.name
                    for p in type_root.iterdir()
                    if p.is_dir() and p.name.startswith("id_")
                }
            )
            if ids:
                machine_ids_by_type[mt] = ids

        all_machine_ids: List[Tuple[str, str]] = []
        for mt, ids in machine_ids_by_type.items():
            for mid in ids:
                all_machine_ids.append((mt, mid))

        random.shuffle(all_machine_ids)
        n = len(all_machine_ids)
        n_train = max(1, int(n * train_ratio))
        n_val = max(0, int(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        split_assign: Dict[Tuple[str, str], str] = {}
        for i, key in enumerate(all_machine_ids):
            if i < n_train:
                split_assign[key] = "train"
            elif i < n_train + n_val:
                split_assign[key] = "val"
            else:
                split_assign[key] = "test"

        rows: List[Dict[str, Any]] = []
        file_index = 0

        split_targets = {
            "train": int(max_samples * train_ratio),
            "val": int(max_samples * val_ratio),
            "test": max_samples - int(max_samples * train_ratio) - int(max_samples * val_ratio),
        }
        split_counts = {"train": 0, "val": 0, "test": 0}
        # Cap per-label share in each split so test/val have both normal and abnormal (AUC is defined).
        max_label_ratio = float(data_cfg.get("max_label_ratio", 0.75))
        split_label_caps: Dict[str, Dict[int, int]] = {
            s: {0: int(split_targets[s] * max_label_ratio), 1: int(split_targets[s] * max_label_ratio)}
            for s in ("train", "val", "test")
        }
        split_label_counts: Dict[str, Dict[int, int]] = {
            s: {0: 0, 1: 0} for s in ("train", "val", "test")
        }

        machine_keys_by_split: Dict[str, List[Tuple[str, str]]] = {"train": [], "val": [], "test": []}
        for key, sp in split_assign.items():
            machine_keys_by_split[sp].append(key)
        for sp in ("train", "val", "test"):
            random.shuffle(machine_keys_by_split[sp])

        ordered_machine_keys: List[Tuple[str, str]] = (
            machine_keys_by_split["train"] + machine_keys_by_split["val"] + machine_keys_by_split["test"]
        )

        for machine_type, machine_id in ordered_machine_keys:
            type_root = self.raw_dir / machine_type
            machine_dir = type_root / machine_id
            if not machine_dir.exists():
                continue

            split = split_assign.get((machine_type, machine_id), "train")
            if split_counts[split] >= split_targets[split]:
                continue

            # Iterate abnormal first so capped generation includes anomalies.
            for label_name, anomaly_label, default_fault in (
                ("abnormal", 1, "abnormal"),
                ("normal", 0, "normal"),
            ):
                if split_label_counts[split][anomaly_label] >= split_label_caps[split][anomaly_label]:
                    continue
                label_dir = machine_dir / label_name
                if not label_dir.exists():
                    continue
                wav_files = sorted(label_dir.rglob("*.wav"))
                if not wav_files:
                    continue
                random.shuffle(wav_files)

                for wav_path in wav_files:
                    if split_counts[split] >= split_targets[split]:
                        break
                    if split_label_counts[split][anomaly_label] >= split_label_caps[split][anomaly_label]:
                        break
                    fault_type = self._infer_fault_type(
                        wav_path, machine_type, label_name, default_fault
                    )
                    try:
                        y, sr_file = librosa.load(
                            str(wav_path), sr=sample_rate, mono=True
                        )
                    except Exception:
                        continue
                    if sr_file != sample_rate:
                        y = librosa.resample(
                            y, orig_sr=sr_file, target_sr=sample_rate
                        )

                    # Optionally mix with synthetic noise at each SNR
                    for snr_db in snr_levels:
                        if snr_db == 0 and len(snr_levels) == 1:
                            # No separate noise file; use mild synthetic for 0 dB
                            noise = np.random.randn(len(y)).astype(np.float32) * 0.01
                            mixed = self.mix_at_snr(y, noise, float(snr_db))
                        else:
                            noise = np.random.randn(len(y)).astype(np.float32) * 0.02
                            mixed = self.mix_at_snr(y, noise, float(snr_db))

                        segs = self.segment_audio(
                            mixed, sample_rate, segment_len_s
                        )
                        for seg_idx, seg in enumerate(segs):
                            for aug_idx in range(augment_factor):
                                if split_counts[split] >= split_targets[split]:
                                    break
                                if aug_idx == 0:
                                    seg_use = seg
                                else:
                                    seg_use = self.augment_waveform(
                                        seg, sample_rate
                                    )
                                rel_name = (
                                    f"{machine_type}_{machine_id}_{seg_idx}_{aug_idx}_{file_index}.wav"
                                )
                                out_path = (
                                    self.output_dir / split / rel_name
                                )
                                sf.write(
                                    str(out_path),
                                    seg_use,
                                    sample_rate,
                                    subtype="FLOAT",
                                )
                                rows.append(
                                    {
                                        "filepath": str(out_path),
                                        "label": anomaly_label,
                                        "machine_type": machine_type,
                                        "machine_id": machine_id,
                                        "snr_db": snr_db,
                                        "fault_type": fault_type,
                                        "split": split,
                                    }
                                )
                                file_index += 1
                                split_counts[split] += 1
                                split_label_counts[split][anomaly_label] += 1
                                if len(rows) >= max_samples:
                                    df = pd.DataFrame(rows)
                                    return df
                            if split_counts[split] >= split_targets[split]:
                                break
                        if split_counts[split] >= split_targets[split]:
                            break
                    if split_counts[split] >= split_targets[split]:
                        break

        df = pd.DataFrame(rows)
        return df

    def _infer_fault_type(
        self,
        wav_path: Path,
        machine_type: str,
        label_name: str,
        default: str,
    ) -> str:
        """Infer fault_type from path (e.g. abnormal/00) or use default."""
        if label_name == "normal":
            return "normal"
        # MIMII sometimes has abnormal/00, abnormal/01 etc. for fault types
        parent = wav_path.parent.name
        if parent.isdigit():
            idx = int(parent)
            if machine_type == "fan" and 0 <= idx < len(FAN_FAULT_TYPES):
                return FAN_FAULT_TYPES[idx]
            if machine_type == "slider" and 0 <= idx < len(SLIDER_FAULT_TYPES):
                return SLIDER_FAULT_TYPES[idx]
        if machine_type == "fan":
            return FAN_FAULT_TYPES[0]
        if machine_type == "slider":
            return SLIDER_FAULT_TYPES[0]
        return default
