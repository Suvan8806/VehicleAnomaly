"""Feature extraction utilities for VehicleAnomalyNet.

All hyperparameters must come from `config.yaml` (no magic numbers).
"""

from __future__ import annotations

import torch
import torchaudio


def extract_log_mel(
    waveform: torch.Tensor,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    """Returns log-Mel spectrogram of shape (n_mels, T).

    Args:
        waveform: Tensor of shape (T,) or (1, T) in float32.
        sr: Sample rate.
    """
    if waveform.dim() == 2:
        waveform_mono = waveform[:1, :]
    elif waveform.dim() == 1:
        waveform_mono = waveform.unsqueeze(0)
    else:
        raise ValueError(f"waveform must be 1D or 2D, got shape {tuple(waveform.shape)}")

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
        normalized=False,
    )(waveform_mono)  # (1, n_mels, T)

    log_mel = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)(mel)
    return log_mel.squeeze(0)  # (n_mels, T)


def extract_mfcc_delta(
    waveform: torch.Tensor,
    sr: int,
    n_mfcc: int,
) -> torch.Tensor:
    """Returns [MFCC; delta; delta-delta] concatenated, shape (3*n_mfcc, T)."""
    if waveform.dim() == 2:
        waveform_mono = waveform[:1, :]
    elif waveform.dim() == 1:
        waveform_mono = waveform.unsqueeze(0)
    else:
        raise ValueError(f"waveform must be 1D or 2D, got shape {tuple(waveform.shape)}")

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 1024, "hop_length": 512, "n_mels": 128},
    )(waveform_mono)  # (1, n_mfcc, T)

    d1 = torchaudio.functional.compute_deltas(mfcc)
    d2 = torchaudio.functional.compute_deltas(d1)
    feats = torch.cat([mfcc, d1, d2], dim=1).squeeze(0)  # (3*n_mfcc, T)
    return feats


def pad_or_truncate(
    feature: torch.Tensor,
    target_frames: int,
) -> torch.Tensor:
    """Pad with zeros or truncate time axis to fixed length.

    Args:
        feature: Tensor shaped (C, T).
        target_frames: desired T.
    """
    if feature.dim() != 2:
        raise ValueError(f"feature must be 2D (C, T), got shape {tuple(feature.shape)}")
    c, t = feature.shape
    if t == target_frames:
        return feature
    if t > target_frames:
        return feature[:, :target_frames]
    pad = torch.zeros((c, target_frames - t), dtype=feature.dtype, device=feature.device)
    return torch.cat([feature, pad], dim=1)

