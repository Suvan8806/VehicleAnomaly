"""Evaluation utilities: compute metrics and plot results on the test set."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Full evaluation on test set.

    Returns:
        Dict with: auc_roc, f1_macro, precision, recall, per_machine_auc,
        confusion_matrix (2x2 for anomaly), anomaly_scores, labels, machine_types.
    """
    model.eval()
    all_scores: list = []
    all_labels: list = []
    all_fault_labels: list = []
    all_fault_preds: list = []
    all_machine_types: list = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["features"].to(device)
            anomaly_label = batch["anomaly_label"]
            fault_label = batch["fault_label"]
            machine_type = batch["machine_type"]

            anomaly_logit, fault_logits = model(x)
            score = torch.sigmoid(anomaly_logit).squeeze(1).cpu().numpy()
            fault_pred = fault_logits.argmax(dim=1).cpu().numpy()

            all_scores.append(score)
            all_labels.append(anomaly_label.numpy())
            all_fault_labels.append(fault_label.numpy())
            all_fault_preds.append(fault_pred)
            all_machine_types.append(np.array(machine_type))

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    fault_labels = np.concatenate(all_fault_labels, axis=0)
    fault_preds = np.concatenate(all_fault_preds, axis=0)
    machine_types = np.concatenate([np.asarray(m) for m in all_machine_types], axis=0)
    if isinstance(machine_types[0], str):
        pass
    else:
        machine_types = np.array([str(m) for m in machine_types])

    # Anomaly metrics
    n_classes = len(np.unique(labels))
    if n_classes < 2:
        auc_roc = 0.0
    else:
        auc_roc = float(roc_auc_score(labels, scores))

    pred_binary = (scores >= 0.5).astype(np.int64)
    precision = float(precision_score(labels, pred_binary, zero_division=0.0))
    recall = float(recall_score(labels, pred_binary, zero_division=0.0))
    cm = confusion_matrix(labels, pred_binary, labels=[0, 1])
    if cm.shape != (2, 2):
        cm = np.zeros((2, 2), dtype=np.int64)
        for i, j in zip(labels, pred_binary):
            cm[int(i), int(j)] += 1

    # Fault F1 (only on abnormal samples; fault_label >= 0)
    abnormal_mask = fault_labels >= 0
    if abnormal_mask.sum() > 0:
        f1_macro = float(
            f1_score(
                fault_labels[abnormal_mask],
                fault_preds[abnormal_mask],
                average="macro",
                zero_division=0.0,
            )
        )
    else:
        f1_macro = 0.0

    # Per-machine-type AUC
    per_machine_auc: Dict[str, float] = {}
    for mt in np.unique(machine_types):
        mask = machine_types == mt
        if mask.sum() == 0:
            continue
        y_m = labels[mask]
        s_m = scores[mask]
        if len(np.unique(y_m)) < 2:
            per_machine_auc[str(mt)] = 0.0
        else:
            per_machine_auc[str(mt)] = float(roc_auc_score(y_m, s_m))

    return {
        "auc_roc": auc_roc,
        "f1_macro": f1_macro,
        "precision": precision,
        "recall": recall,
        "per_machine_auc": per_machine_auc,
        "confusion_matrix": cm,
        "anomaly_scores": scores,
        "labels": labels,
        "machine_types": machine_types,
    }


def plot_results(eval_dict: Dict[str, Any], output_dir: str | Path) -> None:
    """Save ROC curve, confusion matrix, and score distribution to output_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = eval_dict["anomaly_scores"]
    labels = eval_dict["labels"]
    cm = eval_dict["confusion_matrix"]
    auc_roc = eval_dict["auc_roc"]

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {auc_roc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Anomaly Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(5, 4))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)
    plt.title("Confusion Matrix (Normal vs Anomaly)")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal", "Anomaly"])
    plt.yticks(tick_marks, ["Normal", "Anomaly"])
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Score distribution (normal vs abnormal)
    plt.figure(figsize=(6, 4))
    normal_scores = scores[labels == 0]
    abnormal_scores = scores[labels == 1]
    if len(normal_scores) > 0:
        plt.hist(normal_scores, bins=30, alpha=0.6, label="Normal", color="green", density=True)
    if len(abnormal_scores) > 0:
        plt.hist(abnormal_scores, bins=30, alpha=0.6, label="Anomaly", color="red", density=True)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close()
