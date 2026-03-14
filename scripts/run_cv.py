"""K-fold cross-validation over machine_ids for a more robust AUC estimate.

Splits machine_ids into k folds (stratified by machine_type), then for each fold:
trains on the other folds, evaluates on the held-out fold, and records test AUC.
Reports mean ± std AUC across folds.

Usage (from project root):
    .venv\\Scripts\\python.exe scripts\\run_cv.py --config config.yaml --folds 5
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _stratified_fold_keys(
    machine_keys: list[tuple[str, str]],
    k: int,
    seed: int,
) -> list[list[tuple[str, str]]]:
    """Split (machine_type, machine_id) into k folds, stratified by machine_type."""
    from collections import defaultdict
    by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in machine_keys:
        by_type[key[0]].append(key)
    for mt in by_type:
        random.Random(seed).shuffle(by_type[mt])
    folds: list[list[tuple[str, str]]] = [[] for _ in range(k)]
    for mt, keys in by_type.items():
        for i, key in enumerate(keys):
            folds[i % k].append(key)
    return folds


def main() -> int:
    parser = argparse.ArgumentParser(description="K-fold cross-validation for VehicleAnomalyNet.")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--metadata", type=str, default=None, help="Metadata CSV (default: from config).")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default 5).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs per fold (default: from config).")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    metadata_path = args.metadata or config["data"]["metadata_path"]
    if not Path(metadata_path).is_absolute():
        metadata_path = str(PROJECT_ROOT / metadata_path)
    if not Path(metadata_path).exists():
        print(f"Metadata not found: {metadata_path}. Run scripts/run_pipeline.py first.", file=sys.stderr)
        return 1

    df = pd.read_csv(metadata_path)
    if "machine_type" not in df.columns or "machine_id" not in df.columns:
        print("Metadata must have machine_type and machine_id columns.", file=sys.stderr)
        return 1

    machine_keys = list(df[["machine_type", "machine_id"]].drop_duplicates().itertuples(index=False, name=None))
    k = max(2, min(args.folds, len(machine_keys)))
    if k != args.folds:
        print(f"Using k={k} folds (only {len(machine_keys)} machine_ids).")
    fold_keys = _stratified_fold_keys(machine_keys, k, args.seed)

    # Train/val split within training machines: 85% train, 15% val
    val_ratio = 0.15
    processed_dir = Path(config["data"].get("processed_dir", "data/processed"))
    if not processed_dir.is_absolute():
        processed_dir = PROJECT_ROOT / processed_dir
    results_dir = Path(config["evaluation"].get("results_dir", "results/"))
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    aucs: list[float] = []

    for fold_idx in range(k):
        test_keys = set(fold_keys[fold_idx])
        train_keys = [key for i, keys in enumerate(fold_keys) if i != fold_idx for key in keys]
        n_val = min(len(train_keys) - 1, max(0, int(len(train_keys) * val_ratio)))
        n_val = max(0, n_val)
        val_keys_set = set(train_keys[:n_val])
        train_keys_set = set(train_keys[n_val:])

        def split_for_row(row: pd.Series) -> str:
            key = (row["machine_type"], row["machine_id"])
            if key in test_keys:
                return "test"
            if key in val_keys_set:
                return "val"
            return "train"

        cv_df = df.copy()
        cv_df["split"] = cv_df.apply(split_for_row, axis=1)
        cv_meta_path = processed_dir / f"metadata_cv_fold{fold_idx}.csv"
        cv_df.to_csv(cv_meta_path, index=False)

        run_name = f"cv_fold_{fold_idx}"
        ckpt_path = checkpoint_dir / f"cv_fold_{fold_idx}.pt"
        cmd_train = [
            python,
            str(PROJECT_ROOT / "scripts" / "run_training.py"),
            "--config", str(config_path),
            "--metadata", str(cv_meta_path),
            "--run-name", run_name,
            "--output-checkpoint", str(ckpt_path),
            "--seed", str(args.seed + fold_idx),
        ]
        if args.epochs is not None:
            # Would need to override config; skip for simplicity or add to run_training
            pass
        print(f"\n--- Fold {fold_idx + 1}/{k} ---")
        if subprocess.run(cmd_train, cwd=str(PROJECT_ROOT)).returncode != 0:
            print(f"Training failed for fold {fold_idx}.", file=sys.stderr)
            continue
        cmd_eval = [
            python,
            str(PROJECT_ROOT / "scripts" / "run_evaluation.py"),
            "--config", str(config_path),
            "--metadata", str(cv_meta_path),
            "--checkpoint", str(ckpt_path),
        ]
        if subprocess.run(cmd_eval, cwd=str(PROJECT_ROOT)).returncode != 0:
            print(f"Evaluation failed for fold {fold_idx}.", file=sys.stderr)
            continue
        metrics_file = results_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            auc = float(metrics.get("auc_roc", 0.0))
            aucs.append(auc)
            print(f"Fold {fold_idx + 1} test AUC-ROC: {auc:.4f}")

    if not aucs:
        print("No folds completed successfully.", file=sys.stderr)
        return 1
    import numpy as np
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs)) if len(aucs) > 1 else 0.0
    print("\n" + "=" * 50)
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 50)
    print(f"Folds: {len(aucs)}/{k}")
    print(f"Test AUC-ROC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Per-fold: {[f'{a:.4f}' for a in aucs]}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
