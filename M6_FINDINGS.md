## Milestone 6 Findings — Evaluation & Current Scores

This document captures the current state of Milestone 6 (Evaluation), how the metrics are produced, what the code is doing end-to-end, and why the latest scores look the way they do. It is meant as a snapshot for future planning and debugging, not a final report.

---

### Current approach (per-type)

We now **train and evaluate each machine type separately** to avoid cross-domain evaluation (e.g. training on fan and testing on slider, which yielded test AUC ~0.48).

- **Per-type configs:** `config_fan.yaml` and `config_slider.yaml` restrict `data.machine_types` to `["fan"]` and `["slider"]` respectively, with type-specific `processed_dir`, `metadata_path`, and `results_dir` so fan and slider runs do not overwrite each other.
- **Checkpoints:** Save one model per type: `checkpoints/fan.pt` and `checkpoints/slider.pt` (via `--output-checkpoint` in `run_training.py`).
- **Evaluation:** Run evaluation with the same config and checkpoint for that type (e.g. `--config config_fan.yaml --checkpoint checkpoints/fan.pt`). Test set is always the same machine type as training.
- **Goal:** Test AUC-ROC ≥ 0.75 per type. See `TARGET_AUC_ROADMAP.md` for the exact pipeline → train → eval steps and success criteria.
- **Combine later:** One system can load the appropriate checkpoint by machine type (router by user choice or API parameter); no merging of networks. A unified model trained on all types is an optional future step.

---

### 1. Latest Run: What We Did

- **Data pipeline**
  - Command used: `.venv\Scripts\python.exe scripts\run_pipeline.py --clean --config config.yaml`.
  - The pipeline removed `data/processed` and regenerated the synthetic ground-truth dataset from the MIMII raw data in `data/raw/`.
  - For this run, only the **`fan`** machine type was present in `data/raw/`, so all generated samples were `machine_type = "fan"`.
  - `config.yaml` caps the total number of generated samples via `data.max_generated_samples` (currently 3000). The split ratios are 70% train / 15% val / 15% test at the **machine-id level**, and the sample-level targets respect those proportions.
  - The pipeline summary for this run reported approximately:
    - **Total samples**: 2550
    - **Split counts**: train ≈ 2100, **val = 0**, test ≈ 450
    - **Label balance**: normal ≈ 1020, anomalous ≈ 1530 (i.e., anomaly-heavy)
    - **Machine types**: fan = 2550, slider = 0

- **Training**
  - Command used: `.venv\Scripts\python.exe scripts\run_training.py --config config.yaml --run-name baseline_v2`.
  - Training uses `vehicleanomalynet/train.py`, which:
    - Builds DataLoaders from the pipeline metadata (`train` and `val` splits).
    - Constructs the `VehicleAnomalyNet` model (CNN + BiGRU + dual heads).
    - Uses `DualTaskLoss` with:
      - `BCEWithLogitsLoss` for anomaly head, with `pos_weight` taken from `training.anomaly_pos_weight` in `config.yaml` (currently 0.67).
      - `CrossEntropyLoss(ignore_index=-1)` for fault classification head; normal samples have fault label masked to `-1`.
    - Optimizer: Adam with LR and weight decay from `config.yaml` (weight decay increased to 0.0005 for regularization).
    - Dropout increased to 0.45 in the model for stronger regularization.
    - Logs metrics to MLflow, performs early stopping on **validation AUC-ROC**.
  - Because the **validation set was empty (`val` split had 0 samples)**:
    - `val_epoch` effectively sees no data; AUC stays at 0.0 by construction.
    - Early stopping criterion does not receive meaningful validation feedback.
    - Training still runs for a minimum number of epochs; train loss is driven very low (strong overfitting to the training data).

- **Evaluation**
  - Command used: `.venv\Scripts\python.exe scripts\run_evaluation.py`.
  - This script:
    - Loads the test DataLoader (split = `test`) using the same feature extraction and dataset logic as training.
    - Loads the best checkpoint from training.
    - Calls `evaluate()` from `vehicleanomalynet/evaluate.py`.
    - Saves `results/metrics.json` and plots (ROC, confusion matrix, score distribution).
    - Prints a plain-ASCII results table to the console.

---

### 2. How Metrics Are Computed (Code-Level)

- **Core evaluation function** (`vehicleanomalynet/evaluate.py`)
  - Iterates over the `test_loader`, which yields batches of:
    - `features`: log-Mel spectrogram features.
    - `anomaly_label`: 0 = normal, 1 = abnormal.
    - `fault_label`: integer index for fault type, with normal samples masked as `-1`.
    - `machine_type`: string (e.g., `"fan"` or `"slider"`).
  - For each batch:
    - Runs `model(features)` to get:
      - `anomaly_logit` (shape `[B, 1]`).
      - `fault_logits` (shape `[B, num_fault_classes]`).
    - Converts:
      - `anomaly_logit` → `score = sigmoid(logit)` as anomaly score \([0,1]\).
      - `fault_logits` → `fault_pred = argmax(dim=1)` as predicted fault class.
    - Accumulates scores, labels, fault labels/preds, and machine types into lists.
  - After the loop:
    - Concatenates all batches into full-length NumPy arrays.
    - **Anomaly AUC-ROC**
      - Checks how many unique values are in `labels`.
      - If there is **only a single class** present in `labels`, it sets `auc_roc = 0.0` to avoid `roc_auc_score` errors.
      - Otherwise computes `roc_auc_score(labels, scores)` (1D binary AUC).
    - **Threshold-based metrics**
      - Converts scores to binary predictions with a fixed threshold of 0.5.
      - Computes:
        - `precision_score(labels, preds, zero_division=0.0)`.
        - `recall_score(labels, preds, zero_division=0.0)`.
        - `confusion_matrix(labels, preds, labels=[0, 1])`.
      - If the confusion matrix is not 2×2 (e.g., due to missing labels), it manually reconstructs a 2×2 count.
    - **Fault classification F1**
      - Filters to **abnormal-only** samples (`labels == 1`) because normal samples do not have a meaningful fault class.
      - Computes macro F1 over these samples using `f1_score(fault_labels_abnormal, fault_preds_abnormal, average="macro")`.
      - If there are no abnormal samples in the test set, F1 is set to 0.0.
    - **Per-machine-type AUC**
      - For each unique `machine_type` in the test data, filters scores and labels and computes an AUC-ROC over that subset, using the same logic (0.0 if only one class present).
    - Returns a dictionary with:
      - `auc_roc`, `f1_macro`, `precision`, `recall`, `confusion_matrix`, `per_machine_auc`, plus raw arrays if needed by plotting.

- **Result serialization**
  - `scripts/run_evaluation.py` takes the `eval_dict`, converts NumPy scalars to native Python types, and writes `results/metrics.json`.
  - It also generates three plots and saves them under `results/`:
    - `roc_curve.png`
    - `confusion_matrix.png`
    - `score_distribution.png`

---

### 3. Latest Metrics and What They Mean

From `results/metrics.json` for the most recent full run:

- **Overall anomaly AUC-ROC**: ~0.494
  - This is effectively **no better than random guessing** (0.5).
  - Confirms that, on this configuration and data slice, the anomaly scores are not ranking abnormal vs. normal reliably for the test set.

- **Fault F1 (macro, abnormal-only)**: 1.000
  - Under current conditions, this indicates the model is **perfectly predicting fault classes on the abnormal samples in the test set**.
  - This is plausible because:
    - The synthetic pipeline might be producing relatively clean, easily separable fault patterns for `fan`.
    - The test set is only `fan` and might be relatively simple compared to multi-machine scenarios.
  - It also suggests that the **feature extractor + CNN+BiGRU backbone** has enough capacity to memorize patterns in the training set and generalize reasonably to faults (at least in this narrow domain).

- **Precision and recall for anomaly detection**:
  - **Precision**: ~0.596
  - **Recall**: ~0.815
  - These are computed at a fixed threshold of 0.5 and relate to the confusion matrix:
    - Confusion matrix (rows = true, columns = predicted):
      - TN = 31 (normal correctly predicted normal)
      - FP = 149 (normal predicted as abnormal)
      - FN = 50 (abnormal predicted as normal)
      - TP = 220 (abnormal correctly predicted abnormal)
  - Interpretation:
    - The model is **very recall-heavy** (it catches most anomalies) but has **poor precision** (a lot of normal sounds are flagged as abnormal).
    - Despite decent recall, the ranking of scores is not good enough to produce a strong AUC, hence the ~0.494.

- **Per-machine-type AUC**
  - Only `"fan"` appears in `per_machine_auc`:
    - `"fan"` AUC-ROC ≈ 0.494
  - There is no `"slider"` entry because no slider samples were present in the test set for this run.

---

### 4. Why We Are Getting These Scores (Root Causes)

- **1) Empty validation set → ineffective early stopping**
  - The current machine-id-level split logic in `pipeline.py`:
    - Stratifies by `machine_type` (`fan` vs `slider`).
    - Randomly assigns machine IDs for each type into train/val/test with fixed ratios (70/15/15).
  - When only `fan` is present and there are very few `fan` machine IDs in the raw dataset:
    - It is possible (and in this run, it happened) that **no machine IDs ended up in the `val` set**, especially when combined with sample caps and rounding.
    - The downstream effect is that `val` DataLoader has 0 samples; `val_epoch` cannot provide a real signal and AUC is forced to 0.0.
  - Consequences:
    - Early stopping cannot meaningfully detect overfitting or pick the best epoch.
    - The "best" checkpoint is basically arbitrary relative to true generalization.
    - The model easily overfits the training data (train loss ~0.002), but this does not translate to robust test performance.

- **2) Single machine type (fan-only) and limited diversity**
  - With only `fan` data, the model does **not see multiple machine types** (fan + slider) during training.
  - This narrows the distribution the model must learn and may actually **inflate fault F1** (one domain, synthetic faults) while not helping anomaly AUC much.
  - Also, if the number of fan machine IDs is small, the synthetic pipeline might be overusing the same underlying raw recordings for many segments and augmentations, making the training/test boundary leaky in terms of underlying patterns.

- **3) Strong class imbalance and label caps**
  - The pipeline intentionally tries to enforce `data.max_label_ratio` per split (currently 0.60) to avoid splits becoming nearly all-normal or all-abnormal.
  - However, in this run, we still ended up with:
    - Normal ≈ 1020
    - Anomalous ≈ 1530
    - So anomalies are the majority (ratio ≈ 0.6:0.4 in favor of anomalies).
  - This, combined with:
    - `anomaly_pos_weight = 0.67` in `DualTaskLoss` (which pushes the loss to treat anomalies slightly differently).
    - Threshold fixed at 0.5 at evaluation time.
  - Leads to a model that tends to predict "anomaly" frequently (high recall, low precision), hurting AUC.

- **4) Synthetic pipeline behavior and leakage risk**
  - The `GroundTruthPipeline`:
    - Iterates machine IDs, then within each ID iterates:
      - Abnormal first, then normal.
      - Multiple segments per file (sliding windows) and possible augmentations.
    - Caps the number of samples per split using `split_targets` and `split_label_caps`.
  - If raw data for each machine type/machine id is limited, many train and test samples can be derived from closely related time windows of the same underlying recording.
  - This can make it easy for the model to **memorize** training segments but still fail to achieve robust AUC if the noise and artifacts across training vs. test segments are similar or if the anomaly boundary is not clearly defined.

- **5) Regularization helps but does not fix data issues**
  - We increased:
    - Dropout (0.3 → 0.45).
    - Weight decay (0.0001 → 0.0005).
  - These changes reduce overfitting somewhat at the model level, but they **cannot fix the structural data problems**:
    - Empty validation set.
    - Limited machine-id diversity and single machine type.
    - Potential label imbalance and threshold mismatch.

---

### 5. High-Level Takeaways for Future Planning

- **Data/split issues are currently the main bottleneck**, not model capacity.
  - Evidence: perfect fault F1 on abnormal samples (model can learn patterns) but near-random AUC for anomaly detection.
  - Empty validation split prevents meaningful hyperparameter tuning or early stopping.

- **To move AUC-ROC toward the Milestone 6 target (> 0.75), we will likely need:**
  - More diverse raw data:
    - Include additional MIMII machine types (e.g., `slider`) at 0 dB.
    - Ensure multiple machine IDs per type so that each split (train/val/test) has enough unique machines.
  - Safer splitting:
    - Explicitly guarantee at least one machine ID per split per machine type (even under sample caps).
    - Consider per-file-level splitting or stratification by `(machine_type, machine_id)` with fallback rules when counts are small.
  - Validation set health checks:
    - After generating metadata, assert that each split has:
      - Non-zero samples.
      - Both classes (normal and abnormal) wherever possible.
  - Potential threshold tuning:
    - Instead of a fixed 0.5 threshold, we can derive an operating point from ROC or Precision-Recall curves (e.g., maximizing F1).

- **This document is a snapshot:**
  - It describes the current behavior (fan-only, empty val, AUC ~0.494).
  - As we add more data, adjust splitting logic, and re-run experiments, we should either:
    - Update this file with new findings, or
    - Create `M6_FINDINGS_v2.md` / `M7_FINDINGS.md` as the project progresses.

