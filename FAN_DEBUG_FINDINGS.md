# Fan vs Slider: Debug Findings (Low Fan AUC)

This document summarizes why the **fan** model achieves test AUC ~0.40 and predicts almost every sample as "anomaly" (TN=0, FP=180, FN=10, TP=260), while **slider** achieves AUC ~0.74 with a more balanced confusion matrix (TN=73, TP=224, FP=107, FN=46). It is based on code and config inspection only (no training or evaluation was run).

---

## 1. Score Distribution and Metrics

### What the fan score distribution implies

- **Fan confusion matrix** (at default threshold 0.5): TN=0, FP=180, FN=10, TP=260.
  - All **180 normal** test samples have anomaly score ≥ 0.5 (all predicted anomaly).
  - **260 anomaly** samples have score ≥ 0.5 (correct); **10 anomaly** samples have score < 0.5 (wrong).
- So: **normal and abnormal scores overlap on the high side**. The model gives high scores to almost everyone; the few lower scores are (wrongly) mostly anomaly. That produces **AUC ≈ 0.40** (< 0.5): on average, a randomly chosen normal has a *higher* score than a randomly chosen abnormal, i.e. the ranking is inverted or nearly random.
- **If `results_fan/score_distribution.png` exists**, it would show: green (normal) and red (abnormal) histograms both concentrated at **high anomaly scores** (e.g. 0.5–1.0), with a small tail of abnormal scores in the lower range. So the plot would show **no separation** between normal and abnormal; both classes sit in the high-score region.

### Evaluation code (`vehicleanomalynet/evaluate.py`)

- Anomaly score = `sigmoid(anomaly_logit)`; binary prediction = `score >= 0.5`.
- Precision, recall, and confusion matrix use this **fixed 0.5** threshold (lines 92–95).
- The code also computes **Youden’s J** (threshold maximizing TPR − FPR) and **best F1 threshold** over ROC thresholds (lines 76–90) and returns them in the eval dict, but **they are not written to `metrics.json`** (see `scripts/run_evaluation.py`): only `auc_roc`, `f1_macro`, `precision`, `recall`, `per_machine_auc`, and `confusion_matrix` are saved. So for fan, we cannot see from saved metrics what a Youden or F1-optimal threshold would be.

### Fan vs slider

- **Slider**: TN=73, TP=224 → many normals get score < 0.5 and many anomalies get score ≥ 0.5; scores separate the two classes reasonably (AUC 0.74).
- **Fan**: TN=0, TP=260 → normals are never below 0.5; scores do not separate. So the main difference is that **fan scores are collapsed high** (no separation), while **slider scores spread across the range** and rank normal vs anomaly correctly.

---

## 2. Data: Pipeline, Balance, and Splits

### How fan data is produced (`vehicleanomalynet/pipeline.py`)

- **Machine types**: `config_fan.yaml` sets `machine_types: ["fan"]`, so only `data/raw/fan` is used. Same pattern for slider with `config_slider.yaml` and `data/raw/slider`.
- **Splits**: Train/val/test are stratified **by machine_id** (no machine_id in more than one split). For each type, IDs under `raw_dir/<machine_type>/` (e.g. `id_00`, `id_01`) are shuffled and assigned to train/val/test by ratios `train_split` (0.70), `val_split` (0.15), `test_split` (0.15).
- **Label balance**: `max_label_ratio: 0.60` caps each label’s share per split at 60% of that split’s target. With `max_generated_samples: 3000`, split targets are train 2100, val 450, test 450; each label is capped at 60% of those (e.g. 1260 per label for train). The pipeline iterates **abnormal first**, then normal, so anomaly hits the cap first; with typical MIMII data you get **train ~60% anomaly** (e.g. 840 normal, 1260 anomaly), as in your setup.
- **Config**: Fan and slider configs are the same in structure (`max_label_ratio: 0.60`, `max_generated_samples: 3000`, same split ratios). So the difference is not from config values but from **data volume and machine_id count** per type.

### Number of machine IDs and diversity

- Pipeline collects machine IDs with `type_root.iterdir()` for dirs named `id_*`. **If fan has fewer such IDs than slider**, then:
  - Fan has **less machine diversity** (fewer distinct devices).
  - With **one ID**: all samples go to train (no val/test).
  - With **two IDs**: one train, one val; no test.
  - With **three or more**: train/val/test each get at least one ID.
- So **fewer fan IDs** would mean: (1) less diversity, (2) possible overfitting to one or two machines, (3) test set from very few machines. You can confirm by checking `data/raw/fan` (count `id_*` dirs) and comparing to `data/raw/slider`, and by inspecting `data/processed_fan/metadata.csv` for `machine_id` and `split` counts.

### Split leakage

- There is **no sample-level leakage**: each machine_id is in exactly one split. So the same machine never appears in both train and test.
- **“Same-machine heavy”** means: if fan has only a few IDs, then train (and val/test) are each dominated by one or two machines; the model can overfit to those and fail to generalize to the test machine(s). That is a **diversity** issue, not leakage.

---

## 3. Training and Loss

### How `anomaly_pos_weight` is used

- **`vehicleanomalynet/losses.py`**: `DualTaskLoss` passes `anomaly_pos_weight` into `BCEWithLogitsLoss(pos_weight=...)`. In PyTorch, `pos_weight` is the weight on the **positive (anomaly)** class.
- **`vehicleanomalynet/train.py`**: Reads `anomaly_pos_weight` from config (0.67 in both configs) and passes it into `DualTaskLoss` (lines 54–61).
- With **train ~840 normal, 1260 anomaly**, a balanced total loss contribution would use `pos_weight = 840/1260 ≈ 0.67`. So **0.67 is appropriate** for balancing class counts; it does **not** by itself push the model to predict anomaly more. The problem is that **despite this**, the fan model still predicts almost everything as anomaly and gives overlapping high scores, so the failure is in **representation or generalization**, not in a simple class-weight bias toward anomaly.

### Why fan’s best val AUC might be at epoch 1 and then decrease

- **Overfitting**: With fewer fan machine IDs and less diversity, the model can fit the training machines quickly; val (and test) may come from different machines and get worse as training continues.
- **Capacity / regularization**: Same capacity and dropout (0.45) for fan and slider; if fan has less data diversity, the same model may overfit more on fan. So **increasing dropout** or **reducing capacity** for fan could help.
- **Learning rate**: Same LR (0.001) and CosineAnnealingLR for both; no evidence it’s wrong, but early overfitting can make the best val AUC appear at epoch 1 if the first epoch generalizes slightly and later epochs overfit.
- **Early stopping**: Best checkpoint is by **validation AUC**; if the best val AUC is at epoch 1, the saved fan model is that early checkpoint. So we are already using “best by val AUC at epoch 1” in practice; the issue is that even that best checkpoint has poor score separation (and low test AUC).

---

## 4. Evaluation: Threshold and Youden

- **Default 0.5**: Used for precision, recall, and confusion matrix. For fan, this yields TN=0 and very high recall (almost all predicted anomaly).
- **Youden / F1-optimal**: `evaluate.py` computes `youden_j_threshold` and `best_f1_threshold` but they are **not** saved in `metrics.json`. For a model with collapsed high scores, a different threshold would not fix **AUC** (ranking is already bad). But **reporting** Youden and F1-optimal thresholds (and optionally using one for deployment) would help interpret fan and might give a slightly better precision/recall trade-off; the main fix remains data/training.

---

## 5. Root Causes (Summary)

- **Score distribution collapsed**: Normal and abnormal fan samples both get high anomaly scores; there is no separation, so AUC is low (~0.40) and at 0.5 almost everything is predicted anomaly (TN=0).
- **Likely fewer fan machine IDs**: If `data/raw/fan` has fewer `id_*` directories than slider, fan has less diversity and is more prone to overfitting to one or two machines, so scores do not generalize to test.
- **Train balance and pos_weight**: Train is ~60% anomaly; pos_weight 0.67 correctly balances the loss. The issue is not “predict anomaly because of imbalance” but that the model fails to learn a discriminative pattern for fan (or overfits and loses it on test).
- **Best val at epoch 1**: Suggests quick overfitting on fan; the chosen checkpoint is already the “least overfit” one, but it still has poor score separation.

---

## 6. Fix Recommendations (Ordered)

1. **Set `anomaly_pos_weight` from actual train class ratio (e.g. neg/pos)**  
   In the training script or config builder, compute train normal vs anomaly counts (from metadata or dataloader), set `pos_weight = count_normal / count_anomaly` for that run (or allow config override). For fan this will be ~0.67; for slider possibly different. This keeps loss balanced even if pipeline balance changes.

2. **Increase regularization for fan**  
   In `config_fan.yaml`: try **higher dropout** (e.g. 0.50–0.55) and/or **smaller capacity** (e.g. `cnn_channels: [24, 48, 96]`, `gru_hidden: 192`) to reduce overfitting when fan has fewer machines.

3. **Confirm and, if needed, increase fan data diversity**  
   Check `data/raw/fan` (number of `id_*` dirs) and `data/processed_fan/metadata.csv` (machine_id per split). If fan has only 2–3 IDs, consider adding more MIMII fan machines or stronger augmentation (e.g. more augment_factor, or extra augmentations in `pipeline.py`) so the model sees more variation.

4. **Early stopping and minimum epochs**  
   Optionally add a **minimum epochs** (e.g. 5) before early stopping so the first epoch is not chosen by default when val is noisy. Also consider **longer patience** for fan (e.g. 15) if val is small or from one machine, so a brief val AUC dip does not stop too early.

5. **Report Youden and F1-optimal threshold for fan**  
   In `scripts/run_evaluation.py`, add to the saved `metrics` dict: `best_f1_threshold`, `best_f1`, `youden_j_threshold`, `youden_j` from `eval_dict`. Optionally, for fan-only runs, log or use the Youden threshold for the reported confusion matrix so that precision/recall are interpretable even when 0.5 is poor. This does not fix AUC but improves interpretability and deployment choice.

---

## Suggested config edits (for reference; do not run)

- **config_fan.yaml**  
  - `training.anomaly_pos_weight`: leave at 0.67 or set via script from train ratio.  
  - `model.dropout`: e.g. `0.52`.  
  - `model.cnn_channels`: e.g. `[24, 48, 96]` (optional).  
  - `training.early_stopping_patience`: e.g. `15`.

- **scripts/run_evaluation.py**  
  - When building `metrics`, add:  
    `"best_f1_threshold": float(eval_dict["best_f1_threshold"])`,  
    `"best_f1": float(eval_dict["best_f1"])`,  
    `"youden_j_threshold": float(eval_dict["youden_j_threshold"])`,  
    `"youden_j": float(eval_dict["youden_j"])`.
