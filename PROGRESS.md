# VehicleAnomalyNet — Build Progress

## Phase 0: Planning

- [ ] ARCHITECTURE.md written and reviewed
- [ ] Project structure scaffolded
- [ ] MIMII dataset downloaded and verified

---

## Milestone 0 — Planning & Scaffolding

**Goal:** Architecture decided, repo scaffolded, no code written yet.

**Tasks:**
- [ ] Read full PRD
- [ ] Write `ARCHITECTURE.md` (answer every question in the Planning Mode section)
- [ ] Create full directory structure with empty `__init__.py` files
- [ ] Write `requirements.txt` based on architecture decisions
- [ ] Write `config.yaml` with all hyperparameters (fill in values after architecture decisions)
- [ ] Create `PROGRESS.md` with all milestones as `[ ]`

**Acceptance Criterion:**  
`ARCHITECTURE.md` exists and answers every question. `PROGRESS.md` exists. `python -c "import vehicleanomalynet"` runs without error.

---

## Milestone 1 — Data Download & EDA

**Goal:** MIMII fan + slider data is local, visualized, understood.

**Tasks:**
- [ ] Write `scripts/download_data.py` that:
  - [ ] Downloads fan (0dB) and slider (0dB) from Zenodo using `requests` or `urllib`
  - [ ] Verifies file checksums
  - [ ] Extracts to `data/raw/`
  - [ ] Prints dataset statistics (n_normal, n_abnormal, duration per class)
- [ ] Write `notebooks/01_eda.ipynb` that:
  - [ ] Loads 5 normal + 5 abnormal samples for fan and slider
  - [ ] Plots raw waveforms side by side (normal vs. abnormal)
  - [ ] Plots log-Mel spectrograms side by side
  - [ ] Plots class distribution bar chart
  - [ ] Computes and prints: mean duration, sample rate, min/max amplitude per class
  - [ ] Saves all plots to `notebooks/figures/`

**Acceptance Criterion:**  
`python scripts/download_data.py` completes without error. Notebook runs top-to-bottom. You can visually distinguish normal vs. abnormal spectrograms in the plots.

---

## Milestone 2 — Ground Truth Data Generation Pipeline

**Goal:** Automated pipeline that generates labeled, augmented dataset with metadata CSV.

**Tasks:**
- [ ] Implement `vehicleanomalynet/pipeline.py` with class `GroundTruthPipeline`
- [ ] Write `scripts/run_pipeline.py` that instantiates and runs the pipeline
- [ ] Pipeline must produce `data/processed/metadata.csv`
- [ ] Pipeline must print a summary: total samples, class balance, split sizes

**Requirements:**
- [ ] SNR mixing must be mathematically correct (RMS-based, not amplitude-based)
- [ ] Augmentations: time-stretch (±5%), gain jitter (±3dB), add mild Gaussian noise
- [ ] Stratified train/val/test split: 70/15/15, split by `machine_id` (not random across machines — this tests generalization)
- [ ] Save processed WAVs as 16kHz mono float32

**Acceptance Criterion:**  
`python scripts/run_pipeline.py` completes. `data/processed/metadata.csv` exists with correct columns. Class balance printed. At least 3,000 total samples generated.

---

## Milestone 3 — Feature Extraction

**Goal:** Audio → tensor pipeline, fast and deterministic.

**Tasks:**
- [ ] Implement `vehicleanomalynet/features.py`
- [ ] Implement `vehicleanomalynet/dataset.py`
- [ ] Write a unit test (inline, run with `python -m pytest` or just `python`) that:
  - [ ] Loads 10 samples from dataset
  - [ ] Prints feature shape, label distribution, min/max values
  - [ ] Asserts feature tensor has no NaN or Inf values
  - [ ] Asserts all features have identical shape

**Acceptance Criterion:**  
Feature shapes are consistent. No NaN/Inf in features. Dataset `__getitem__` returns correct dtypes. Unit test passes.

---

## Milestone 4 — Model Architecture

**Goal:** VehicleAnomalyNet built, forward pass verified, parameter count logged.

**Tasks:**
- [ ] Implement `vehicleanomalynet/model.py` with `VehicleAnomalyNet`
- [ ] Implement `vehicleanomalynet/losses.py` with `DualTaskLoss`
- [ ] Write a model smoke test:
  - [ ] Instantiates model from config
  - [ ] Runs a forward pass with dummy batch
  - [ ] Prints: total parameters, output shapes, loss value
  - [ ] Asserts output shapes match expected

**Acceptance Criterion:**  
`python -c "from vehicleanomalynet.model import VehicleAnomalyNet"` works. Forward pass on dummy batch succeeds. Loss is a finite scalar. Parameter count printed.

---

## Milestone 5 — Training Loop

**Goal:** Full training run completes, model improves over baseline, MLflow tracks everything.

**Tasks:**
- [ ] Implement `vehicleanomalynet/train.py` with `Trainer` class
- [ ] Training must include:
  - [ ] AdamW optimizer
  - [ ] Cosine annealing LR scheduler
  - [ ] Early stopping (patience from config)
  - [ ] Best model checkpoint saving (by val AUC-ROC)
  - [ ] MLflow run logging: all config params + per-epoch metrics
  - [ ] Progress bar via `tqdm`
  - [ ] Print per-epoch summary: `Epoch X | Train Loss: X.XX | Val AUC: X.XX | Val F1: X.XX | LR: X.XXXXX`
- [ ] Write `scripts/run_training.py` as CLI entry point:
  - [ ] `python scripts/run_training.py --config config.yaml --run-name "baseline_v1"`

**Acceptance Criterion:**  
Training runs for at least 5 epochs without error. Val AUC-ROC improves vs. epoch 1. Checkpoint saved. MLflow `mlruns/` directory populated. `mlflow ui` shows the run.

---

## Milestone 6 — Evaluation

**Goal:** Complete quantitative results on held-out test set. Numbers ready for resume.

**Tasks:**
- [ ] Implement `vehicleanomalynet/evaluate.py` with `evaluate()` + `plot_results()`
- [ ] Produce these output files in `results/`:
  - [ ] `metrics.json` — all numeric results
  - [ ] `roc_curve.png` — ROC curve with AUC annotated
  - [ ] `confusion_matrix.png` — labeled confusion matrix
  - [ ] `score_distribution.png` — histogram of anomaly scores (normal vs. abnormal)
- [ ] Print final results table (as specified in PRD)

**Acceptance Criterion:**  
`results/metrics.json` exists. All four plots saved. Test AUC-ROC > 0.75 (if below, debug before proceeding). Per-machine breakdown populated.

---

## Milestone 7 — ONNX Export & Latency Benchmark

**Goal:** Production-ready model export with measured inference latency.

**Tasks:**
- [ ] Implement `scripts/export_onnx.py`:
  - [ ] Export trained model to `vehicleanomalynet.onnx`
  - [ ] Verify ONNX model output matches PyTorch model output (within 1e-4 tolerance)
  - [ ] Benchmark latency: 200 warm-up runs + 500 timed runs
  - [ ] Report: mean latency, std, p95, p99
  - [ ] Test batch sizes: 1, 8, 32
- [ ] Print benchmark table (as specified in PRD)
- [ ] Add latency numbers to `results/metrics.json`

**Acceptance Criterion:**  
`vehicleanomalynet.onnx` exists. PyTorch vs. ONNX output diff < 1e-4. Latency benchmark table printed. Mean batch=1 latency saved to metrics.

---

## Milestone 8 — Streamlit Dashboard

**Goal:** Interactive diagnostic UI that a non-technical NVH team member could use.

**Tasks:**
- [ ] Implement `dashboard.py` with these panels:
  - [ ] Panel 1 — File Upload & Inference
  - [ ] Panel 2 — Spectrogram Viewer
  - [ ] Panel 3 — Model Performance
  - [ ] Panel 4 — Experiment History (MLflow)
- [ ] Dashboard must run with: `streamlit run dashboard.py`
- [ ] No hardcoded paths — all paths from `config.yaml`

**Acceptance Criterion:**  
`streamlit run dashboard.py` opens without error. Upload a test WAV → anomaly score appears. All 4 panels render. No hardcoded paths.

---

## Milestone 9 — README & Cleanup

**Goal:** GitHub-ready repo a recruiter can run in 10 minutes.

**Tasks:**
- [ ] Write `README.md` with these sections:
  - [ ] Overview
  - [ ] Motivation
  - [ ] Architecture
  - [ ] Results table
  - [ ] Quick Start
  - [ ] Project Structure
  - [ ] Dataset
- [ ] Final cleanup:
  - [ ] `requirements.txt` has every dependency with pinned versions
  - [ ] `config.yaml` has no magic numbers — every hyperparameter is there
  - [ ] Remove all debug print statements
  - [ ] All functions have docstrings
  - [ ] `.gitignore` excludes `data/`, `mlruns/`, `__pycache__/`, `*.onnx`
  - [ ] `PROGRESS.md` has all milestones marked `[x]`

**Acceptance Criterion:**  
Fresh clone + `pip install -r requirements.txt` + follow README reproduces the project. All PROGRESS.md items `[x]`. README has real numbers in the results table.

