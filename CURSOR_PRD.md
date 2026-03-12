# VehicleAnomalyNet — Cursor Agent PRD
## Mechanical Sound Anomaly Detection for Automotive Diagnostics

---

## 🤖 AGENT INSTRUCTIONS — READ THIS FIRST

You are a senior ML engineer. Before writing a single line of code, you must:

1. **Enter Planning Mode** — read this entire PRD, then produce a written `ARCHITECTURE.md` file outlining your technical decisions before touching any implementation files
2. **Create `PROGRESS.md`** — a living checklist you update every time you complete a milestone. Mark tasks `[ ]` todo, `[~]` in-progress, `[x]` done
3. **Ask clarifying questions** if any requirement is ambiguous before building
4. **Never skip a milestone** — each one has an acceptance criterion you must verify passes before moving on

Your deliverable is a working, GitHub-ready ML project. Every file should be production-quality: typed, documented, and importable.

---

## 📋 PLANNING MODE — DO THIS BEFORE ANY CODE

Before creating any `.py` files, create two files:

### File 1: `ARCHITECTURE.md`
Write your answers to every question below. This is your contract for what you're building.

```
Questions to answer in ARCHITECTURE.md:

FRAMEWORK DECISIONS
- Which audio feature library will you use: torchaudio, librosa, or both? Why?
- What input representation will you use: log-Mel spectrogram, MFCC, raw waveform, or multi-feature? Justify with tradeoffs.
- What is the input tensor shape to the model? (batch, channels, height, width)?
- Will you use a pretrained backbone (e.g., CNN14 from PANNs, AST) or train from scratch? Why?
- What framework for experiment tracking: MLflow locally, or Weights & Biases?
- What is the ONNX export strategy for edge inference?

ARCHITECTURE DECISIONS
- Describe the full model architecture in prose: input → backbone → temporal module → output heads
- Why CNN + recurrent vs. pure CNN vs. pure Transformer for this task?
- How will you handle the dual-task objective (anomaly detection + fault classification)?
- How will you handle class imbalance (anomalies are rare)?
- What loss function(s) and why?

DATA DECISIONS
- Which MIMII machine types will you use and why?
- How will you structure the train/val/test splits?
- What augmentation strategy will you use and why is it valid for audio anomaly detection?
- How will you generate the synthetic ground truth data (SNR mixing)?

EVALUATION DECISIONS
- What is your primary metric and why (AUC-ROC, F1, accuracy)?
- What does "done" look like numerically for this project?

DEPLOYMENT DECISIONS  
- What is the latency target for ONNX inference?
- What does the Streamlit dashboard need to show to be useful?
```

### File 2: `PROGRESS.md`
Create this file with all milestones pre-populated as `[ ]`. Update it as you build. Format:

```markdown
# VehicleAnomalyNet — Build Progress

## Phase 0: Planning
[ ] ARCHITECTURE.md written and reviewed
[ ] Project structure scaffolded
[ ] MIMII dataset downloaded and verified

## Phase 1: Data Pipeline
...etc (copy from milestones below)
```

---

## 🎯 PROJECT GOAL

Build **VehicleAnomalyNet**: an end-to-end audio ML system that detects anomalous mechanical sounds and classifies fault types from machine audio. Uses the MIMII industrial machine dataset — the academic benchmark closest to Tesla NVH's in-vehicle diagnostic audio work.

**Why this matters to Tesla NVH:**
- Tesla's NVH team builds audio algorithms for vehicle diagnostics
- Tesla's NVH team builds automated ground truth data generation pipelines  
- MIMII = rotating mechanical machinery (fans, bearings, slide rails) = direct analog to EV drivetrain / HVAC / motor components

---

## 📦 DATASET

**MIMII Dataset (Malfunctioning Industrial Machine Investigation & Inspection)**
- Download: https://zenodo.org/record/3384388
- License: Creative Commons Attribution 4.0
- Use machine types: **`fan`** and **`slider`** only (keeps scope manageable, ~3GB)
- Format: WAV, 16kHz, 16-bit, single-channel
- Structure per machine:
  - `normal/` — clean operating sounds (thousands of 10s clips)
  - `abnormal/` — faulty sounds (hundreds of 10s clips per fault type)
  - SNR variants: `-6dB`, `0dB`, `+6dB` noise mixing already provided

**Fault types you will detect:**
- Fan: `rotating_imbalance`, `contamination`, `voltage_change`  
- Slider: `rail_damage`, `no_grease`, `loose_screws`
- Plus: `normal` class (label = 0 for anomaly head, no fault class)

---

## 🏗️ REQUIRED FILE STRUCTURE

The agent must create exactly this structure (no extra files, no missing files):

```
vehicleanomalynet/
│
├── ARCHITECTURE.md          ← Written FIRST in planning mode
├── PROGRESS.md              ← Living checklist, updated throughout
├── README.md                ← Written LAST with final results
├── requirements.txt         ← All dependencies pinned
├── config.yaml              ← All hyperparameters, no magic numbers in code
│
├── data/
│   ├── raw/                 ← MIMII downloads go here (gitignored)
│   └── processed/           ← Pipeline output (gitignored)
│
├── vehicleanomalynet/       ← Main package (has __init__.py)
│   ├── __init__.py
│   ├── pipeline.py          ← Ground truth data generation pipeline
│   ├── features.py          ← Feature extraction (mel, mfcc, contrast)
│   ├── dataset.py           ← PyTorch Dataset + DataLoader factory
│   ├── model.py             ← VehicleAnomalyNet architecture
│   ├── losses.py            ← Combined loss function
│   ├── train.py             ← Training loop
│   └── evaluate.py          ← Metrics: AUC-ROC, F1, confusion matrix
│
├── scripts/
│   ├── download_data.py     ← Automates MIMII download from Zenodo
│   ├── run_pipeline.py      ← Runs full data generation pipeline
│   ├── run_training.py      ← Entry point for training
│   └── export_onnx.py       ← ONNX export + latency benchmark
│
├── notebooks/
│   └── 01_eda.ipynb         ← EDA: waveform viz, spectrogram analysis, class distribution
│
└── dashboard.py             ← Streamlit diagnostic UI
```

---

## 🔨 MILESTONES

Work through these in order. Do not start a milestone until the previous one is verified.

---

### MILESTONE 0 — Planning & Scaffolding
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

**Update PROGRESS.md when done.**

---

### MILESTONE 1 — Data Download & EDA
**Goal:** MIMII fan + slider data is local, visualized, understood.

**Tasks:**
- [ ] Write `scripts/download_data.py` that:
  - Downloads fan (0dB) and slider (0dB) from Zenodo using `requests` or `urllib`
  - Verifies file checksums
  - Extracts to `data/raw/`
  - Prints dataset statistics (n_normal, n_abnormal, duration per class)
- [ ] Write `notebooks/01_eda.ipynb` that:
  - Loads 5 normal + 5 abnormal samples for fan and slider
  - Plots raw waveforms side by side (normal vs. abnormal)
  - Plots log-Mel spectrograms side by side
  - Plots class distribution bar chart
  - Computes and prints: mean duration, sample rate, min/max amplitude per class
  - Saves all plots to `notebooks/figures/`

**Acceptance Criterion:**  
`python scripts/download_data.py` completes without error. Notebook runs top-to-bottom. You can visually distinguish normal vs. abnormal spectrograms in the plots.

**Update PROGRESS.md when done.**

---

### MILESTONE 2 — Ground Truth Data Generation Pipeline
**Goal:** Automated pipeline that generates labeled, augmented dataset with metadata CSV.

**Tasks:**
- [ ] Implement `vehicleanomalynet/pipeline.py` with class `GroundTruthPipeline`:

```python
class GroundTruthPipeline:
    """
    Automated ground truth data generation pipeline.
    Produces high-quality labeled datasets for acoustic perception problems.
    """
    def __init__(self, raw_dir: str, output_dir: str, config: dict): ...
    
    def mix_at_snr(self, signal: np.ndarray, noise: np.ndarray, 
                   target_snr_db: float) -> np.ndarray:
        """Mix clean machine audio with background noise at exact target SNR."""
        ...
    
    def augment_waveform(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Apply time-stretch, pitch-shift, gain normalization."""
        ...
    
    def segment_audio(self, waveform: np.ndarray, sr: int, 
                      segment_len_s: float) -> List[np.ndarray]:
        """Segment long recordings into fixed-length clips."""
        ...
    
    def generate(self, snr_levels: List[float], augment_factor: int) -> pd.DataFrame:
        """
        Main entry point. Returns metadata DataFrame and saves processed WAVs.
        
        Returns DataFrame with columns:
        [filepath, label, machine_type, machine_id, snr_db, fault_type, split]
        
        label: 0=normal, 1=anomalous
        split: train/val/test (stratified by machine_id and fault_type)
        """
        ...
```

- [ ] Write `scripts/run_pipeline.py` that instantiates and runs the pipeline
- [ ] Pipeline must produce `data/processed/metadata.csv`
- [ ] Pipeline must print a summary: total samples, class balance, split sizes

**Requirements:**
- SNR mixing must be mathematically correct (RMS-based, not amplitude-based)
- Augmentations: time-stretch (±5%), gain jitter (±3dB), add mild Gaussian noise
- Stratified train/val/test split: 70/15/15, split by `machine_id` (not random across machines — this tests generalization)
- Save processed WAVs as 16kHz mono float32

**Acceptance Criterion:**  
`python scripts/run_pipeline.py` completes. `data/processed/metadata.csv` exists with correct columns. Class balance printed. At least 3,000 total samples generated.

**Update PROGRESS.md when done.**

---

### MILESTONE 3 — Feature Extraction
**Goal:** Audio → tensor pipeline, fast and deterministic.

**Tasks:**
- [ ] Implement `vehicleanomalynet/features.py`:

```python
def extract_log_mel(
    waveform: torch.Tensor, 
    sr: int,
    n_mels: int,       # from config
    n_fft: int,        # from config  
    hop_length: int,   # from config
    f_min: float,      # from config
    f_max: float       # from config
) -> torch.Tensor:
    """Returns log-Mel spectrogram of shape (n_mels, T)."""
    ...

def extract_mfcc_delta(
    waveform: torch.Tensor,
    sr: int, 
    n_mfcc: int        # from config
) -> torch.Tensor:
    """Returns [MFCC; delta; delta-delta] concatenated, shape (3*n_mfcc, T)."""
    ...

def pad_or_truncate(
    feature: torch.Tensor, 
    target_frames: int  # from config
) -> torch.Tensor:
    """Pad with zeros or truncate time axis to fixed length."""
    ...
```

- [ ] Implement `vehicleanomalynet/dataset.py`:

```python
class MIMIIDataset(Dataset):
    def __init__(self, metadata_df: pd.DataFrame, config: dict, split: str): ...
    def __len__(self): ...
    def __getitem__(self, idx) -> dict:
        # Returns: {"features": Tensor, "anomaly_label": int, 
        #           "fault_label": int, "filepath": str}
        ...

def get_dataloaders(metadata_path: str, config: dict) -> dict:
    """Returns {"train": DataLoader, "val": DataLoader, "test": DataLoader}"""
    ...
```

- [ ] Write a unit test (inline, run with `python -m pytest` or just `python`) that:
  - Loads 10 samples from dataset
  - Prints feature shape, label distribution, min/max values
  - Asserts feature tensor has no NaN or Inf values
  - Asserts all features have identical shape

**Acceptance Criterion:**  
Feature shapes are consistent. No NaN/Inf in features. Dataset `__getitem__` returns correct dtypes. Unit test passes.

**Update PROGRESS.md when done.**

---

### MILESTONE 4 — Model Architecture
**Goal:** VehicleAnomalyNet built, forward pass verified, parameter count logged.

**Tasks:**
- [ ] Implement `vehicleanomalynet/model.py` with `VehicleAnomalyNet`:
  - CNN backbone: ≥3 conv blocks with BatchNorm + MaxPool
  - Temporal module: BiGRU (or per ARCHITECTURE.md decision)
  - Dual output heads:
    - `anomaly_head`: outputs scalar logit (BCE loss)
    - `fault_head`: outputs class logits (CrossEntropy loss)
  - Must use `nn.Dropout` for regularization
  - Must have a `get_embedding(x)` method that returns the pooled feature vector before the heads (useful for future analysis)

- [ ] Implement `vehicleanomalynet/losses.py`:

```python
class DualTaskLoss(nn.Module):
    """
    Combined anomaly detection + fault classification loss.
    Ignores fault_head loss for normal samples (they have no fault type).
    """
    def __init__(self, anomaly_weight: float, fault_weight: float): ...
    def forward(self, anomaly_logit, fault_logit, 
                anomaly_label, fault_label) -> dict:
        # Returns: {"total": ..., "anomaly": ..., "fault": ...}
        ...
```

- [ ] Write a model smoke test:
  - Instantiates model from config
  - Runs a forward pass with dummy batch
  - Prints: total parameters, output shapes, loss value
  - Asserts output shapes match expected

**Acceptance Criterion:**  
`python -c "from vehicleanomalynet.model import VehicleAnomalyNet"` works. Forward pass on dummy batch succeeds. Loss is a finite scalar. Parameter count printed.

**Update PROGRESS.md when done.**

---

### MILESTONE 5 — Training Loop
**Goal:** Full training run completes, model improves over baseline, MLflow tracks everything.

**Tasks:**
- [ ] Implement `vehicleanomalynet/train.py` with `Trainer` class:

```python
class Trainer:
    def __init__(self, model, dataloaders, config): ...
    
    def train_epoch(self) -> dict:
        """Returns {"loss": float, "anomaly_loss": float, "fault_loss": float}"""
        ...
    
    def val_epoch(self) -> dict:
        """Returns {"loss": float, "auc_roc": float, "f1": float}"""
        ...
    
    def fit(self) -> dict:
        """Full training loop. Returns best val metrics."""
        ...
    
    def save_checkpoint(self, path: str): ...
    def load_checkpoint(self, path: str): ...
```

- [ ] Training must include:
  - AdamW optimizer
  - Cosine annealing LR scheduler
  - Early stopping (patience from config)
  - Best model checkpoint saving (by val AUC-ROC)
  - MLflow run logging: all config params + per-epoch metrics
  - Progress bar via `tqdm`
  - Print per-epoch summary: `Epoch X | Train Loss: X.XX | Val AUC: X.XX | Val F1: X.XX | LR: X.XXXXX`

- [ ] Write `scripts/run_training.py` as CLI entry point:
  ```
  python scripts/run_training.py --config config.yaml --run-name "baseline_v1"
  ```

**Acceptance Criterion:**  
Training runs for at least 5 epochs without error. Val AUC-ROC improves vs. epoch 1. Checkpoint saved. MLflow `mlruns/` directory populated. `mlflow ui` shows the run.

**Update PROGRESS.md when done.**

---

### MILESTONE 6 — Evaluation
**Goal:** Complete quantitative results on held-out test set. Numbers ready for resume.

**Tasks:**
- [ ] Implement `vehicleanomalynet/evaluate.py`:

```python
def evaluate(model, test_loader, device) -> dict:
    """
    Full evaluation on test set.
    Returns: {
        "auc_roc": float,              # Primary metric
        "f1_macro": float,             # Fault classification
        "precision": float,
        "recall": float,
        "per_machine_auc": dict,       # AUC per machine_id — shows robustness
        "confusion_matrix": np.ndarray,
        "anomaly_scores": np.ndarray,  # Raw scores for ROC curve plotting
        "labels": np.ndarray
    }
    """
    ...

def plot_results(eval_dict: dict, output_dir: str):
    """Saves: ROC curve, confusion matrix, anomaly score histogram."""
    ...
```

- [ ] Produce these output files in `results/`:
  - `metrics.json` — all numeric results (this goes on your resume)
  - `roc_curve.png` — ROC curve with AUC annotated
  - `confusion_matrix.png` — labeled confusion matrix
  - `score_distribution.png` — histogram of anomaly scores (normal vs. abnormal)

- [ ] Print final results table:
  ```
  ┌─────────────────────────────────────────┐
  │         VehicleAnomalyNet Results       │
  ├──────────────────────┬──────────────────┤
  │ Test AUC-ROC         │ 0.XXX            │
  │ Test F1 (macro)      │ 0.XXX            │
  │ Test Precision       │ 0.XXX            │
  │ Test Recall          │ 0.XXX            │
  ├──────────────────────┼──────────────────┤
  │ Fan AUC-ROC          │ 0.XXX            │
  │ Slider AUC-ROC       │ 0.XXX            │
  └──────────────────────┴──────────────────┘
  ```

**Acceptance Criterion:**  
`results/metrics.json` exists. All four plots saved. Test AUC-ROC > 0.75 (if below, debug before proceeding). Per-machine breakdown populated.

**Update PROGRESS.md when done.**

---

### MILESTONE 7 — ONNX Export & Latency Benchmark
**Goal:** Production-ready model export with measured inference latency.

**Tasks:**
- [ ] Implement `scripts/export_onnx.py`:
  - Export trained model to `vehicleanomalynet.onnx`
  - Verify ONNX model output matches PyTorch model output (within 1e-4 tolerance)
  - Benchmark latency: 200 warm-up runs + 500 timed runs
  - Report: mean latency, std, p95, p99
  - Test batch sizes: 1, 8, 32

- [ ] Print benchmark table:
  ```
  ONNX Latency Benchmark (CPU)
  ┌────────────┬────────────┬────────────┬────────────┐
  │ Batch Size │ Mean (ms)  │ P95 (ms)   │ P99 (ms)   │
  ├────────────┼────────────┼────────────┼────────────┤
  │ 1          │ X.XX       │ X.XX       │ X.XX       │
  │ 8          │ X.XX       │ X.XX       │ X.XX       │
  │ 32         │ X.XX       │ X.XX       │ X.XX       │
  └────────────┴────────────┴────────────┴────────────┘
  ```

- [ ] Add latency numbers to `results/metrics.json`

**Acceptance Criterion:**  
`vehicleanomalynet.onnx` exists. PyTorch vs. ONNX output diff < 1e-4. Latency benchmark table printed. Mean batch=1 latency saved to metrics.

**Update PROGRESS.md when done.**

---

### MILESTONE 8 — Streamlit Dashboard
**Goal:** Interactive diagnostic UI that a non-technical NVH team member could use.

**Tasks:**
- [ ] Implement `dashboard.py` with these panels:

**Panel 1 — File Upload & Inference**
- Upload `.wav` file or select from test set samples
- Run inference on upload
- Display: anomaly score gauge (0–1, red if >threshold), predicted fault type + confidence, processing time

**Panel 2 — Spectrogram Viewer**
- Log-Mel spectrogram of uploaded audio (matplotlib in Streamlit)
- Raw waveform plot
- Color: viridis colormap, axes labeled (time in seconds, frequency in Hz)

**Panel 3 — Model Performance**
- Load `results/metrics.json` and display as formatted table
- Show ROC curve image from `results/roc_curve.png`
- Show confusion matrix from `results/confusion_matrix.png`

**Panel 4 — Experiment History (MLflow)**
- Pull last 5 MLflow runs
- Display as table: run_name, val_auc, val_f1, epochs, timestamp

- [ ] Dashboard must run with: `streamlit run dashboard.py`
- [ ] No hardcoded paths — all paths from `config.yaml`

**Acceptance Criterion:**  
`streamlit run dashboard.py` opens without error. Upload a test WAV → anomaly score appears. All 4 panels render. No hardcoded paths.

**Update PROGRESS.md when done.**

---

### MILESTONE 9 — README & Cleanup
**Goal:** GitHub-ready repo a Tesla recruiter can run in 10 minutes.

**Tasks:**
- [ ] Write `README.md` with these sections:
  - **Overview**: 2-sentence description + diagram (ASCII is fine)
  - **Motivation**: Why MIMII → Tesla NVH connection (1 paragraph)
  - **Architecture**: Model diagram + description
  - **Results table**: AUC-ROC, F1, latency — fill from `results/metrics.json`
  - **Quick Start**: Copy-pasteable commands to reproduce from scratch
  - **Project Structure**: File tree with one-line description per file
  - **Dataset**: How to download MIMII, what it contains

- [ ] Final cleanup:
  - [ ] `requirements.txt` has every dependency with pinned versions
  - [ ] `config.yaml` has no magic numbers — every hyperparameter is there
  - [ ] Remove all debug print statements
  - [ ] All functions have docstrings
  - [ ] `.gitignore` excludes `data/`, `mlruns/`, `__pycache__/`, `*.onnx`
  - [ ] `PROGRESS.md` has all milestones marked `[x]`

**Acceptance Criterion:**  
Fresh clone + `pip install -r requirements.txt` + follow README reproduces the project. All PROGRESS.md items `[x]`. README has real numbers in the results table.

**Update PROGRESS.md when done.**

---

## ⚙️ CONFIG TEMPLATE

Start with this `config.yaml`. Agent must fill in values based on architecture decisions.

```yaml
# VehicleAnomalyNet Configuration
# All hyperparameters live here — no magic numbers in code

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  metadata_path: "data/processed/metadata.csv"
  machine_types: ["fan", "slider"]
  snr_levels: [-6, 0, 6]
  augment_factor: 3
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  segment_length_s: 4.0
  sample_rate: 16000

features:
  n_mels: 128
  n_fft: 1024
  hop_length: 512
  f_min: 20.0
  f_max: 8000.0
  n_mfcc: 40
  target_frames: 128    # fixed time dimension
  feature_type: "log_mel"  # or "mfcc" or "combined"

model:
  cnn_channels: [32, 64, 128]
  gru_hidden: 256
  gru_layers: 2
  gru_bidirectional: true
  dropout: 0.3
  n_fault_classes: 7   # normal + 6 fault types (fill after EDA)

training:
  batch_size: 64
  epochs: 50
  lr: 0.001
  weight_decay: 0.0001
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"
  early_stopping_patience: 10
  loss_weights:
    anomaly: 1.0
    fault: 0.5
  device: "auto"   # auto-selects cuda > mps > cpu
  checkpoint_dir: "checkpoints/"
  best_model_path: "checkpoints/best_model.pt"

evaluation:
  anomaly_threshold: 0.5
  results_dir: "results/"

export:
  onnx_path: "vehicleanomalynet.onnx"
  benchmark_runs: 500
  warmup_runs: 200
  batch_sizes: [1, 8, 32]

dashboard:
  port: 8501
  mlflow_tracking_uri: "mlruns/"
  n_recent_runs: 5
```

---

## 📐 TECHNICAL CONSTRAINTS

The agent must follow these non-negotiable constraints:

1. **No magic numbers in code** — every hyperparameter comes from `config.yaml` via a config loader
2. **Type hints on all functions** — `def foo(x: torch.Tensor, sr: int) -> torch.Tensor:`
3. **No global state** — everything passed explicitly, no module-level mutable globals
4. **Reproducibility** — set random seeds for `torch`, `numpy`, `random` at training start
5. **Device agnostic** — `config.device = "auto"` detects CUDA → MPS → CPU
6. **No data leakage** — val/test sets never touched until their respective phases
7. **SNR mixing must be RMS-based** — not amplitude-based (common mistake)
8. **Fault head loss must be masked** — normal samples have `fault_label = -1`, masked in CrossEntropy with `ignore_index=-1`

---

## 🏁 DEFINITION OF DONE

The project is complete when ALL of the following are true:

- [ ] `PROGRESS.md` has every item marked `[x]`
- [ ] `python scripts/run_pipeline.py` generates dataset from scratch
- [ ] `python scripts/run_training.py --config config.yaml` trains a model
- [ ] `python scripts/export_onnx.py` exports and benchmarks the model
- [ ] `streamlit run dashboard.py` launches the UI
- [ ] `results/metrics.json` contains AUC-ROC, F1, and latency numbers
- [ ] `README.md` has real numbers in the results table
- [ ] All PROGRESS.md milestones complete

---

## 🧾 RESUME BULLET TEMPLATE

Fill this in after Milestone 6 and 7 are complete:

```
Built VehicleAnomalyNet, a dual-task CNN+BiGRU pipeline for mechanical fault 
detection and sound event classification on the MIMII industrial machine benchmark, 
achieving AUC-ROC of [M6_AUC] and [M6_F1] macro-F1; engineered an automated 
ground truth data generation pipeline producing [M2_N_SAMPLES] SNR-mixed augmented 
samples via torchaudio and librosa, with ONNX export achieving [M7_LATENCY]ms 
average inference latency.
```

---

*This PRD was designed to mirror Tesla NVH's production workflow:*
*audio diagnostics → automated data pipelines → scalable deployment.*