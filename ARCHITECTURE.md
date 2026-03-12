# VehicleAnomalyNet — Architecture Decisions (Contract)

## Project summary

VehicleAnomalyNet is an end-to-end audio ML system that (1) **detects anomalous mechanical sounds** and (2) **classifies the likely fault type** from short WAV clips. The deliverable includes a data download + EDA workflow, a ground-truth generation pipeline with RMS-correct SNR mixing and augmentation, a dual-head neural model, evaluation, ONNX export with CPU latency benchmarking, and a Streamlit dashboard for diagnostics.

## Dataset

- **Dataset**: **MIMII (Malfunctioning Industrial Machine Investigation & Inspection)** from Zenodo (`https://zenodo.org/record/3384388`).
- **Machine types used**: `fan` and `slider` only (explicit PRD requirement; keeps scope manageable).
- **Audio format assumptions**: WAV, 16 kHz, mono (per PRD).
- **Fault types (classification target)**:
  - Fan: `rotating_imbalance`, `contamination`, `voltage_change`
  - Slider: `rail_damage`, `no_grease`, `loose_screws`
  - Plus `normal` (handled specially; see “dual-task objective”).

---

## FRAMEWORK DECISIONS

### Which audio feature library will you use: torchaudio, librosa, or both? Why?

**Both (`torchaudio` + `librosa`)**.

- **torchaudio**: integrates cleanly with PyTorch tensors, is performant, and is the most straightforward path to keeping feature extraction deterministic and GPU-friendly during training.
- **librosa**: provides reliable, widely-used utilities for certain audio operations and visualization/EDA workflows; it can also serve as a fallback for transforms not as convenient in torchaudio.

Primary pipeline intent: **training-time features via torchaudio**; EDA/plotting and any non-critical utilities may use librosa.

### What input representation will you use: log-Mel spectrogram, MFCC, raw waveform, or multi-feature? Justify with tradeoffs.

**Primary representation: log-Mel spectrogram.**

- **Pros**: strong baseline for machine acoustics, robust to small time shifts, compact, and well-matched to CNN feature extractors.
- **Cons**: discards some phase/fine temporal detail vs raw waveform.

MFCC(+deltas) will be supported as an alternative feature mode if needed, but baseline training will start with **log-Mel** to minimize complexity and maximize signal-to-noise.

### What is the input tensor shape to the model? (batch, channels, height, width)?

**\((B, 1, n\_mels, T)\)** where:

- \(B\) = batch size
- channels = 1 (single “image channel”)
- height = `n_mels` (e.g., 128)
- width = \(T\) = `target_frames` (fixed via pad/truncate)

### Will you use a pretrained backbone (e.g., CNN14 from PANNs, AST) or train from scratch? Why?

**Train from scratch (CNN + temporal module).**

- Keeps the project lightweight and reproducible on typical CPUs/GPUs.
- Aligns with the PRD’s milestone-driven implementation (custom `model.py` with ≥3 conv blocks + BiGRU).
- Avoids adding complexity around adapting large pretrained models, their preprocessing constraints, and potentially heavier dependencies.

### What framework for experiment tracking: MLflow locally, or Weights & Biases?

**MLflow locally.** This is explicitly required by Milestone 5 (populate `mlruns/`, viewable via `mlflow ui`).

### What is the ONNX export strategy for edge inference?

**Export the trained PyTorch model to ONNX** (feature extraction remains in Python for this project’s scope).

- Export `VehicleAnomalyNet` with a fixed input shape \((B, 1, n\_mels, T)\) for benchmarked batch sizes \([1, 8, 32]\).
- Validate parity: run a small batch through PyTorch and ONNXRuntime and assert output differences are < \(1e^{-4}\).
- Benchmark CPU latency with warmup + timed runs and record mean/std/p95/p99 in `results/metrics.json`.

---

## ARCHITECTURE DECISIONS

### Describe the full model architecture in prose: input → backbone → temporal module → output heads

**Input**: log-Mel features \((B, 1, n\_mels, T)\)

1. **CNN backbone**: 3+ convolutional blocks (Conv2d → BatchNorm → nonlinearity → MaxPool), progressively increasing channels. Pooling reduces time/frequency resolution and yields a compact feature map.
2. **Temporal module**: reshape/aggregate CNN features across time to a sequence and pass through a **BiGRU** (2 layers, hidden size from config).
3. **Pooling**: temporal pooling over GRU outputs (e.g., mean or attention-less pooling) to create a fixed-dimensional embedding.
4. **Dual heads**:
   - **Anomaly head**: linear layer → **single logit** (binary anomaly).
   - **Fault head**: linear layer → **logits over fault classes** (normal + 6 faults, per config).

Also provide `get_embedding(x)` that returns the pooled embedding before the heads.

### Why CNN + recurrent vs. pure CNN vs. pure Transformer for this task?

**CNN + BiGRU** is a strong “signal-processing + sequence” baseline:

- **Pure CNN**: can work, but often benefits from explicitly modeling temporal dependencies; pooling can blur sequence-level patterns.
- **Transformer**: powerful but typically heavier, more sensitive to data scale/hyperparameters, and may increase project complexity (tokenization, positional encoding choices, compute).
- **CNN + BiGRU**: CNN learns local time-frequency patterns; GRU captures longer temporal evolution with modest compute and stable training.

### How will you handle the dual-task objective (anomaly detection + fault classification)?

- **Two supervised outputs** sharing a common embedding.
- **Anomaly detection** is defined for all samples (normal vs abnormal).
- **Fault classification** is defined only for **abnormal** samples.

Implementation contract:

- Encode **normal samples** with `fault_label = -1`.
- Compute fault loss with `ignore_index=-1` so normal samples do not contribute to fault classification loss.

### How will you handle class imbalance (anomalies are rare)?

Use a layered approach (starting simple, expanding only if needed):

1. **Loss weighting**: separate weights for anomaly vs fault loss (`training.loss_weights`).
2. **Anomaly positive weighting**: optionally use `pos_weight` in `BCEWithLogitsLoss` based on class ratio.
3. **Sampling strategy (optional)**: weighted random sampler at the dataset level if imbalance strongly hurts learning stability.

### What loss function(s) and why?

- **Anomaly head**: `BCEWithLogitsLoss`
  - numerically stable for logits and matches binary detection objective.
- **Fault head**: `CrossEntropyLoss(ignore_index=-1)`
  - handles multi-class faults; ignores normals as required by PRD constraints.
- **Total**: weighted sum

\[
\mathcal{L} = \lambda_{anom}\,\mathcal{L}_{BCE} + \lambda_{fault}\,\mathcal{L}_{CE(masked)}
\]

---

## DATA DECISIONS

### Which MIMII machine types will you use and why?

**`fan` and `slider` only.**

- Explicitly required by the PRD to keep scope manageable.
- Provides multiple fault types across two mechanical systems with distinct acoustic signatures.

### How will you structure the train/val/test splits?

**70/15/15 split, stratified and grouped by machine identity.**

Constraints:

- **No data leakage**: split by `machine_id` so audio from the same machine does not appear across splits.
- **Stratify** within the constraints of machine IDs by `fault_type` and `machine_type` as much as possible.

This evaluates generalization to unseen machines, which is closer to real diagnostic deployment.

### What augmentation strategy will you use and why is it valid for audio anomaly detection?

Augmentations (applied primarily to training set):

- **Time-stretch**: ±5% (small speed variations simulate operational variation without changing the fault identity).
- **Gain jitter**: ±3 dB (simulates microphone distance / recording gain changes).
- **Mild Gaussian noise**: low amplitude (simulates sensor noise).

These are valid because they preserve the underlying mechanical signatures while increasing robustness to nuisance variability.

### How will you generate the synthetic ground truth data (SNR mixing)?

Use **RMS-based mixing** to hit exact target SNR (PRD constraint: not amplitude-based).

Let:

- \(s\) = clean signal waveform
- \(n\) = noise waveform (cropped/looped to match length of \(s\))
- \(RMS(x) = \sqrt{\frac{1}{N}\sum x_i^2}\)

To mix at target SNR in dB:

1. Compute \(r_s = RMS(s)\), \(r_n = RMS(n)\)
2. Desired noise RMS: \(r_{n}' = r_s / 10^{SNR_{dB}/20}\)
3. Scale noise: \(n' = n \cdot (r_{n}'/(r_n+\epsilon))\)
4. Mix: \(x = s + n'\)

Optionally normalize/clamp to avoid clipping and save as 16 kHz mono float32 for processed output.

---

## EVALUATION DECISIONS

### What is your primary metric and why (AUC-ROC, F1, accuracy)?

**Primary metric: AUC-ROC for anomaly detection.**

- Threshold-free, robust under class imbalance, and explicitly called out as primary in the PRD milestones.

Secondary metrics:

- **Macro-F1** for fault classification (balanced across fault classes).
- Precision/recall for anomaly decision at a chosen threshold (configurable).

### What does "done" look like numerically for this project?

Minimum “done” criteria (aligned to PRD acceptance gates):

- **Test AUC-ROC > 0.75** (Milestone 6 acceptance criterion).
- Evaluation artifacts produced (`results/metrics.json` + plots) with per-machine AUC populated.

Stretch targets (if achievable without scope creep):

- Macro-F1 for fault classification that is materially above chance (document final number in `results/metrics.json` and `README.md`).

---

## DEPLOYMENT DECISIONS

### What is the latency target for ONNX inference?

Target: **CPU batch=1 mean latency < 20 ms** for a single forward pass on a typical developer laptop CPU.

Notes:

- This is a **goal**, not a guarantee; the project will **measure** mean/std/p95/p99 as required (Milestone 7) and record results in `results/metrics.json`.
- Batch sizes 1/8/32 will be benchmarked to characterize throughput/latency trade-offs.

### What does the Streamlit dashboard need to show to be useful?

It must implement the PRD’s 4 panels:

1. **File Upload & Inference**
   - Upload `.wav` or select a test sample
   - Show anomaly score (0–1), anomaly threshold indicator, predicted fault + confidence, processing time
2. **Spectrogram Viewer**
   - Log-Mel spectrogram plot (viridis, labeled axes)
   - Raw waveform plot
3. **Model Performance**
   - Load and display `results/metrics.json` in a readable table
   - Show ROC curve and confusion matrix images
4. **Experiment History (MLflow)**
   - List last 5 MLflow runs with key metrics and metadata

