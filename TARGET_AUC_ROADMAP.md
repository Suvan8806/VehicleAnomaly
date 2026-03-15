# Target AUC-ROC Roadmap (≥ 0.75 per type)

This document describes the strategy and steps to reach **test AUC-ROC ≥ 0.75** for each machine type (fan, slider) and where results are stored.

---

## Goal

Achieve **test AUC-ROC ≥ 0.75** for each machine type when trained and evaluated **per type** (same type in train/val/test). No cross-domain evaluation: fan model is evaluated on fan test data only; slider model on slider test data only.

---

## Strategy

- **Train one model per type.** Save checkpoints as `checkpoints/fan.pt` and `checkpoints/slider.pt`.
- **Evaluate each model** with its own metadata and checkpoint so test set matches the training domain.
- **Combine later** via one app or API that routes by machine type (user choice or parameter): load the corresponding checkpoint and run inference. No merging of networks.

---

## Steps (per type)

For **fan**:

1. **Pipeline (fan-only)**  
   `python scripts/run_pipeline.py --clean --config config_fan.yaml`  
   Writes to `data/processed_fan/` and `data/processed_fan/metadata.csv`.

2. **Train**  
   `python scripts/run_training.py --config config_fan.yaml --run-name fan_v1 --output-checkpoint checkpoints/fan.pt`  
   Best checkpoint saved to `checkpoints/fan.pt`.

3. **Evaluate**  
   `python scripts/run_evaluation.py --config config_fan.yaml --checkpoint checkpoints/fan.pt`  
   Uses `data/processed_fan/metadata.csv` for the test set. Results and plots go to `results_fan/` (see `config_fan.yaml`).

For **slider**:

1. **Pipeline (slider-only)**  
   `python scripts/run_pipeline.py --clean --config config_slider.yaml`  
   Writes to `data/processed_slider/` and `data/processed_slider/metadata.csv`.

2. **Train**  
   `python scripts/run_training.py --config config_slider.yaml --run-name slider_v1 --output-checkpoint checkpoints/slider.pt`  
   Best checkpoint saved to `checkpoints/slider.pt`.

3. **Evaluate**  
   `python scripts/run_evaluation.py --config config_slider.yaml --checkpoint checkpoints/slider.pt`  
   Results and plots go to `results_slider/`.

---

## Success criteria

- **Per type:** Test AUC-ROC ≥ 0.75 for that type (reported in `results_fan/metrics.json` or `results_slider/metrics.json` for that run).
- **Milestone 6:** Consider the acceptance criterion (“Test AUC-ROC > 0.75”) met when **at least one** type reaches ≥ 0.75 under this per-type setup; ideally both fan and slider.

---

## If below 0.75

- Tune hyperparameters (e.g. `anomaly_pos_weight`, `max_label_ratio`, `dropout`, `epochs`) in the type-specific config and re-run **training and evaluation** only.
- Re-run the **pipeline** only if you change data or split settings (e.g. `max_generated_samples`, `max_label_ratio`).

---

## Where results are stored

| Type   | Processed data      | Metadata                  | Checkpoint          | Evaluation results   |
|--------|---------------------|---------------------------|---------------------|-----------------------|
| Fan    | `data/processed_fan/` | `data/processed_fan/metadata.csv` | `checkpoints/fan.pt`   | `results_fan/`        |
| Slider | `data/processed_slider/` | `data/processed_slider/metadata.csv` | `checkpoints/slider.pt` | `results_slider/`     |

---

## Later

- **Add pump/valve:** Use the same per-type workflow: add `config_pump.yaml` and `config_valve.yaml` (and fault-type mappings in the pipeline/dataset), then pipeline → train → eval per type; save `checkpoints/pump.pt`, `checkpoints/valve.pt`.
- **Optional CLI overrides:** Add `--machine-types` and `--output-suffix` to `run_pipeline.py` so you can run per-type pipeline without maintaining multiple config files.
