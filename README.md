# VehicleAnomalyNet

Dual-task audio ML for **mechanical anomaly detection** + **fault-type classification** on **MIMII** (`fan`, `slider`).

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Data (MIMII)

Place these in `data/raw/`:
- `0_dB_fan.zip` (md5 `6354d1cc2165c52168f9ef1bcd9c7c52`)
- `0_dB_slider.zip` (md5 `4d674c21474f0646ecd75546db6c0c4e`)

Then run:

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py --config config.yaml
```

## Generate processed dataset + metadata

```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py --config config.yaml --seed 42
```

Outputs:
- `data/processed/metadata.csv`
- processed float32 WAV clips under `data/processed/{train,val,test}/`

**Class balance:** In `config.yaml`, `data.max_label_ratio` (default `0.60`) caps the fraction of each split from one label so both normal and anomaly appear in train/val/test. Use `0.5` for 50/50 balance.

**How to run (clean slate, pipeline, train, evaluate):**

- Clean and run pipeline in one go:
  ```powershell
  .venv\Scripts\python.exe scripts\run_pipeline.py --clean --config config.yaml
  ```
- See what would be removed without deleting:
  ```powershell
  .venv\Scripts\python.exe scripts\clean_processed_data.py --dry-run
  ```
- Re-train (new data, so new checkpoint):
  ```powershell
  .venv\Scripts\python.exe scripts\run_training.py --config config.yaml --run-name baseline_v2
  ```
- Re-run evaluation:
  ```powershell
  .venv\Scripts\python.exe scripts\run_evaluation.py
  ```

## K-fold cross-validation

For a more robust AUC estimate (mean ± std over folds):

```powershell
.\.venv\Scripts\python.exe scripts\run_cv.py --config config.yaml --folds 5
```

This trains and evaluates on 5 folds (stratified by machine_type) and prints mean test AUC-ROC ± std.

## Feature/Dataset smoke test (Milestone 3)

```powershell
.\.venv\Scripts\python.exe -m vehicleanomalynet.dataset data\processed\metadata.csv --config config.yaml
```

## Train

```powershell
.\.venv\Scripts\python.exe scripts\run_training.py --config config.yaml --run-name baseline_v2
```

## MLflow UI

From project root (so `mlruns/` is found):

```powershell
.\.venv\Scripts\python.exe scripts\run_mlflow_ui.py
```

Then open **http://127.0.0.1:5000** in your browser. To use another port:

```powershell
.\.venv\Scripts\python.exe scripts\run_mlflow_ui.py --port 5001
```

**Seeing your training runs:** In MLflow 3.x the default view is GenAI (Traces). To see the **Runs** table (params, metrics), use the **workflow selector in the top navigation** and switch to **"Classical ML"** (or **"ML"**), then open **Experiments → Default**. You can also list runs from the terminal: `python scripts/list_mlflow_runs.py`.

**Runs/Models empty?** The UI reads from `mlflow.db` (SQLite). If you trained before that was set up, your runs are in `mlruns/`. Stop the MLflow UI (Ctrl+C), then run once: `python scripts/migrate_mlruns_to_sqlite.py`, then start the UI again. New training runs are already logged to `mlflow.db`.

## EDA notebook

Open and run `notebooks/01_eda.ipynb` (figures are saved to `notebooks/figures/`).

