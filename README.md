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

## Feature/Dataset smoke test (Milestone 3)

```powershell
.\.venv\Scripts\python.exe -m vehicleanomalynet.dataset data\processed\metadata.csv --config config.yaml
```

## EDA notebook

Open and run `notebooks/01_eda.ipynb` (figures are saved to `notebooks/figures/`).

