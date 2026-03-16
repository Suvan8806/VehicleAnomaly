"""Training loop and Trainer class for VehicleAnomalyNet."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from vehicleanomalynet.losses import DualTaskLoss


def _get_device(device_cfg: str) -> torch.device:
    """Resolve device: 'auto' -> cuda > mps > cpu."""
    if device_cfg != "auto":
        return torch.device(device_cfg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dataloaders: Dict[str, DataLoader],
        config: dict,
        run_name: str = "vehicleanomalynet",
    ) -> None:
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self._run_name = run_name
        train_cfg = config["training"]
        self.device = _get_device(str(train_cfg.get("device", "auto")))
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        self.epochs = int(train_cfg["epochs"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs
        )
        anomaly_pos_weight = train_cfg.get("anomaly_pos_weight")
        if anomaly_pos_weight is not None:
            anomaly_pos_weight = float(anomaly_pos_weight)
        self.criterion = DualTaskLoss(
            anomaly_weight=float(train_cfg["loss_weights"]["anomaly"]),
            fault_weight=float(train_cfg["loss_weights"]["fault"]),
            anomaly_pos_weight=anomaly_pos_weight,
        )
        self.patience = int(train_cfg.get("early_stopping_patience", 10))
        self.best_model_path = Path(train_cfg.get("best_model_path", "checkpoints/best_model.pt"))
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Optional: stop when val metric reaches target (so fan/slider can run different epoch counts)
        _auc = train_cfg.get("target_val_auc_roc")
        self.target_val_auc_roc = float(_auc) if _auc is not None else None
        _loss = train_cfg.get("target_val_loss")
        self.target_val_loss = float(_loss) if _loss is not None else None

        self.best_val_auc = -1.0
        self.epochs_without_improvement = 0

    def train_epoch(self) -> Dict[str, float]:
        """Returns {"loss": float, "anomaly_loss": float, "fault_loss": float}."""
        self.model.train()
        loader = self.dataloaders["train"]
        total_loss = 0.0
        total_anom = 0.0
        total_fault = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc="Train", leave=False)
        for batch in pbar:
            x = batch["features"].to(self.device)
            anomaly_label = batch["anomaly_label"].to(self.device)
            fault_label = batch["fault_label"].to(self.device)

            self.optimizer.zero_grad()
            anomaly_logit, fault_logit = self.model(x)
            loss_dict = self.criterion(
                anomaly_logit, fault_logit, anomaly_label, fault_label
            )
            loss_dict["total"].backward()
            self.optimizer.step()

            total_loss += loss_dict["total"].item()
            total_anom += loss_dict["anomaly"].item()
            total_fault += loss_dict["fault"].item()
            n_batches += 1
            pbar.set_postfix(loss=loss_dict["total"].item())

        n_batches = max(n_batches, 1)
        return {
            "loss": total_loss / n_batches,
            "anomaly_loss": total_anom / n_batches,
            "fault_loss": total_fault / n_batches,
        }

    def val_epoch(self) -> Dict[str, float]:
        """Returns {"loss": float, "auc_roc": float, "f1": float}."""
        self.model.eval()
        loader = self.dataloaders["val"]
        if len(loader) == 0:
            raise RuntimeError(
                "Validation DataLoader is empty. "
                "Ensure that GroundTruthPipeline generated a non-empty 'val' split."
            )
        total_loss = 0.0
        n_batches = 0
        all_anomaly_scores: list = []
        all_anomaly_labels: list = []
        all_fault_preds: list = []
        all_fault_labels: list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Val", leave=False):
                x = batch["features"].to(self.device)
                anomaly_label = batch["anomaly_label"].to(self.device)
                fault_label = batch["fault_label"].to(self.device)

                anomaly_logit, fault_logit = self.model(x)
                loss_dict = self.criterion(
                    anomaly_logit, fault_logit, anomaly_label, fault_label
                )
                total_loss += loss_dict["total"].item()
                n_batches += 1

                scores = torch.sigmoid(anomaly_logit).cpu().numpy().ravel()
                all_anomaly_scores.extend(scores.tolist())
                all_anomaly_labels.extend(anomaly_label.cpu().numpy().tolist())

                preds = fault_logit.argmax(dim=1).cpu().numpy().ravel()
                all_fault_preds.extend(preds.tolist())
                all_fault_labels.extend(fault_label.cpu().numpy().tolist())

        n_batches = max(n_batches, 1)
        val_loss = total_loss / n_batches

        y_true = np.array(all_anomaly_labels)
        y_score = np.array(all_anomaly_scores)
        try:
            auc_roc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auc_roc = 0.0
        if np.isnan(auc_roc):
            auc_roc = 0.0

        fp = np.array(all_fault_preds)
        fl = np.array(all_fault_labels)
        mask = fl >= 0
        if mask.sum() > 0:
            f1 = float(f1_score(fl[mask], fp[mask], average="macro", zero_division=0))
        else:
            f1 = 0.0

        return {"loss": val_loss, "auc_roc": auc_roc, "f1": f1}

    def fit(self) -> Dict[str, Any]:
        """Full training loop. Returns best val metrics."""
        import mlflow

        mlflow.set_tracking_uri(
            self.config.get("dashboard", {}).get("mlflow_tracking_uri", "mlruns/")
        )
        with mlflow.start_run(run_name=self._run_name):
            mlflow.log_params(_flatten_config(self.config))

            for epoch in range(1, self.epochs + 1):
                train_metrics = self.train_epoch()
                val_metrics = self.val_epoch()
                self.scheduler.step()
                lr = self.optimizer.param_groups[0]["lr"]

                mlflow.log_metrics(
                    {
                        "train_loss": train_metrics["loss"],
                        "train_anomaly_loss": train_metrics["anomaly_loss"],
                        "train_fault_loss": train_metrics["fault_loss"],
                        "val_loss": val_metrics["loss"],
                        "val_auc_roc": val_metrics["auc_roc"],
                        "val_f1": val_metrics["f1"],
                        "lr": lr,
                    },
                    step=epoch,
                )

                print(
                    f"Epoch {epoch} | Train Loss: {train_metrics['loss']:.3f} | "
                    f"Val AUC: {val_metrics['auc_roc']:.3f} | Val F1: {val_metrics['f1']:.3f} | LR: {lr:.5f}"
                )

                auc = val_metrics["auc_roc"]
                if not (np.isfinite(auc)):
                    auc = 0.0
                if auc > self.best_val_auc:
                    self.best_val_auc = auc
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(str(self.best_model_path))
                else:
                    self.epochs_without_improvement += 1
                    if self.epochs_without_improvement >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

                # Stop when target val AUC or val loss is reached (epochs can differ per type)
                if self.target_val_auc_roc is not None and auc >= self.target_val_auc_roc:
                    print(f"Target val AUC-ROC {self.target_val_auc_roc} reached at epoch {epoch}")
                    break
                if self.target_val_loss is not None and val_metrics["loss"] <= self.target_val_loss:
                    print(f"Target val loss {self.target_val_loss} reached at epoch {epoch}")
                    break

            best = {
                "best_val_auc_roc": self.best_val_auc,
                "epochs_run": epoch,
            }
            mlflow.log_metrics(best)
        return best

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model state (and optimizer if present)."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])


def _flatten_config(config: dict, prefix: str = "") -> dict:
    """Flatten nested config for MLflow log_params."""
    out = {}
    for k, v in config.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_config(v, prefix=f"{key}."))
        else:
            out[key] = str(v)
    return out
