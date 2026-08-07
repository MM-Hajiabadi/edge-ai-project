import os, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from src.edge_ai_anomaly_detection.components.model_trainer import Autoencoder
from src.edge_ai_anomaly_detection.utils.common import log_experiment, save_json


class ModelEvaluation:
    def __init__(self, config, model_path: Path, processed_dir: Path, experiments_path: Path):
        self.config = config
        self.model_path = model_path
        self.processed_dir = processed_dir
        self.experiments_path = experiments_path

    def _norm_split(self, name):
        data = np.load(self.processed_dir / f"{name}.npy")
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        return torch.from_numpy(data.astype(np.float32)).unsqueeze(1)

    def _errors(self, model, tens):
        model.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(tens), 32):
                b = tens[i:i+32]
                recon = model(b)
                errs.extend(((recon - b) ** 2).mean(dim=[1, 2, 3]).cpu().numpy())
        return np.array(errs)

    def evaluate(self):
        model = Autoencoder()
        model.load_state_dict(torch.load(self.model_path, map_location='cpu'))

        src_n = self._norm_split("src_test_normal")
        src_a = self._norm_split("src_test_anomaly")
        err_n = self._errors(model, src_n)
        err_a = self._errors(model, src_a)

        y_true = np.concatenate([np.zeros(len(err_n)), np.ones(len(err_a))])
        y_score = np.concatenate([err_n, err_a])
        auc = roc_auc_score(y_true, y_score)

        metrics = {"auc_source": float(auc)}
        save_json(metrics, self.config.metrics_path)

        log_experiment(self.experiments_path, "autoencoder",
                       params={"model": "autoencoder", "auc_threshold": self.config.auc_threshold},
                       metrics=metrics)

        print(f"  AUC = {auc:.4f}")
        if auc < self.config.auc_threshold:
            print(f"  ⚠️ AUC کمتر از آستانه {self.config.auc_threshold} است!")
        else:
            print(f"  ✅ AUC بالاتر از آستانه {self.config.auc_threshold}")
        return auc
