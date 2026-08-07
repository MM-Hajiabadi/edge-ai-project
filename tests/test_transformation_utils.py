import numpy as np
from src.edge_ai_anomaly_detection.utils.common import save_json, load_json
from pathlib import Path

def test_save_load_json(tmp_path):
    data = {"auc": 0.9, "name": "test"}
    p = tmp_path / "metrics.json"
    save_json(data, p)
    loaded = load_json(p)
    assert loaded == data


def test_log_experiment(tmp_path):
    from src.edge_ai_anomaly_detection.utils.common import log_experiment
    f = tmp_path / "exp.json"
    log_experiment(f, "exp1", {"lr": 0.001}, {"auc": 0.9})
    log_experiment(f, "exp2", {"lr": 0.01}, {"auc": 0.85})
    data = load_json(f)
    assert len(data) == 2
    assert data[0]["name"] == "exp1"
