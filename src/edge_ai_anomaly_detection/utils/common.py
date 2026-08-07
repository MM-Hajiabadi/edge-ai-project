import json
import time
from pathlib import Path


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def log_experiment(experiments_path: Path, name: str, params: dict, metrics: dict):
    
    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "name": name,
        "params": params,
        "metrics": metrics
    }
    if experiments_path.exists():
        data = load_json(experiments_path)
    else:
        data = []
    data.append(record)
    save_json(data, experiments_path)


def ensure_dirs(paths: list[Path]):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
