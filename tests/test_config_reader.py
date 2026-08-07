from pathlib import Path
from src.edge_ai_anomaly_detection.config.configuration import ConfigurationManager


def test_config_loaded():
    cm = ConfigurationManager()
    assert cm.config is not None
    assert "data_ingestion" in cm.config
    assert "model_trainer" in cm.config


def test_data_ingestion_entity():
    cm = ConfigurationManager()
    cfg = cm.get_data_ingestion_config()
    assert cfg.root_dir is not None
    assert cfg.source_zenodo_id != ""


def test_params_loaded():
    cm = ConfigurationManager()
    assert cm.params["feature_extraction"]["n_mels"] == 32
    assert cm.params["training"]["epochs"] > 0
