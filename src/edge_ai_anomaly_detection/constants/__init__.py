from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
SCHEMA_PATH = PROJECT_ROOT / "schema.yaml"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"