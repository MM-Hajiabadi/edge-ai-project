import zipfile
from pathlib import Path
from src.edge_ai_anomaly_detection.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def prepare_source(self, zip_path: Path):
        self.config.data_dir.mkdir(parents=True, exist_ok=True)
        
        if not (self.config.data_dir / "train").is_dir():
            print(f"  اکسترکت {zip_path.name} ...")
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(self.config.data_dir)
            print(f"✅ اکسترکت شد → {self.config.data_dir}")
        else:
            print(f"✅ داده از قبل اکسترکت شده — رد می‌شویم")

    def verify_structure(self):
        
        required = ["train", "source_test", "target_test"]
        missing = []
        for d in required:
            if not (self.config.data_dir / d).is_dir():
                missing.append(d)
        if missing:
            raise FileNotFoundError(f"پوشه‌های ناقص: {missing}")
        print(f"✅ ساختار داده معتبر است: {required}")
        return True
