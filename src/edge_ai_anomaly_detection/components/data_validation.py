import numpy as np
import yaml
from pathlib import Path
from src.edge_ai_anomaly_detection.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate(self):
        
        schema = yaml.safe_load(Path(self.config.schema_path).read_text())
        splits = schema["splits"]
        results = []
        all_ok = True

        
        for split, req in splits.items():
            fpath = self.config.processed_dir / f"{split}.npy"
            if not fpath.exists():
                results.append(f"{split}: ❌ فایل یافت نشد")
                all_ok = False
                continue

            data = np.load(fpath)
            n = data.shape[0]
            min_files = req.get("min_files", 0)

            
            if n < min_files:
                results.append(f"{split}: ⚠️ تعداد {n} < حداقل {min_files}")
                all_ok = False
            else:
                results.append(f"{split}: ✅ {n} فایل (shape {data.shape})")

        
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config.status_file, "w") as f:
            f.write(f"Validation: {'VALID ✅' if all_ok else 'INVALID ❌'}\n")
            f.write("\n".join(results))

        print(f"✅ DataValidation: \"{('VALID' if all_ok else 'INVALID')}\"")
        print("\n".join(results))
        return all_ok
