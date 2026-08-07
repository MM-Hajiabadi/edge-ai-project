from pathlib import Path
from src.edge_ai_anomaly_detection.components.quantization import Quantization
from src.edge_ai_anomaly_detection.config.configuration import ConfigurationManager


class QuantizationPipeline:

    def main(self):
        
        cm = ConfigurationManager()
        config = cm.get_quantization_config()
        q = Quantization(config, config.model_path, config.processed_dir)
        q.run()
