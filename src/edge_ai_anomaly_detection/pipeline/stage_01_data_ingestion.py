from pathlib import Path
from src.edge_ai_anomaly_detection.components.data_ingestion import DataIngestion
from src.edge_ai_anomaly_detection.config.configuration import ConfigurationManager


class DataIngestionTrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def main(self):
        config = self.config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=config)
        
        data_ingestion.verify_structure()
