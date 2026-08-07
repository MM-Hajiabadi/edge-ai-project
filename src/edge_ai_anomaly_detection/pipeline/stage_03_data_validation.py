from src.edge_ai_anomaly_detection.components.data_validation import DataValidation
from src.edge_ai_anomaly_detection.config.configuration import ConfigurationManager


class DataValidationTrainingPipeline:
    def main(self):
        cm = ConfigurationManager()
        config = cm.get_data_validation_config()
        validation = DataValidation(config)
        validation.validate()