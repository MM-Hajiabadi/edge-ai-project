from src.edge_ai_anomaly_detection.components.model_trainer import ModelTrainer
from src.edge_ai_anomaly_detection.config.configuration import ConfigurationManager


class ModelTrainerTrainingPipeline:
    def main(self):
        cm = ConfigurationManager()
        config = cm.get_model_trainer_config()
        trainer = ModelTrainer(config, config.processed_dir)
        trainer.train()
