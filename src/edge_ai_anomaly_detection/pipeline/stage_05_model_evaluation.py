from pathlib import Path
from src.edge_ai_anomaly_detection.components.model_evaluation import ModelEvaluation
from src.edge_ai_anomaly_detection.config.configuration import ConfigurationManager


class ModelEvaluationPipeline:
    def main(self):
        cm = ConfigurationManager()
        config = cm.get_model_evaluation_config()

        model_path = Path(cm.config["model_trainer"]["model_path"])
        
        experiments_path = Path(cm.config["experiments_log"])

        evaluator = ModelEvaluation(
            config=config,
            model_path=model_path,
            processed_dir=config.processed_dir,
            experiments_path=experiments_path
        )
        auc = evaluator.evaluate()
        print(f"✅ Model Evaluation: AUC = {auc:.4f}")
        return auc
