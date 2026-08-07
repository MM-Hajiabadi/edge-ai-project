from src.edge_ai_anomaly_detection.components.data_transformation import DataTransformation
from src.edge_ai_anomaly_detection.config.configuration import ConfigurationManager


class DataTransformationTrainingPipeline:
    def main(self):
        cm = ConfigurationManager()
        config = cm.get_data_transformation_config()
        data_trans = DataTransformation(config)

        data_trans.process_split("train", "normal", "train_normal")
        data_trans.process_split("source_test", "normal", "src_test_normal")
        data_trans.process_split("source_test", "anomaly", "src_test_anomaly")
        data_trans.process_split("target_test", "normal", "tgt_test_normal")
        data_trans.process_split("target_test", "anomaly", "tgt_test_anomaly")