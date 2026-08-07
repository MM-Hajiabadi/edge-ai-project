import yaml
from pathlib import Path
from src.edge_ai_anomaly_detection.entity.config_entity import (
    DataIngestionConfig, DataTransformationConfig, DataValidationConfig,
    ModelTrainerConfig, ModelEvaluationConfig, QuantizationConfig
)

class ConfigurationManager:

    def __init__(self, config_path="config/config.yaml",
                 params_path="params.yaml", schema_path="schema.yaml"):
        
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.params = yaml.safe_load(Path(params_path).read_text())
        self.schema = yaml.safe_load(Path(schema_path).read_text())

        Path(self.config["artifacts_root"]).mkdir(parents=True, exist_ok=True)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        c = self.config["data_ingestion"]
        return DataIngestionConfig(
            root_dir=Path(c["root_dir"]),
            source_zenodo_id=c["source_zenodo_id"],
            data_dir=Path(c["data_dir"])
        )
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        c = self.config["data_transformation"]
        ing = self.config["data_ingestion"]
        p = self.params["feature_extraction"]
        return DataTransformationConfig(
            root_dir=Path(c["root_dir"]),
            data_dir_root=Path(ing["data_dir"]),
            processed_dir=Path(c["processed_dir"]),
            sample_rate=int(p["sample_rate"]),   
            n_fft=int(p["n_fft"]),               
            hop_length=int(p["hop_length"]),    
            n_mels=int(p["n_mels"]),             
            max_frames=int(p["max_frames"])      
        )


    def get_data_validation_config(self) -> DataValidationConfig:

        c = self.config["data_validation"]
        processed_dir = self.config["data_transformation"]["processed_dir"]
        return DataValidationConfig(
            root_dir=Path(c["root_dir"]),
            status_file=Path(c["status_file"]),
            schema_path=Path(c["schema_path"]),
            processed_dir=Path(processed_dir)
        )
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:

        c = self.config["model_trainer"]
        p = self.params["training"]
        m = self.params["model"]
        return ModelTrainerConfig(
            root_dir=Path(c["root_dir"]),
            model_path=Path(c["model_path"]),
            metrics_path=Path(c["metrics_path"]),
            processed_dir=Path(self.config["data_transformation"]["processed_dir"]),
            channels=tuple(m["enc_channels"]),               
            batch_size=int(p["batch_size"]),                  
            epochs=int(p["epochs"]),                         
            learning_rate=float(p["learning_rate"]),          
            patience=int(p["patience"]),                      
            val_split=float(p["val_split"]),                  
            seed=int(p["seed"]),                              
            noise_augmentation=float(p["noise_augmentation"])  
        )
    

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        c = self.config["model_evaluation"]
        processed_dir = self.config["data_transformation"]["processed_dir"]
        threshold = float(self.params["evaluation"]["auc_threshold"])
        return ModelEvaluationConfig(
            root_dir=Path(c["root_dir"]),
            metrics_path=Path(c["metrics_path"]),
            processed_dir=Path(processed_dir),
            auc_threshold=threshold
        )


    def get_quantization_config(self) -> QuantizationConfig:

        c = self.config["quantization"]
        p = self.params["quantization"]
        model_path = Path(self.config["model_trainer"]["model_path"])   
        processed_dir = Path(self.config["data_transformation"]["processed_dir"])  
        return QuantizationConfig(
            root_dir=Path(c["root_dir"]),
            golden_weights=Path(c["golden_weights"]),
            activation_ranges=Path(c["activation_ranges"]),
            bits=int(p["bits"]),
            calibration_samples=int(p["calibration_samples"]),
            percentile=float(p["percentile"]),
            model_path=model_path,
            processed_dir=processed_dir
        )
