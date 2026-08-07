from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_zenodo_id: str
    data_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_dir_root: Path          
    processed_dir: Path
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int
    max_frames: int


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    status_file: Path
    schema_path: Path
    processed_dir: Path        


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    model_path: Path
    metrics_path: Path
    processed_dir: Path        
    channels: tuple            
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    val_split: float
    seed: int
    noise_augmentation: float


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    metrics_path: Path
    processed_dir: Path        
    auc_threshold: float       


@dataclass(frozen=True)
class QuantizationConfig:
    root_dir: Path
    golden_weights: Path
    activation_ranges: Path
    bits: int
    calibration_samples: int
    percentile: float
    model_path: Path           
    processed_dir: Path        


@dataclass(frozen=True)
class RTLVerificationConfig:
    root_dir: Path
    test_vectors: Path
    golden_weights: Path       
    rtl_file: Path             
    test_file: Path           
