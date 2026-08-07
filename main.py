from src.edge_ai_anomaly_detection.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.edge_ai_anomaly_detection.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from src.edge_ai_anomaly_detection.pipeline.stage_03_data_validation import DataValidationTrainingPipeline
from src.edge_ai_anomaly_detection.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.edge_ai_anomaly_detection.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from src.edge_ai_anomaly_detection.pipeline.stage_06_quantization import QuantizationPipeline
from src.edge_ai_anomaly_detection.pipeline.stage_07_rtl_verification import RTLVerificationPipeline


def run_all():
    
    pipeline_steps = [
        ("Data Ingestion", DataIngestionTrainingPipeline),
        ("Data Transformation", DataTransformationTrainingPipeline),
        ("Data Validation", DataValidationTrainingPipeline),
        ("Model Training", ModelTrainerTrainingPipeline),
        ("Model Evaluation", ModelEvaluationPipeline),
        ("Quantization", QuantizationPipeline),
        ("RTL Verification", RTLVerificationPipeline),
    ]

    for name, cls in pipeline_steps:
        print(f"\n{'='*40}")
        print(f">>> STAGE: {name}")
        print(f"{'='*40}")
        try:
            cls().main()
            print(f">>> {name}: ✅ موفق")
        except Exception as e:
            print(f">>> {name}: ❌ خطا — {e}")
            print("⚠️ به دلیل خطا، زنجیره متوقف شد.")
            break


if __name__ == "__main__":
    run_all()
