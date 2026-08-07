from src.edge_ai_anomaly_detection.config.configuration import ConfigurationManager

def test_all_entities_build():
    
    cm = ConfigurationManager()
    cm.get_data_ingestion_config()
    cm.get_data_transformation_config()
    cm.get_data_validation_config()   
    cm.get_model_trainer_config()
    cm.get_model_evaluation_config()
    cm.get_quantization_config()
    assert True
