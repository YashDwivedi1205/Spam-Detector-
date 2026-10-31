import sys
from src.components import data_ingestion, data_transformation, data_validation, model_trainer
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelTrainerConfig, TrainingPipelineConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact
from src.exception import SpamDetectorException
from src.logger import logging

class TrainPipeline:
    def __init__(self):
        self.pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_artifact = data_ingestion.DataIngestion(DataIngestionConfig(self.pipeline_config)).initiate_data_ingestion()
            return self.data_ingestion_artifact
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def start_data_validation(self, data_ingestion_artifact) -> DataValidationArtifact:
        try:
            self.data_validation_artifact = data_validation.DataValidation(data_ingestion_artifact, DataValidationConfig(self.pipeline_config)).initiate_data_validation()
            return self.data_validation_artifact
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def start_data_transformation(self, data_validation_artifact) -> DataTransformationArtifact:
        try:
            self.data_transformation_artifact = data_transformation.DataTransformation(data_validation_artifact, DataTransformationConfig(self.pipeline_config)).initiate_data_transformation()
            return self.data_transformation_artifact
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def start_model_trainer(self, data_transformation_artifact) -> ModelTrainerArtifact:
        try:
            self.model_trainer_artifact = model_trainer.ModelTrainer(data_transformation_artifact, ModelTrainerConfig(self.pipeline_config)).initiate_model_trainer()
            return self.model_trainer_artifact
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def run_pipeline(self) -> None:
        try:
            ingestion_artifact = self.start_data_ingestion()
            logging.info("Data Ingestion Completed successfully")
            validation_artifact = self.start_data_validation(ingestion_artifact)
            logging.info("Data validation completed successfully")
            transformation_artifact = self.start_data_transformation(validation_artifact)
            logging.info("Data Transformation completed successfully.")
            trainer_artifact = self.start_model_trainer(transformation_artifact)
            logging.info(f"Model Training completed successfully. {trainer_artifact}")       
        except Exception as e:
            raise SpamDetectorException(e, sys)

if __name__ =="__main__":
    TrainPipeline().run_pipeline()
    