import os
from dataclasses import dataclass
from datetime import datetime
from src.constant.training_pipeline import *

class TrainingPipelineConfig:
    def __init__(self):
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline = PIPELINE_NAME
        self.artifact_dir = os.path.join(ARTIFACT_DIR, timestamp)

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
        self.feature_store_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
        self.training_file_path = os.path.join(self.data_ingestion_dir, TRAIN_FILE_NAME)
        self.testing_file_path = os.path.join(self.data_ingestion_dir, TEST_FILE_NAME)
        self.train_test_split_ratio = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.schema_file_path = SCHEMA_FILE_PATH



class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_ARRAY_DIR, TRANSFORMED_TRAIN_FILE_NAME)
        self.transformed_test_file_path = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_ARRAY_DIR, TRANSFORMED_TEST_FILE_NAME)
        self.transformed_vectorizer_object_file_path = os.path.join(self.data_transformation_dir, SAVED_TRANSFORMED_OBJECT, VECTORIZER_OBJECT_DIR)
        self.transformed_encoder_object_file_path = os.path.join(self.data_transformation_dir, SAVED_TRANSFORMED_OBJECT, ENCODER_OBJECT_DIR)

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR)
        self.trained_model_file_path = os.path.join(self.model_trainer_dir, TRAINED_MODEL_DIR, MODEL_FILE_NAME)
        self.expected_accuracy = EXPECTED_SCORE
        self.model_config_file_path = MODEL_CONFIG_PATH
    
