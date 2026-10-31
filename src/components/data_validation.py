import sys
import pandas as pd
from src.utils.main_utils import read_yaml_file
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.exception import SpamDetectorException
from src.logger import logging

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = read_yaml_file(self.data_validation_config.schema_file_path)
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def validate_columns(self, df:pd.DataFrame) -> bool:
        try:
            status = set(df.columns) == set(self.schema["columns"])
            if status:
                logging.info("Columns matched")
            else:
                logging.warning(f"Columns mismatched. Found: {list(df.columns)}")
            return status
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def validate_no_nulls(self, df: pd.DataFrame) -> bool:
        try:
            null_count = df.isna().sum().sum()
            status = null_count == 0
            if status:
                logging.info("No null value present")
            else:
                logging.warning(f"Null values found: {null_count}")
            return bool(status)
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:

            train_df = pd.read_csv(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_cols_ok = self.validate_columns(train_df)
            test_cols_ok = self.validate_columns(test_df)
            train_nulls_ok = self.validate_no_nulls(train_df)
            test_nulls_ok = self.validate_no_nulls(test_df)

            validation_status = train_cols_ok and test_cols_ok and train_nulls_ok and test_nulls_ok

            logging.info(f"Data validation status: {validation_status}")
            data_validation_Artifact = DataValidationArtifact(
                validation_status= validation_status,
                valid_train_file_path= self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path= self.data_ingestion_artifact.test_file_path
            )
            return data_validation_Artifact
        except Exception as e:
            raise SpamDetectorException(e, sys)

if __name__ == "__main__":
    from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
    from src.components.data_ingestion import DataIngestion

    ingestion_artifact = DataIngestion(
        DataIngestionConfig(TrainingPipelineConfig())
    ).initiate_data_ingestion()
    validation_artifact = DataValidation(
        ingestion_artifact, DataValidationConfig()
    ).initiate_data_validation()
    print(validation_artifact)
