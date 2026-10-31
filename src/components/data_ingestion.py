import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import SpamDetectorException
from src.data_access.spam_data import SpamData
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.logger import logging
from src.constant.training_pipeline import COLLECTION_NAME, TARGET_COLUMN

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        try:
            logging.info("Exporting data from MongoDB")
            spam_data = SpamData()
            dataframe = spam_data.export_collection_as_dataframe(collection_name= COLLECTION_NAME)
            feature_store_file_path = self.data_ingestion_config.feature_store_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok= True)
            dataframe.to_csv(feature_store_file_path,index=False, header= True)
            return dataframe
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def split_data_as_train_test(self, df:pd.DataFrame) -> None:
        try:
            logging.info("Performing Train-Test split")
            train_set, test_set = train_test_split(df, test_size= self.data_ingestion_config.train_test_split_ratio,
                                                   random_state=42, stratify= df[TARGET_COLUMN])
            train_path = self.data_ingestion_config.training_file_path
            test_path = self.data_ingestion_config.testing_file_path

            os.makedirs(os.path.dirname(train_path), exist_ok= True)
            train_set = train_set.to_csv(train_path, index = False, header = True)
            test_set = test_set.to_csv(test_path, index = False, header = True)
            logging.info("Train-Test split completed.")

        except Exception as e:
            raise SpamDetectorException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:

            logging.info("Initiating data ingestion")
            df = self.export_data_into_feature_store()
            self.split_data_as_train_test(df=df)
            logging.info("Data Ingestion completed successfully.")
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path= self.data_ingestion_config.training_file_path,
                test_file_path= self.data_ingestion_config.testing_file_path
            )
            logging.info("Data ingestion completed successfully")
            return data_ingestion_artifact
        except Exception as e:
            raise SpamDetectorException(e,sys)

if __name__ == '__main__':
    from src.entity.config_entity import TrainingPipelineConfig

    config = DataIngestionConfig(TrainingPipelineConfig())
    artifact = DataIngestion(config).initiate_data_ingestion()
    print(artifact)