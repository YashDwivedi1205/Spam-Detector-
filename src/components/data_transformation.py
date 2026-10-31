import sys
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from src.utils.main_utils import save_object, save_sparse_matrix
from src.utils.text_utils import clean_text
from src.exception import SpamDetectorException
from src.logger import logging
from src.constant.training_pipeline import FEATURE_COLUMN, TARGET_COLUMN
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config

    def get_cleaned_text(self, df: pd.DataFrame) -> list:
        try:
            cleaned_str = df[FEATURE_COLUMN].apply(clean_text)
            return cleaned_str.to_list()
        except Exception as e:
            raise SpamDetectorException(e, sys)
    def get_vectorized_data(self, train_df, test_df):
        try:

            cleaned_train_data = self.get_cleaned_text(train_df)
            cleaned_test_data = self.get_cleaned_text(test_df)
            vectorizer = TfidfVectorizer()
            X_train = vectorizer.fit_transform(cleaned_train_data)
            X_test = vectorizer.transform(cleaned_test_data)
            return X_train, X_test, vectorizer

        except Exception as e:
            raise SpamDetectorException(e, sys)
        

    def get_encoded_target(self, train_df, test_df):
        try:

            y_train, y_test = train_df[TARGET_COLUMN], test_df[TARGET_COLUMN]
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)
            y_test = encoder.transform(y_test)
            return y_train, y_test, encoder
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception("Validation Failed. Please check and validate again.")
            train_csv = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_csv = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            X_trained, X_tested, vectorizer1 = self.get_vectorized_data(train_csv, test_csv)
            y_trained, y_tested, encoder1 = self.get_encoded_target(train_csv, test_csv)
            y_trained = y_trained.reshape(-1, 1)
            y_tested = y_tested.reshape(-1, 1)
            train_matrix = sparse.hstack([X_trained, y_trained]).tocsr()
            test_matrix = sparse.hstack([X_tested, y_tested]).tocsr()
            save_sparse_matrix(self.data_transformation_config.transformed_train_file_path, train_matrix)
            save_sparse_matrix(self.data_transformation_config.transformed_test_file_path, test_matrix)
            save_object(self.data_transformation_config.transformed_vectorizer_object_file_path, vectorizer1)
            save_object(self.data_transformation_config.transformed_encoder_object_file_path, encoder1)
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_path= self.data_transformation_config.transformed_train_file_path,
                transformed_test_path= self.data_transformation_config.transformed_test_file_path,
                transformed_vectorizer_object_path= self.data_transformation_config.transformed_vectorizer_object_file_path,
                transformed_encoder_object_path= self.data_transformation_config.transformed_encoder_object_file_path
            )
            logging.info("Data Transformation completed.")
            return data_transformation_artifact
        except Exception as e:
            raise SpamDetectorException(e, sys)
        
        



    
if __name__ == '__main__':
    # train_df = pd.read_csv("artifact/10_11_2026_16_28_40/data_ingestion/train.csv")
    # test_df = pd.read_csv("artifact/10_11_2026_16_28_40/data_ingestion/test.csv")
    # dt = DataTransformation(None, None)
    # X_train, X_test, vectorizer = dt.get_vectorized_data(train_df, test_df)
    # y_train, y_test, encoder = dt.get_encoded_target(train_df, test_df)

    # print(X_train.shape, X_test.shape)
    # print(y_train.shape, y_test.shape)
    # print(encoder.classes_)
    from src.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.utils.main_utils import load_sparse_matrix
    pipeline_config = TrainingPipelineConfig()
    ingestion_artifact = DataIngestion(DataIngestionConfig(pipeline_config)).initiate_data_ingestion()
    validation_artifact = DataValidation(ingestion_artifact, DataValidationConfig(pipeline_config)).initiate_data_validation()
    transformation_artifact = DataTransformation(
        validation_artifact, DataTransformationConfig(pipeline_config)).initiate_data_transformation()
    print(transformation_artifact)

    arr = load_sparse_matrix(transformation_artifact.transformed_train_path)
    print(arr.shape)
