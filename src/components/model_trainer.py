import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.utils.main_utils import read_yaml_file, load_sparse_matrix, save_object
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception import SpamDetectorException
from src.logger import logging
from src.utils.main_utils import load_object
from src.ml.model.estimator import SpamClassifier

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.model_config = read_yaml_file(self.model_trainer_config.model_config_file_path)

    def split_x_y(self, matrix):
        X,y = matrix[:, :-1], matrix[:, -1].toarray().ravel().astype(int)
        return X, y
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_matrix = load_sparse_matrix(self.data_transformation_artifact.transformed_train_path)
            test_matrix = load_sparse_matrix(self.data_transformation_artifact.transformed_test_path)
            X_train, y_train = self.split_x_y(train_matrix)
            X_test, y_test = self.split_x_y(test_matrix)
            grid_cfg = self.model_config["grid_search"]

            grid_search = GridSearchCV(estimator=MultinomialNB(), 
                                       param_grid=self.model_config["param_grid"], 
                                       cv=grid_cfg['cv'], scoring= grid_cfg['scoring'], refit=grid_cfg['refit'])
            best_model = grid_search.fit(X_train, y_train).best_estimator_
            y_pred = best_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)
            logging.info(f"Best params: {grid_search.best_params_}")
            logging.info(f"Test accuracy={test_accuracy:.4f}, precision={test_precision:.4f}, recall={test_recall:.4f}")
            if test_accuracy < self.model_trainer_config.expected_accuracy:
                raise Exception("Warning!! Desired Accuracy not achieved.")
            
            vectorizer = load_object(self.data_transformation_artifact.transformed_vectorizer_object_path)
            encoder = load_object(self.data_transformation_artifact.transformed_encoder_object_path)
            classifer = SpamClassifier(vectorizer, best_model, encoder)
            save_object(self.model_trainer_config.trained_model_file_path, classifer)
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_accuracy= grid_search.cv_results_["mean_test_accuracy"][grid_search.best_index_],
                test_accuracy= test_accuracy,
                test_precision= test_precision,
                test_recall= test_recall
            )
            return model_trainer_artifact
        except Exception as e:
            raise SpamDetectorException(e, sys)


if __name__ == "__main__":
    from src.entity.config_entity import (TrainingPipelineConfig, DataIngestionConfig,
                                          DataValidationConfig, DataTransformationConfig)
    from src.components.data_ingestion import DataIngestion
    from src.components.data_validation import DataValidation
    from src.components.data_transformation import DataTransformation
    from src.utils.main_utils import load_object

    pipeline_config = TrainingPipelineConfig()

    ingestion_artifact = DataIngestion(DataIngestionConfig(pipeline_config)).initiate_data_ingestion()
    validation_artifact = DataValidation(ingestion_artifact, DataValidationConfig(pipeline_config)).initiate_data_validation()
    transformation_artifact = DataTransformation(validation_artifact, DataTransformationConfig(pipeline_config)).initiate_data_transformation()
    trainer_artifact = ModelTrainer(transformation_artifact, ModelTrainerConfig(pipeline_config)).initiate_model_trainer()

    print(trainer_artifact)
    print(load_object(trainer_artifact.trained_model_file_path))
            