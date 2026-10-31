from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: str
    valid_test_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path: str
    transformed_vectorizer_object_path: str
    transformed_encoder_object_path: str

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_accuracy: float
    test_accuracy: float
    test_precision: float
    test_recall: float