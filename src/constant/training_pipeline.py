import os
## Defining constants in this file for training process
DATABASE_NAME = "SpamDetection"
COLLECTION_NAME = "SpamCollect"

# PIPELINE LEVEL CONSTANTS
PIPELINE_NAME = "spam_detection"
ARTIFACT_DIR = "artifact"

#file names
FILE_NAME = "spamham_final.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

#COLUMNS NAMES
TARGET_COLUMN = "Label"

#lOGGING
LOG_DIR = "logs"
LOG_FILE = ""

#Ingestion Dir
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

FEATURE_COLUMN = "Message"
DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_ARRAY_DIR = "transformed"
SAVED_TRANSFORMED_OBJECT = "transformed_object"
TRANSFORMED_TRAIN_FILE_NAME = "train.npz"
TRANSFORMED_TEST_FILE_NAME = "test.npz"
VECTORIZER_OBJECT_DIR = "vectorizer.pkl"
ENCODER_OBJECT_DIR = "encoder.pkl"

MODEL_TRAINER_DIR = "model_trainer"
TRAINED_MODEL_DIR = "trained_model"
MODEL_FILE_NAME = "model.pkl"
MODEL_CONFIG_PATH = os.path.join("config", "model.yaml")
EXPECTED_SCORE = 0.9

PREDICTION_MODEL_PATH = os.path.join("saved_models", "model.pkl")
