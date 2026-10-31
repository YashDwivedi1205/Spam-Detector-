import sys
from src.utils.main_utils import load_object
from src.constant.training_pipeline import PREDICTION_MODEL_PATH
from src.exception import SpamDetectorException
from src.logger import logging

class PredictionPipeline:
    def __init__(self, model_path: str = PREDICTION_MODEL_PATH):
        try:
            self.model = load_object(model_path)
            logging.info(f"Model loaded from {model_path}")
        except Exception as e:
            raise SpamDetectorException(e, sys)

    def predict_one(self, text:str) -> dict:
        if not text or not text.strip():
            raise ValueError("Text is empty.Please enter correct text.")
        try:
            predicted = self.model.predict([text])
            return predicted[0]
        except Exception as e:
            raise SpamDetectorException(e, sys)