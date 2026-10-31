import sys
from src.utils.text_utils import clean_text
from src.utils.main_utils import load_object
from src.exception import SpamDetectorException
from src.logger import logging

class SpamClassifier:
    def __init__(self, vectorizer, model, encoder):
        self.vectorizer = vectorizer
        self.model = model
        self.encoder = encoder

    def predict(self, texts:list) -> list:
        try:
            cleaned_text = [clean_text(text) for text in texts]
            vectorized_text = self.vectorizer.transform(cleaned_text)

            predicted = self.model.predict(vectorized_text)
            labels = self.encoder.inverse_transform(predicted)

            spam_index = list(self.encoder.classes_).index("spam")
            spam_prob = self.model.predict_proba(vectorized_text)[:, spam_index]

            result = []
            for label, prob in zip(labels, spam_prob):
                result.append({
                    "Label": str(label),
                    "Spam Probability": round(float(prob), 4),
                })
            return result
        except Exception as e:
            raise SpamDetectorException(e, sys)