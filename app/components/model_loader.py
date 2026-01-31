from tensorflow.keras.models import load_model
from app.utils.logger import logger
from app.utils.exception import CustomException
import sys

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path

    def load(self):
        try:
            logger.info("Loading ML model")
            model = load_model(self.model_path)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error("Error while loading model")
            raise CustomException(e, sys)
