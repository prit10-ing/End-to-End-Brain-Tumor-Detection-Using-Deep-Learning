import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from app.utils.logger import logger
from app.utils.exception import CustomException
import sys

class TumorPredictor:
    def __init__(self, model):
        self.model = model
        self.class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

    def predict(self, image_path):
        try:
            IMAGE_SIZE = 128
            img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = self.model.predict(img_array)
            index = np.argmax(preds)
            confidence = np.max(preds)

            label = self.class_labels[index]
            logger.info(f"Prediction: {label}, Confidence: {confidence}")

            if label == "notumor":
                return "No Tumor", confidence
            return f"Tumor: {label}", confidence

        except Exception as e:
            logger.error("Prediction failed")
            raise CustomException(e, sys)
