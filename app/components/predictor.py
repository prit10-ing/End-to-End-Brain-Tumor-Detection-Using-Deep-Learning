import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array
from app.utils.logger import logger

class TumorPredictor:
    def __init__(self, model):
        self.model = model
        self.image_size = 128
        self.class_labels = ["pituitary", "glioma", "notumor", "meningioma"]

    def predict(self, image_path):
        try:
            if not os.path.exists(image_path):
                raise ValueError("Image not found")

            img = load_img(
                image_path,
                target_size=(self.image_size, self.image_size)
            )

            img_array = img_to_array(img)
            img_array = img_array.astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = self.model.predict(img_array)

            if preds.ndim != 2:
                raise ValueError("Invalid model output")

            index = int(np.argmax(preds[0]))
            confidence = float(np.max(preds[0]))

            label = self.class_labels[index]

            logger.info(f"Prediction: {label}, Confidence: {confidence}")

            if label == "notumor":
                return "No Tumor", confidence

            return f"Tumor: {label}", confidence

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return "Prediction Error", 0.0
