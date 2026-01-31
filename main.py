from flask import Flask, render_template, request, send_from_directory
import os

from app.components.model_loader import ModelLoader
from app.components.predictor import TumorPredictor
from app.utils.logger import logger

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

model = ModelLoader("models/model.h5").load()
predictor = TumorPredictor(model)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            result, confidence = predictor.predict(path)
            return render_template(
                "index.html",
                result=result,
                confidence=f"{confidence*100:.2f}%",
                file_path=f"/uploads/{file.filename}"
            )
    return render_template("index.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    logger.info("Starting Flask app")
    app.run(host="0.0.0.0", port=5000)
