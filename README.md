MRI Brain Tumor Detection using Transfer LearningThis project implements a high-precision medical imaging classifier to identify anomalies from MRI diagnostic images. By leveraging the VGG16 architecture and custom fine-tuning, the model achieves state-of-the-art performance in tumor classification.


🚀 Performance MetricsBased on the latest training logs, the model demonstrates exceptional reliability:Training Accuracy: 99.08% over 10 epochs.Weighted Average F1-Score: 0.97.Precision: Perfect classification (1.00) for specific tumor categories.

🛠️ Technical FeaturesArchitecture: Transfer Learning with VGG16 backbone.Fine-Tuning: Unfrozen final convolutional blocks with a custom Sequential head featuring Dropout layers (0.3 and 0.2) to prevent overfitting.Augmentation Pipeline: Engineered using PIL and ImageEnhance to dynamically adjust brightness and contrast, ensuring robustness against variations in scan quality.Data Handling: Implemented an efficient Python Data Generator for memory-intensive batch processing and real-time normalization.

📂 Project StructurePlaintext├── app.py              # Flask/Web application for real-time inference 
├── model/              # Saved VGG16 model weights and architecture
├── static/             # CSS and image assets for the web UI
├── templates/          # HTML templates for the Flask front-end
├── src/                # Modular source code for training and preprocessing
└── requirements.txt    # Project dependencies


💻 How to Run1. Clone the RepositoryBashgit clone https://github.com/prit10-ing/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
2. Set Up EnvironmentIt is recommended to use a virtual environment:Bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3. Run the ApplicationThe project includes a web interface for real-time medical image processing:Bashpython app.py
Once the server starts, navigate to http://127.0.0.1:5000 in your web browser.4. Real-time InferenceUpload an MRI scan through the dashboard.The model will process the image array and return the tumor classification with the associated confidence score.🧪 Technologies UsedDeep Learning: TensorFlow, Keras Computer Vision: OpenCV, PIL Web Framework: Flask Analysis: NumPy, Matplotlib, Seaborn 