"""
NutriScan AI — backend
Serves the custom HTML/CSS/JS frontend and exposes a /api/predict endpoint
that runs the existing MobileNet fruit-freshness model.
"""

import io
import pickle

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

app = Flask(__name__, static_folder="static", template_folder="templates")

MODEL_PATH = "model/fruit_mobilenet_final.h5"
CLASS_NAMES_PATH = "model/class_names.pkl"

model = None
class_names = None


def load_assets():
    """Load the Keras model and class labels once at startup."""
    global model, class_names
    import tensorflow as tf  # imported here so the app can still boot without TF installed, for quick frontend checks

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "rb") as f:
        class_names = pickle.load(f)


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded on the server."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image was provided."}), 400

    file = request.files["image"]

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Uploaded file is not a readable image."}), 400

    image = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(img_array)
    idx = int(np.argmax(prediction))
    predicted_class = class_names[idx]
    confidence = float(np.max(prediction)) * 100

    quality = "Fresh" if "fresh" in predicted_class.lower() else "Rotten"

    return jsonify({
        "quality": quality,
        "confidence": round(confidence, 2),
        "raw_class": predicted_class,
    })


if __name__ == "__main__":
    load_assets()
    app.run(debug=True, port=5000)
