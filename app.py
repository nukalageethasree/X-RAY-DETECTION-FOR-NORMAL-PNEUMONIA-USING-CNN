from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import keras
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "chest_xray_cnn.keras")
model = keras.models.load_model(MODEL_PATH)

IMG_SIZE   = (150, 150)
THRESHOLD  = 0.5          # sigmoid threshold


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize, convert to RGB, normalise → (1, 150, 150, 3)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "chest_xray_cnn", "input_shape": list(IMG_SIZE)})


@app.route("/predict", methods=["POST"])
def predict():
    # ── accept multipart/form-data OR base64 JSON ──────────────────────────
    try:
        if "file" in request.files:
            image_bytes = request.files["file"].read()
        elif request.is_json and "image" in request.json:
            header, encoded = request.json["image"].split(",", 1)
            image_bytes = base64.b64decode(encoded)
        else:
            return jsonify({"error": "No image provided. Send 'file' (multipart) or 'image' (base64 JSON)."}), 400

        img_array = preprocess_image(image_bytes)
        prob      = float(model.predict(img_array, verbose=0)[0][0])   # sigmoid output

        label      = "PNEUMONIA" if prob >= THRESHOLD else "NORMAL"
        confidence = prob if label == "PNEUMONIA" else 1 - prob

        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2),   # percent
            "raw_score":  round(prob, 6),
            "threshold":  THRESHOLD,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
