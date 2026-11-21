import os

import mlflow
import mlflow.pyfunc
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = os.getenv("MODEL_NAME", "Model_Risiko_Saham_Ezekiel")
STAGE = os.getenv("MODEL_STAGE", "Staging")
MODEL_URI = f"models:/{MODEL_NAME}/{STAGE}"
EXPECTED_COLUMNS = ["Open", "High", "Low", "Volume"]

mlflow.set_tracking_uri(TRACKING_URI)
model = None


def load_model():
    """Memuat model terdaftar dari MLflow ke memori."""
    global model
    try:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    except Exception as exc:
        app.logger.warning("Gagal memuat model dari %s: %s", MODEL_URI, exc)
        model = None


load_model()


@app.route("/health", methods=["GET"])
def health():
    """Mengembalikan status layanan dan kesiapan model."""
    return jsonify({
        "status": "ok",
        "tracking_uri": TRACKING_URI,
        "model_uri": MODEL_URI,
        "model_loaded": model is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Menjalankan inferensi klasifikasi risiko saham dari input JSON."""
    if model is None:
        load_model()
    if model is None:
        return jsonify({"status": "error", "message": "Model tidak tersedia di server"}), 500

    try:
        request_data = request.get_json()
        if not request_data or "features" not in request_data:
            return jsonify({"status": "error", "message": "Payload tidak valid. Harus memiliki key 'features'"}), 400

        df = pd.DataFrame(request_data["features"])
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            return jsonify({"status": "error", "message": f"Fitur wajib tidak ditemukan: {missing_cols}"}), 400

        predictions = model.predict(df[EXPECTED_COLUMNS])
        return jsonify({
            "status": "success",
            "model_version": STAGE,
            "prediction": predictions.tolist(),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)