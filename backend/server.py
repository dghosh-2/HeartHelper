from flask import Flask, request, jsonify
from flask_cors import CORS
from model import load_model, predict

app = Flask(__name__)
CORS(app)

model, scaler, feature_names = load_model()

@app.route("/api/predict", methods=["POST"])
def make_prediction():
    data = request.json
    result = predict(data, model, scaler, feature_names)
    return jsonify(result)

@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "name": "HeartHelper Neural Network",
        "version": "2.0",
        "architecture": "Input → 16 → 8 → 1",
        "threshold": 0.4,
        "features": {
            "input_features": 13,
            "engineered_features": 5,
            "total_after_encoding": len(feature_names)
        },
        "training": {
            "dataset": "Cleveland Heart Disease Dataset",
            "samples": 297,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss (weighted)",
            "early_stopping": "Validation Recall"
        },
        "description": "A PyTorch neural network trained to predict heart disease risk. Uses feature engineering including age-heart rate ratio, cholesterol-age interaction, and exercise risk scores. The model uses a 0.4 threshold (instead of 0.5) to prioritize catching disease cases (higher recall)."
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(port=5001, debug=False)
