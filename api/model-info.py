from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        result = {
            "name": "HeartHelper Neural Network",
            "version": "2.0",
            "architecture": "Input → 16 → 8 → 1",
            "threshold": 0.4,
            "features": {
                "input_features": 13,
                "engineered_features": 5,
                "total_after_encoding": 32
            },
            "training": {
                "dataset": "Cleveland Heart Disease Dataset",
                "samples": 297,
                "optimizer": "Adam",
                "loss": "BCEWithLogitsLoss (weighted)",
                "early_stopping": "Validation Recall"
            },
            "description": "A PyTorch neural network trained to predict heart disease risk. Uses feature engineering including age-heart rate ratio, cholesterol-age interaction, and exercise risk scores. The model uses a 0.4 threshold (instead of 0.5) to prioritize catching disease cases (higher recall)."
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
