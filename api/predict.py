from http.server import BaseHTTPRequestHandler
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
import base64

DEVICE = torch.device("cpu")

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["cp", "restecg", "slope", "ca", "thal", "age_group"]
ENGINEERED_CONTINUOUS = ["age_thalach_ratio", "chol_age", "oldpeak_slope", "exercise_risk"]

class HeartDiseaseNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim_1: int = 16, hidden_dim_2: int = 8, dropout_rate: float = 0.25):
        super(HeartDiseaseNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        self.layer2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        self.layer3 = nn.Linear(hidden_dim_2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.relu(self.bn1(self.layer1(x))))
        x = self.dropout(self.relu(self.bn2(self.layer2(x))))
        return self.layer3(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()
    df_eng["age_thalach_ratio"] = df_eng["age"] / (df_eng["thalach"] + 1)
    df_eng["age_group"] = df_eng["age"].apply(lambda a: "young" if a < 45 else ("middle" if a <= 60 else "senior"))
    df_eng["chol_age"] = df_eng["chol"] * df_eng["age"] / 1000
    df_eng["oldpeak_slope"] = df_eng["oldpeak"] * (df_eng["slope"] + 1)
    df_eng["exercise_risk"] = df_eng["exang"] * (220 - df_eng["age"] - df_eng["thalach"])
    return df_eng

_model = None
_scaler = None
_feature_names = None

def load_model():
    global _model, _scaler, _feature_names
    if _model is not None:
        return _model, _scaler, _feature_names
    
    api_dir = os.path.dirname(os.path.abspath(__file__))
    saved_dir = os.path.join(api_dir, "saved")
    
    with open(os.path.join(saved_dir, "feature_names.pkl"), "rb") as f:
        _feature_names = pickle.load(f)
    with open(os.path.join(saved_dir, "scaler.pkl"), "rb") as f:
        _scaler = pickle.load(f)
    
    _model = HeartDiseaseNet(input_dim=len(_feature_names)).to(DEVICE)
    _model.load_state_dict(torch.load(os.path.join(saved_dir, "model.pth"), map_location=DEVICE, weights_only=True))
    _model.eval()
    
    return _model, _scaler, _feature_names

def predict(input_data: dict):
    model, scaler, feature_names = load_model()
    
    df = pd.DataFrame([input_data])
    df = engineer_features(df)
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    
    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_names]
    
    all_continuous = CONTINUOUS_COLS + ENGINEERED_CONTINUOUS
    cols_to_scale = [col for col in all_continuous if col in df_encoded.columns]
    df_encoded[cols_to_scale] = scaler.transform(df_encoded[cols_to_scale])
    
    X = torch.tensor(df_encoded.values.astype(np.float32)).to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        prob = model.predict_proba(X).item()
    
    return {"probability": prob, "prediction": int(prob >= 0.4)}

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        
        result = predict(data)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
