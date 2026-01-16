import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from sklearn.preprocessing import StandardScaler 
# standardize data to z scores

from sklearn.model_selection import train_test_split
#spit data into train/test/validation

from torch.utils.data import DataLoader, TensorDataset
#load and shuffle data effectively

from torch.optim.lr_scheduler import StepLR
#slow the learning rate over time to avoid overshooting towards the end

import pickle
import os

RANDOM_SEED = 42
# for reproducability 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#GPU if available, else CPU

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["cp", "restecg", "slope", "ca", "thal", "age_group"]
BINARY_COLS = ["sex", "fbs", "exang"]
# split data into continuous, categorical, and binary
# standardize continuous, one hot encode categorical, binarize binary data

ENGINEERED_CONTINUOUS = ["age_thalach_ratio", "chol_age", "oldpeak_slope", "exercise_risk"]
# manually engineered/created (continuous) featues to improve learning and performance

TARGET_COL = "condition"
#what we are concerned with, heart disease or not (0 or 1)

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
        # defining the characteristics of the entire neural network
        # introduce batch normalization to improve learning, stablity
        # introduce droout to prevent overfitting
        # 

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

def preprocess_data(df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = True):
    df_processed = df.copy()
    df_encoded = pd.get_dummies(df_processed, columns=CATEGORICAL_COLS, drop_first=False)
    
    if TARGET_COL in df_encoded.columns:
        X = df_encoded.drop(columns=[TARGET_COL])
        y = df_encoded[TARGET_COL].values.astype(np.float32)
    else:
        X = df_encoded
        y = None
    
    feature_names = X.columns.tolist()
    all_continuous = CONTINUOUS_COLS + ENGINEERED_CONTINUOUS
    cols_to_scale = [col for col in all_continuous if col in X.columns]
    
    if fit_scaler:
        scaler = StandardScaler()
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    else:
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])
    
    X = X.values.astype(np.float32)
    return X, y, scaler, feature_names

def train_model():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, "..", "heart_cleveland_upload.csv"))
    df = engineer_features(df)
    X, y, scaler, feature_names = preprocess_data(df, fit_scaler=True)
    
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    pos_weight = n_neg / n_pos
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_SEED)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1)), batch_size=16, shuffle=False)
    
    model = HeartDiseaseNet(input_dim=X.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    
    best_val_recall = 0.0
    patience_counter = 0
    best_state = None
    threshold = 0.4
    
    for epoch in range(200):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_tp, val_pos = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = (torch.sigmoid(model(X_batch)) >= threshold).float()
                val_tp += ((preds == 1) & (y_batch == 1)).sum().item()
                val_pos += (y_batch == 1).sum().item()
        
        val_recall = val_tp / val_pos if val_pos > 0 else 0
        scheduler.step()
        
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 25:
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    os.makedirs(os.path.join(script_dir, "saved"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(script_dir, "saved", "model.pth"))
    with open(os.path.join(script_dir, "saved", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(script_dir, "saved", "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)
    
    return model, scaler, feature_names

def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    saved_dir = os.path.join(script_dir, "saved")
    
    if not os.path.exists(os.path.join(saved_dir, "model.pth")):
        return train_model()
    
    with open(os.path.join(saved_dir, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    with open(os.path.join(saved_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    
    model = HeartDiseaseNet(input_dim=len(feature_names)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(saved_dir, "model.pth"), map_location=DEVICE, weights_only=True))
    model.eval()
    
    return model, scaler, feature_names

def predict(input_data: dict, model, scaler, feature_names):
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

if __name__ == "__main__":
    train_model()
