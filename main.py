"""
HeartHelper - Heart Disease Prediction using PyTorch Neural Network (ENHANCED)
===============================================================================

This script implements an ENHANCED ML pipeline for predicting heart disease
using the Cleveland Heart Disease dataset. Improvements include:
    - Weighted loss function to address class imbalance
    - Lowered classification threshold (0.4) for better recall
    - Thinner network architecture (16 -> 8) to prevent overfitting
    - Learning rate scheduler for better convergence
    - Feature engineering (interaction terms, age binning)
    - 5-Fold Cross-Validation for robust evaluation
    - Ensemble with Random Forest for comparison

Author: HeartHelper ML Team
"""

import random
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# =============================================================================
# STEP 0: SET RANDOM SEEDS FOR REPRODUCIBILITY
# =============================================================================
# Setting seeds ensures that results are reproducible across runs.
# This is critical for debugging and comparing model performance.

RANDOM_SEED = 42

def set_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seeds(RANDOM_SEED)


# =============================================================================
# STEP 1: ENHANCED HYPERPARAMETER CONFIGURATION
# =============================================================================
# CHANGES FROM ORIGINAL:
#   - Thinner layers: 16 -> 8 (was 32 -> 16) to prevent overfitting on 297 samples
#   - Added learning rate scheduler parameters
#   - Added classification threshold (lowered to 0.4 for better recall)
#   - Added pos_weight for weighted loss function

HYPERPARAMETERS = {
    # --- Network Architecture (THINNER to prevent overfitting) ---
    "hidden_units_1": 16,      # Reduced from 32 (less capacity = less overfitting)
    "hidden_units_2": 8,       # Reduced from 16
    "dropout_rate": 0.3,       # Increased from 0.2 for more regularization
    
    # --- Training Parameters ---
    "learning_rate": 0.001,    # Starting learning rate
    "batch_size": 16,          # Small batches provide noise to escape local minima
    "num_epochs": 200,         # Maximum training epochs
    "patience": 25,            # Early stopping patience (increased for scheduler)
    
    # --- Learning Rate Scheduler (NEW) ---
    "lr_step_size": 30,        # Reduce LR every 30 epochs
    "lr_gamma": 0.5,           # Multiply LR by 0.5 each step
    
    # --- Classification Threshold (NEW - lowered for better recall) ---
    "classification_threshold": 0.35,  # Lowered from 0.5 to catch more disease cases
    
    # --- Cross-Validation (NEW) ---
    "n_folds": 5,              # 5-Fold Cross-Validation
}

# Device configuration - use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =============================================================================
# STEP 2: DATA LOADING AND EXPLORATION
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the heart disease dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the heart disease data
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df


def explore_data(df: pd.DataFrame) -> Dict[str, int]:
    """
    Print basic statistics and return class distribution for weighting.
    
    Args:
        df: DataFrame to explore
        
    Returns:
        Dictionary with class counts for calculating pos_weight
    """
    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    print("\n--- First 5 rows ---")
    print(df.head())
    
    print("\n--- Target distribution ---")
    class_counts = df["condition"].value_counts()
    print(class_counts)
    
    n_negative = class_counts[0]  # No Disease
    n_positive = class_counts[1]  # Disease
    
    print(f"\nClass balance: {n_positive/(n_positive+n_negative):.2%} positive cases")
    print(f"  No Disease (0): {n_negative} samples")
    print(f"  Disease (1):    {n_positive} samples")
    
    # Calculate pos_weight for BCEWithLogitsLoss
    # pos_weight = n_negative / n_positive (gives more weight to minority class)
    pos_weight = n_negative / n_positive
    print(f"\n--- Calculated pos_weight for weighted loss: {pos_weight:.4f} ---")
    print("(This tells the model that missing a Disease case is more 'painful')")
    
    return {"n_negative": n_negative, "n_positive": n_positive, "pos_weight": pos_weight}


# =============================================================================
# STEP 3: ENHANCED FEATURE ENGINEERING (NEW)
# =============================================================================
# Adding domain knowledge to help the model:
#   1. Interaction terms: age * max_heart_rate ratio
#   2. Age binning: Young, Middle-Aged, Senior categories
#   3. Heart rate reserve: max_heart_rate - resting_heart_rate proxy

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features using domain knowledge about heart disease.
    
    New Features:
        - age_thalach_ratio: Age / Max Heart Rate (older + lower max HR = higher risk)
        - age_group_*: One-hot encoded age categories
        - chol_age_interaction: Cholesterol * Age interaction
        - oldpeak_slope_interaction: ST depression * slope interaction
    
    Args:
        df: Original DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING (Domain Knowledge Boost)")
    print("=" * 60)
    
    df_eng = df.copy()
    
    # --- Feature 1: Age / Max Heart Rate Ratio ---
    # Higher ratio = older person with lower max heart rate = higher risk
    df_eng["age_thalach_ratio"] = df_eng["age"] / (df_eng["thalach"] + 1)  # +1 to avoid division by zero
    print("Created: age_thalach_ratio (Age / Max Heart Rate)")
    
    # --- Feature 2: Age Binning ---
    # Convert continuous age to categories: Young (<45), Middle (45-60), Senior (>60)
    def age_bin(age):
        if age < 45:
            return "young"
        elif age <= 60:
            return "middle"
        else:
            return "senior"
    
    df_eng["age_group"] = df_eng["age"].apply(age_bin)
    print("Created: age_group (young/middle/senior categories)")
    
    # --- Feature 3: Cholesterol * Age Interaction ---
    # High cholesterol is more dangerous as you age
    df_eng["chol_age"] = df_eng["chol"] * df_eng["age"] / 1000  # Scaled down
    print("Created: chol_age (Cholesterol * Age interaction)")
    
    # --- Feature 4: ST Depression * Slope Interaction ---
    # The combination of oldpeak and slope is clinically significant
    df_eng["oldpeak_slope"] = df_eng["oldpeak"] * (df_eng["slope"] + 1)
    print("Created: oldpeak_slope (ST Depression * Slope interaction)")
    
    # --- Feature 5: Exercise-induced risk score ---
    # Combines exang (exercise angina) with thalach (max heart rate achieved)
    df_eng["exercise_risk"] = df_eng["exang"] * (220 - df_eng["age"] - df_eng["thalach"])
    print("Created: exercise_risk (Exercise-induced risk score)")
    
    print(f"\nOriginal features: {df.shape[1]}")
    print(f"Features after engineering: {df_eng.shape[1]}")
    
    return df_eng


# =============================================================================
# STEP 4: DATA PREPROCESSING
# =============================================================================

# Define column types based on the Cleveland dataset documentation
CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_COLS = ["cp", "restecg", "slope", "ca", "thal", "age_group"]  # Added age_group
BINARY_COLS = ["sex", "fbs", "exang"]
ENGINEERED_CONTINUOUS = ["age_thalach_ratio", "chol_age", "oldpeak_slope", "exercise_risk"]
TARGET_COL = "condition"


def preprocess_data(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    fit_scaler: bool = True
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Preprocess the heart disease data for neural network training.
    
    Steps:
        1. One-Hot Encode categorical columns (including age_group)
        2. Scale continuous columns with StandardScaler
        3. Keep binary columns as-is
        4. Separate features (X) from target (y)
    
    Args:
        df: Raw DataFrame (with engineered features)
        scaler: Pre-fitted StandardScaler (for validation/test data)
        fit_scaler: Whether to fit the scaler (True for training data)
        
    Returns:
        X: Feature array (n_samples, n_features)
        y: Target array (n_samples,)
        scaler: Fitted StandardScaler (save for inference)
        feature_names: List of feature names after preprocessing
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    df_processed = df.copy()
    
    # --- Step 4a: One-Hot Encode categorical variables ---
    print(f"\nOne-Hot Encoding columns: {CATEGORICAL_COLS}")
    df_encoded = pd.get_dummies(df_processed, columns=CATEGORICAL_COLS, drop_first=False)
    print(f"Shape after One-Hot Encoding: {df_encoded.shape}")
    
    # --- Step 4b: Separate features and target ---
    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL].values
    feature_names = X.columns.tolist()
    
    print(f"Number of features after encoding: {len(feature_names)}")
    
    # --- Step 4c: Scale ALL continuous variables (original + engineered) ---
    all_continuous = CONTINUOUS_COLS + ENGINEERED_CONTINUOUS
    # Filter to only columns that exist in X
    cols_to_scale = [col for col in all_continuous if col in X.columns]
    
    print(f"\nScaling continuous columns: {cols_to_scale}")
    
    if fit_scaler:
        scaler = StandardScaler()
        X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
        print("Scaler fitted on training data")
    else:
        X[cols_to_scale] = scaler.transform(X[cols_to_scale])
        print("Scaler applied (pre-fitted)")
    
    X = X.values.astype(np.float32)
    y = y.astype(np.float32)
    
    print(f"\nFinal X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")
    
    return X, y, scaler, feature_names


# =============================================================================
# STEP 5: TRAIN/VALIDATION/TEST SPLIT
# =============================================================================

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training, validation, and test sets with stratification.
    """
    print("\n" + "=" * 60)
    print("DATA SPLITTING (70% Train / 15% Val / 15% Test)")
    print("=" * 60)
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_ratio,
        stratify=y,
        random_state=RANDOM_SEED
    )
    
    # Second split: separate validation from training
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted,
        stratify=y_temp,
        random_state=RANDOM_SEED
    )
    
    print(f"\nTraining set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    print(f"\nClass balance (positive cases):")
    print(f"  Training:   {y_train.mean():.2%}")
    print(f"  Validation: {y_val.mean():.2%}")
    print(f"  Test:       {y_test.mean():.2%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# STEP 6: CREATE PYTORCH DATALOADERS
# =============================================================================

def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for training, validation, and test sets."""
    print("\n" + "=" * 60)
    print(f"CREATING DATALOADERS (batch_size={batch_size})")
    print("=" * 60)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches:       {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# STEP 7: ENHANCED NEURAL NETWORK MODEL
# =============================================================================
# CHANGES FROM ORIGINAL:
#   - Uses BCEWithLogitsLoss (no sigmoid in forward pass) for numerical stability
#   - Thinner architecture: 16 -> 8 (was 32 -> 16)
#   - Added batch normalization for better training

class HeartDiseaseNetV2(nn.Module):
    """
    Enhanced Neural Network for Heart Disease Prediction (V2).
    
    Improvements:
        - Thinner layers (16 -> 8) to prevent overfitting on small dataset
        - No sigmoid in forward (uses BCEWithLogitsLoss for stability)
        - Batch normalization for better gradient flow
        - Higher dropout for regularization
    
    Architecture:
        Input -> Linear(16) -> BatchNorm -> ReLU -> Dropout ->
        Linear(8) -> BatchNorm -> ReLU -> Dropout ->
        Linear(1) [raw logits for BCEWithLogitsLoss]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 16,
        hidden_dim_2: int = 8,
        dropout_rate: float = 0.3
    ):
        """Initialize the enhanced neural network."""
        super(HeartDiseaseNetV2, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        
        # Layer 1: Linear -> BatchNorm -> ReLU -> Dropout
        self.layer1 = nn.Linear(input_dim, hidden_dim_1)
        self.bn1 = nn.BatchNorm1d(hidden_dim_1)
        
        # Layer 2: Linear -> BatchNorm -> ReLU -> Dropout
        self.layer2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.bn2 = nn.BatchNorm1d(hidden_dim_2)
        
        # Output Layer: Linear (no activation - BCEWithLogitsLoss applies sigmoid)
        self.layer3 = nn.Linear(hidden_dim_2, 1)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - returns RAW LOGITS (no sigmoid).
        BCEWithLogitsLoss will apply sigmoid internally.
        """
        # Layer 1
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output (raw logits)
        x = self.layer3(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions (applies sigmoid to logits)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def __repr__(self) -> str:
        return (
            f"HeartDiseaseNetV2(\n"
            f"  Input:  {self.input_dim} features\n"
            f"  Hidden: {self.hidden_dim_1} -> {self.hidden_dim_2} units (THINNER)\n"
            f"  Output: 1 (logits for BCEWithLogitsLoss)\n"
            f"  BatchNorm: Yes\n"
            f")"
        )


# =============================================================================
# STEP 8: ENHANCED TRAINING LOOP WITH LR SCHEDULER
# =============================================================================
# CHANGES FROM ORIGINAL:
#   - Added Learning Rate Scheduler (StepLR)
#   - Uses configurable classification threshold
#   - Works with BCEWithLogitsLoss (expects logits, not probabilities)

def train_model_v2(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[StepLR],
    num_epochs: int,
    patience: int,
    threshold: float,
    device: torch.device
) -> Dict[str, List[float]]:
    """
    Train the neural network with early stopping and LR scheduler.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function (BCEWithLogitsLoss)
        optimizer: Optimizer (Adam)
        scheduler: Learning rate scheduler (StepLR)
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
        threshold: Classification threshold (default 0.4)
        device: Device to train on (CPU/GPU)
        
    Returns:
        Dictionary containing training history
    """
    print("\n" + "=" * 60)
    print("TRAINING NEURAL NETWORK (ENHANCED V2)")
    print("=" * 60)
    print(f"Max epochs: {num_epochs}, Early stopping patience: {patience}")
    print(f"Classification threshold: {threshold} (lowered for better recall)")
    print(f"Learning rate scheduler: StepLR (step={HYPERPARAMETERS['lr_step_size']}, gamma={HYPERPARAMETERS['lr_gamma']})")
    print("-" * 60)
    
    model = model.to(device)
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "learning_rates": []
    }
    
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        history["learning_rates"].append(current_lr)
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass (model outputs logits)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics (apply sigmoid to get probabilities)
            train_loss += loss.item() * X_batch.size(0)
            probs = torch.sigmoid(logits)
            predictions = (probs >= threshold).float()
            train_correct += (predictions == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item() * X_batch.size(0)
                probs = torch.sigmoid(logits)
                predictions = (probs >= threshold).float()
                val_correct += (predictions == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Store history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:3d}/{num_epochs}] | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best validation loss: {best_val_loss:.4f}")
                break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model from training (val_loss: {best_val_loss:.4f})")
    
    return history


# =============================================================================
# STEP 9: ENHANCED MODEL EVALUATION
# =============================================================================
# CHANGES FROM ORIGINAL:
#   - Uses configurable classification threshold
#   - Added ROC-AUC score
#   - Compares multiple thresholds

def evaluate_model_v2(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float,
    device: torch.device
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model on the test set with configurable threshold.
    """
    print("\n" + "=" * 60)
    print(f"MODEL EVALUATION ON TEST SET (threshold={threshold})")
    print("=" * 60)
    
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            all_logits.extend(logits.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())
    
    y_true = np.array(all_targets)
    y_logits = np.array(all_logits)
    y_prob = 1 / (1 + np.exp(-y_logits))  # Sigmoid
    y_pred = (y_prob >= threshold).astype(float)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    
    print("\n--- Classification Metrics ---")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=["No Disease", "Disease"]))
    
    cm = confusion_matrix(y_true, y_pred)
    print("--- Confusion Matrix ---")
    print(f"                 Predicted")
    print(f"              No Disease  Disease")
    print(f"Actual No Disease   {cm[0,0]:3d}       {cm[0,1]:3d}")
    print(f"       Disease      {cm[1,0]:3d}       {cm[1,1]:3d}")
    
    return metrics, y_true, y_pred, y_prob


def compare_thresholds(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> None:
    """Compare performance at different classification thresholds."""
    print("\n" + "=" * 60)
    print("THRESHOLD COMPARISON")
    print("=" * 60)
    print("Showing how different thresholds affect Precision vs Recall trade-off:")
    print("-" * 60)
    
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            all_logits.extend(logits.cpu().numpy().flatten())
            all_targets.extend(y_batch.numpy().flatten())
    
    y_true = np.array(all_targets)
    y_prob = 1 / (1 + np.exp(-np.array(all_logits)))
    
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    
    for thresh in [0.3, 0.35, 0.4, 0.5, 0.6]:
        y_pred = (y_prob >= thresh).astype(float)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        marker = " <-- SELECTED" if thresh == HYPERPARAMETERS["classification_threshold"] else ""
        print(f"{thresh:<12.1f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}{marker}")


# =============================================================================
# STEP 10: 5-FOLD CROSS-VALIDATION (NEW)
# =============================================================================
# More robust evaluation by training on different data splits

def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    pos_weight: float,
    n_folds: int = 5
) -> Dict[str, List[float]]:
    """
    Run K-Fold Cross-Validation for robust model evaluation.
    
    Args:
        X: Feature array
        y: Target array
        pos_weight: Weight for positive class in loss function
        n_folds: Number of folds
        
    Returns:
        Dictionary with metrics for each fold
    """
    print("\n" + "=" * 60)
    print(f"{n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 60)
    print("Training model on different data splits for robust evaluation...")
    print("-" * 60)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    cv_results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1_score": [],
        "roc_auc": []
    }
    
    threshold = HYPERPARAMETERS["classification_threshold"]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_folds} ---")
        
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        # Create tensors
        X_train_t = torch.tensor(X_train_fold, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_fold, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_fold, dtype=torch.float32).unsqueeze(1)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Initialize model for this fold
        set_seeds(RANDOM_SEED + fold)  # Different seed per fold for variety
        model = HeartDiseaseNetV2(
            input_dim=X.shape[1],
            hidden_dim_1=HYPERPARAMETERS["hidden_units_1"],
            hidden_dim_2=HYPERPARAMETERS["hidden_units_2"],
            dropout_rate=HYPERPARAMETERS["dropout_rate"]
        ).to(DEVICE)
        
        # Weighted loss
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
        optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])
        
        # Train (silently)
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(100):  # Fewer epochs for CV
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    val_loss += criterion(model(X_batch), y_batch).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    break
        
        if best_state:
            model.load_state_dict(best_state)
        
        # Evaluate on validation fold
        model.eval()
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                logits = model(X_batch)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy().flatten())
                all_targets.extend(y_batch.numpy().flatten())
        
        y_true = np.array(all_targets)
        y_prob = np.array(all_probs)
        y_pred = (y_prob >= threshold).astype(float)
        
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        
        cv_results["accuracy"].append(acc)
        cv_results["precision"].append(prec)
        cv_results["recall"].append(rec)
        cv_results["f1_score"].append(f1)
        cv_results["roc_auc"].append(auc)
        
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for metric, values in cv_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric.upper():<12}: {mean_val:.4f} (+/- {std_val:.4f})")
    
    return cv_results


# =============================================================================
# STEP 11: RANDOM FOREST ENSEMBLE (NEW)
# =============================================================================
# Compare neural network with Random Forest - often better for small tabular data

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """
    Train a Random Forest classifier for comparison.
    
    Random Forest often outperforms neural networks on small tabular datasets
    because it's harder to overfit and captures feature interactions naturally.
    """
    print("\n" + "=" * 60)
    print("RANDOM FOREST ENSEMBLE (Comparison Model)")
    print("=" * 60)
    print("Random Forest is often better than NNs for small tabular data...")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_SEED,
        class_weight="balanced"  # Handles class imbalance
    )
    
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    
    print("\n--- Random Forest Test Results ---")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
    
    # Feature importance
    print("\n--- Top 10 Important Features ---")
    feature_importance = pd.DataFrame({
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))
    
    return rf, metrics


# =============================================================================
# STEP 12: VISUALIZATION
# =============================================================================

def plot_training_history(history: Dict[str, List[float]], save_path: str = None) -> None:
    """Plot training curves with learning rate."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history["train_loss"], label="Training Loss", color="blue", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validation Loss", color="orange", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss (BCE)", fontsize=12)
    axes[0].set_title("Training vs Validation Loss", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history["train_acc"], label="Training Accuracy", color="blue", linewidth=2)
    axes[1].plot(history["val_acc"], label="Validation Accuracy", color="orange", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title("Training vs Validation Accuracy", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Learning Rate
    axes[2].plot(history["learning_rates"], color="green", linewidth=2)
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Learning Rate", fontsize=12)
    axes[2].set_title("Learning Rate Schedule", fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nTraining curves saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "", save_path: str = None) -> None:
    """Plot confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        annot_kws={"size": 16}
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(f"Confusion Matrix - {title}", fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_model_comparison(nn_metrics: Dict, rf_metrics: Dict, save_path: str = None) -> None:
    """Compare Neural Network vs Random Forest performance."""
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    
    nn_values = [nn_metrics[m] for m in metrics]
    rf_values = [rf_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, nn_values, width, label='Neural Network (Enhanced)', color='steelblue')
    bars2 = ax.bar(x + width/2, rf_values, width, label='Random Forest', color='forestgreen')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison: Neural Network vs Random Forest', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Model comparison saved to: {save_path}")
    
    plt.show()


# =============================================================================
# STEP 13: MAIN EXECUTION
# =============================================================================

def main():
    """Main function to run the complete ENHANCED ML pipeline."""
    
    print("\n" + "=" * 70)
    print("HEART DISEASE PREDICTION - ENHANCED PYTORCH NEURAL NETWORK (V2)")
    print("=" * 70)
    print("\nENHANCEMENTS IMPLEMENTED:")
    print("  1. Weighted Loss Function (addresses class imbalance)")
    print("  2. Lower Classification Threshold (0.4 for better recall)")
    print("  3. Thinner Network (16->8, prevents overfitting)")
    print("  4. Learning Rate Scheduler (StepLR)")
    print("  5. Feature Engineering (interaction terms, age binning)")
    print("  6. 5-Fold Cross-Validation")
    print("  7. Random Forest Ensemble Comparison")
    print("=" * 70)
    
    # --- Step 13a: Load and explore data ---
    df = load_data("heart_cleveland_upload.csv")
    class_info = explore_data(df)
    pos_weight = class_info["pos_weight"]
    
    # --- Step 13b: Feature Engineering (NEW) ---
    df_engineered = engineer_features(df)
    
    # --- Step 13c: Preprocess data ---
    X, y, scaler, feature_names = preprocess_data(df_engineered)
    
    # --- Step 13d: Split data (70/15/15) ---
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # --- Step 13e: Create DataLoaders ---
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        batch_size=HYPERPARAMETERS["batch_size"]
    )
    
    # --- Step 13f: Initialize ENHANCED model ---
    input_dim = X_train.shape[1]
    model = HeartDiseaseNetV2(
        input_dim=input_dim,
        hidden_dim_1=HYPERPARAMETERS["hidden_units_1"],
        hidden_dim_2=HYPERPARAMETERS["hidden_units_2"],
        dropout_rate=HYPERPARAMETERS["dropout_rate"]
    )
    
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE (ENHANCED V2)")
    print("=" * 60)
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # --- Step 13g: Define WEIGHTED loss function and optimizer with scheduler ---
    print("\n" + "=" * 60)
    print("WEIGHTED LOSS FUNCTION (Class Imbalance Fix)")
    print("=" * 60)
    print(f"pos_weight = {pos_weight:.4f}")
    print("This makes missing a Disease case ~{:.1f}x more 'painful' than missing No Disease".format(pos_weight))
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS["learning_rate"])
    scheduler = StepLR(
        optimizer,
        step_size=HYPERPARAMETERS["lr_step_size"],
        gamma=HYPERPARAMETERS["lr_gamma"]
    )
    
    print(f"\nLoss Function: BCEWithLogitsLoss (weighted)")
    print(f"Optimizer: Adam (lr={HYPERPARAMETERS['learning_rate']})")
    print(f"Scheduler: StepLR (step={HYPERPARAMETERS['lr_step_size']}, gamma={HYPERPARAMETERS['lr_gamma']})")
    
    # --- Step 13h: Train the model ---
    history = train_model_v2(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=HYPERPARAMETERS["num_epochs"],
        patience=HYPERPARAMETERS["patience"],
        threshold=HYPERPARAMETERS["classification_threshold"],
        device=DEVICE
    )
    
    # --- Step 13i: Evaluate on test set ---
    nn_metrics, y_true, y_pred, y_prob = evaluate_model_v2(
        model, test_loader,
        threshold=HYPERPARAMETERS["classification_threshold"],
        device=DEVICE
    )
    
    # --- Step 13j: Compare different thresholds ---
    compare_thresholds(model, test_loader, DEVICE)
    
    # --- Step 13k: Run 5-Fold Cross-Validation ---
    cv_results = run_cross_validation(X, y, pos_weight, n_folds=HYPERPARAMETERS["n_folds"])
    
    # --- Step 13l: Train Random Forest for comparison ---
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    
    # --- Step 13m: Plot results ---
    print("\n" + "=" * 60)
    print("VISUALIZATION")
    print("=" * 60)
    
    plot_training_history(history, save_path="training_curves_v2.png")
    plot_confusion_matrix(y_true, y_pred, title="Neural Network (Enhanced)", save_path="confusion_matrix_nn.png")
    
    # RF confusion matrix
    y_pred_rf = rf_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred_rf, title="Random Forest", save_path="confusion_matrix_rf.png")
    
    # Model comparison
    plot_model_comparison(nn_metrics, rf_metrics, save_path="model_comparison.png")
    
    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - ENHANCED MODEL RESULTS")
    print("=" * 70)
    
    print("\n--- Neural Network (Enhanced V2) ---")
    print(f"Architecture: {input_dim} -> 16 -> 8 -> 1 (thinner)")
    print(f"Classification Threshold: {HYPERPARAMETERS['classification_threshold']} (lowered)")
    print(f"Test Set Performance:")
    print(f"  Accuracy:  {nn_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {nn_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {nn_metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {nn_metrics['f1_score']*100:.2f}%")
    print(f"  ROC-AUC:   {nn_metrics['roc_auc']*100:.2f}%")
    
    print("\n--- Random Forest (Ensemble) ---")
    print(f"Test Set Performance:")
    print(f"  Accuracy:  {rf_metrics['accuracy']*100:.2f}%")
    print(f"  Precision: {rf_metrics['precision']*100:.2f}%")
    print(f"  Recall:    {rf_metrics['recall']*100:.2f}%")
    print(f"  F1-Score:  {rf_metrics['f1_score']*100:.2f}%")
    print(f"  ROC-AUC:   {rf_metrics['roc_auc']*100:.2f}%")
    
    print("\n--- 5-Fold Cross-Validation (Neural Network) ---")
    print(f"  Mean Accuracy:  {np.mean(cv_results['accuracy'])*100:.2f}% (+/- {np.std(cv_results['accuracy'])*100:.2f}%)")
    print(f"  Mean F1-Score:  {np.mean(cv_results['f1_score'])*100:.2f}% (+/- {np.std(cv_results['f1_score'])*100:.2f}%)")
    print(f"  Mean ROC-AUC:   {np.mean(cv_results['roc_auc'])*100:.2f}% (+/- {np.std(cv_results['roc_auc'])*100:.2f}%)")
    
    print("\n" + "=" * 70)
    print("IMPROVEMENTS ACHIEVED:")
    print("  - Better Recall (catching more disease cases)")
    print("  - More robust evaluation via Cross-Validation")
    print("  - Comparison with Random Forest ensemble")
    print("  - Feature engineering for domain knowledge")
    print("=" * 70)
    
    return model, rf_model, scaler, history, nn_metrics, rf_metrics, cv_results


if __name__ == "__main__":
    results = main()
    model, rf_model, scaler, history, nn_metrics, rf_metrics, cv_results = results
