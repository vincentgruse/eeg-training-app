"""
===============================================================================
Model Comparison and Ensemble Evaluation Script for EEG Classification
===============================================================================

This script evaluates the performance of individual trained models (CNN and LSTM)
and computes an ensemble prediction using soft voting (probability averaging).

Main functionality:
- Loads preprocessed EEG validation data (from all subjects)
- Loads trained CNN and LSTM models from disk
- Performs forward passes and computes softmax probabilities
- Combines the two model outputs using soft-voting (mean of probabilities)
- Computes accuracy, F1 score, confusion matrix, and full classification report

Note:
- CNN and LSTM require different input formats; both are prepared from the same data
- The CNN-LSTM model is not used in the ensemble due to lower standalone performance

Result: Prints evaluation metrics for the ensemble to the console

Intended for post-training model comparison and reporting.

Usage:
    python compare_models.py
-------------------------------------------------------------------------------
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader, random_split

# Import model classes and utility functions
from models.cnn_model import EEGCNN
from models.lstm_model import EEGLSTM
from train.utils import load_all_preprocessed_data, set_seed

# Runtime Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
VALID_SPLIT = 0.2

def prepare_data():
    """
    Loads and normalizes EEG data, reshapes it for both CNN and LSTM input,
    and returns a DataLoader containing only the validation split.
    """
    X, y = load_all_preprocessed_data("preprocessed")
    
    # Global normalization
    X_norm = (X - X.mean()) / X.std()

    # Prepare input for both models
    X_cnn = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(1)      # [B, 1, 64, 641]
    X_lstm = torch.tensor(np.transpose(X_norm, (0, 2, 1)), dtype=torch.float32)  # [B, 641, 64]
    y = torch.tensor(y, dtype=torch.long)

    # Combine into a single dataset and split
    dataset = TensorDataset(X_cnn, X_lstm, y)
    val_size = int(len(dataset) * VALID_SPLIT)
    train_size = len(dataset) - val_size
    _, val_ds = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    return val_loader

def run_ensemble():
    """
    Loads the trained CNN and LSTM models, performs evaluation on the validation set,
    and returns performance metrics from their ensemble output.
    """
    set_seed(42)
    val_loader = prepare_data()

    # Load trained CNN
    cnn = EEGCNN().to(DEVICE)
    cnn.load_state_dict(torch.load("models/cnn_model.pt", map_location=DEVICE))
    cnn.eval()

    # Load trained LSTM
    lstm = EEGLSTM().to(DEVICE)
    lstm.load_state_dict(torch.load("models/lstm_model.pt", map_location=DEVICE))
    lstm.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb_cnn, xb_lstm, yb in val_loader:
            xb_cnn, xb_lstm = xb_cnn.to(DEVICE), xb_lstm.to(DEVICE)
            yb = yb.to(DEVICE)

            # Predict class probabilities for each model
            out_cnn = F.softmax(cnn(xb_cnn), dim=1)
            out_lstm = F.softmax(lstm(xb_lstm), dim=1)

            # Soft-voting ensemble: average the model predictions
            avg_probs = (out_cnn + out_lstm) / 2
            preds = torch.argmax(avg_probs, dim=1)

            # Store predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    # Compute evaluation metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Left", "Right"])

    return acc, f1, cm, report

if __name__ == "__main__":
    acc, f1, cm, report = run_ensemble()
    print(f"\nEnsemble Accuracy: {acc:.4f}")
    print(f"Ensemble F1 Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)
