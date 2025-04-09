"""
===============================================================================
Full Model Evaluation Script: CNN, LSTM, CNN-LSTM (EEG Motor Imagery)
===============================================================================

This script loads all three trained models (CNN, LSTM, and CNN-LSTM) and evaluates
their performance side-by-side using a consistent validation split from the full dataset.

Metrics calculated for each model:
- Accuracy
- F1 Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1 by class)

Input preprocessing:
- CNN expects shape [N, 1, 64, 641]
- LSTM expects shape [N, 641, 64]
- CNN-LSTM uses same input shape as CNN

Intended usage:
- Run after training to compare model effectiveness
- Use results to inform selection for deployment or further tuning

Output:
- Console printout with metrics per model

Usage:
    python compare_models.py
-------------------------------------------------------------------------------
"""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

# Model imports
from models.cnn_model import EEGCNN
from models.lstm_model import EEGLSTM
from models.cnn_lstm_model import EEGCNNLSTM

# Utility functions for loading data and settings reproducibility
from train.utils import load_all_preprocessed_data, set_seed

# Runtime configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
VALID_SPLIT = 0.2

def prepare_data():
    """
    Load and normalize preprocessed EEG data, then prepare reshaped input
    for all three models (CNN, LSTM, CNN-LSTM). Only the validation set is returned.
    """
    X, y = load_all_preprocessed_data("preprocessed")
    X_norm = (X - X.mean()) / X.std()

    # Prepare CNN input: [Batch, Channel, Channels, Time]
    X_cnn = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(1)
    
    # Prepare LSTM input: [Batch, Time, Channels]
    X_lstm = torch.tensor(np.transpose(X_norm, (0, 2, 1)), dtype=torch.float32)
    
    # Convert labels to tensor
    y = torch.tensor(y, dtype=torch.long)

    # Create dataset and perform a consistent validation split
    dataset = TensorDataset(X_cnn, X_lstm, y)
    val_size = int(len(dataset) * VALID_SPLIT)
    train_size = len(dataset) - val_size
    _, val_ds = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    return val_loader

def evaluate_model(name, model, input_type, val_loader):
    """
    Evaluate a single model using the validation set and print metrics.
    
    Parameters:
    - name: str, the display name for the model
    - model: the PyTorch model to evaluate
    - input_type: "cnn" or "lstm", which input to use from dataset
    - val_loader: DataLoader object with validation data
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for xb_cnn, xb_lstm, yb in val_loader:
            # Select correct input type
            xb = xb_cnn if input_type == "cnn" else xb_lstm
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            # Forward pass
            out = model(xb)
            pred = torch.argmax(F.softmax(out, dim=1), dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(yb.cpu().numpy())

    # Compute and print performance metrics.
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    print(f"\n{name} Evaluation:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Left", "Right"]))

def main():
    """
    Load models, prepare data, and evaluate all models side-by-side.
    """
    set_seed(42)
    val_loader = prepare_data()

    print("Loading best models...\n")

    # Load and evaluate CNN
    cnn = EEGCNN().to(DEVICE)
    cnn.load_state_dict(torch.load("models/cnn_model.pt", map_location=DEVICE))
    evaluate_model("CNN", cnn, input_type="cnn", val_loader=val_loader)

    # Load and evaluate LSTM
    lstm = EEGLSTM().to(DEVICE)
    lstm.load_state_dict(torch.load("models/lstm_model.pt", map_location=DEVICE))
    evaluate_model("LSTM", lstm, input_type="lstm", val_loader=val_loader)

    # Load and evaluate CNN-LSTM
    cnn_lstm = EEGCNNLSTM().to(DEVICE)
    cnn_lstm.load_state_dict(torch.load("models/cnn_lstm_model.pt", map_location=DEVICE))
    evaluate_model("CNN-LSTM", cnn_lstm, input_type="cnn", val_loader=val_loader)

if __name__ == "__main__":
    main()
