"""
===============================================================================
LSTM Model Training Script for EEG Motor Imagery Classification
===============================================================================

This script trains an LSTM model to classify EEG motor imagery data into
left-hand and right-hand imagery classes. The input data is reshaped for
sequence modeling and trained using a standard supervised classification loop.

Key Features:
- Loads preprocessed EEG data from all subjects
- Reshapes input to [batch, 641, 64] for LSTM input
- Normalizes data globally (z-score)
- Splits data into training and validation sets
- Trains with early stopping based on validation accuracy
- Saves the best model to `models/lstm_model.pt`
- Logs loss and accuracy metrics to `logs/lstm_history.json`
- Saves training curves as PNGs under `charts/lstm/`

Dependencies:
- PyTorch, NumPy, scikit-learn, tqdm, matplotlib

Usage:
    python -m train.train_lstm
-------------------------------------------------------------------------------
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.lstm_model import EEGLSTM
from train.utils import load_all_preprocessed_data, set_seed
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Training Configuration
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
VALID_SPLIT = 0.2
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_training_curves(train_losses, val_accuracies):
    """
    Saves training loss and validation accuracy curves to disk under charts/lstm/.
    """
    os.makedirs("charts/lstm", exist_ok=True)

    # Plot and save training loss curve
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM - Train Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("charts/lstm/train_loss.png")
    plt.close()

    # Plot and save validation accuracy curve
    plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM - Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("charts/lstm/val_accuracy.png")
    plt.close()
    
def train_model():
    """
    Loads data, trains the LSTM model, tracks performance, and saves
    the best model and training history.
    """
    print("Loading data...")
    X, y = load_all_preprocessed_data("preprocessed")

    # Reshape data for LSTM: [batch, time_steps, channels]
    X = np.transpose(X, (0, 2, 1))  # from [batch, 64, 641] to [batch, 641, 64]
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Global normalization
    X = (X - X.mean()) / X.std()

    # Split into training and validation sets
    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * VALID_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Initialize model, optimizer, loss
    model = EEGLSTM().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience_counter = 0
    train_losses = []
    val_accuracies = []

    print("Training started...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        
        # Training loop
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation loop
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                preds = model(xb)
                val_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
                val_labels.extend(yb.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

        # Save metrics
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        # Save best model if improved
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "models/lstm_model.pt")
            print(f"New best model saved (val acc = {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    # Save history to JSON log
    os.makedirs("logs", exist_ok=True)
    with open("logs/lstm_history.json", "w") as f:
        json.dump({"train_loss": train_losses, "val_acc": val_accuracies}, f)

    # Plot training curves
    plot_training_curves(train_losses, val_accuracies)
    
if __name__ == "__main__":
    set_seed(42)
    train_model()