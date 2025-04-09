"""
===============================================================================
CNN-LSTM Model Training Script for EEG Motor Imagery Classification
===============================================================================

This script trains a hybrid CNN-LSTM model to classify motor imagery EEG signals.
It combines the spatial feature extraction capabilities of a CNN with the temporal
dependency modeling of an LSTM. The model takes EEG data shaped as a 2D spatial 
grid across time, and learns both local spatial patterns and sequential dynamics.

Key Features:
- Loads preprocessed EEG data from all 109 subjects
- Shapes input into [batch, 1, 64, 641] (channel-first format for CNN)
- Normalizes input globally across the dataset
- Splits data into training and validation sets
- Trains with early stopping based on validation accuracy
- Saves best-performing model weights to `models/cnn_lstm_model.pt`
- Saves training history to `logs/cnn_lstm_history.json`
- Plots loss and accuracy over epochs to `charts/cnn_lstm/`

Usage:
    python -m train.train_cnn_lstm
-------------------------------------------------------------------------------
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.cnn_lstm_model import EEGCNNLSTM
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
    Saves training loss and validation accuracy plots under charts/cnn_lstm/.
    """
    os.makedirs("charts/cnn_lstm", exist_ok=True)

    # Plot training loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN-LSTM - Train Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("charts/cnn_lstm/train_loss.png")
    plt.close()

    # Plot validation accuracy
    plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN-LSTM - Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("charts/cnn_lstm/val_accuracy.png")
    plt.close()

def train_model():
    """
    Trains the CNN-LSTM model on EEG motor imagery data with early stopping.
    Tracks and plots training history and saves the best model.
    """
    print("Loading data...")
    X, y = load_all_preprocessed_data("preprocessed")

    # CNN expects 4D input: [N, 1, 64, 641]
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)

    # Normalize the entire dataset globally (z-score)
    X = (X - X.mean()) / X.std()

    # Split dataset into training and validation subsets
    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * VALID_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Initialize model, loss function, optimizer
    model = EEGCNNLSTM().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Metrics tracking
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

        # Track loss and accuracy for plotting
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        # Save best model if validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "models/cnn_lstm_model.pt")
            print(f"New best model saved (val acc = {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    # Save training history as JSON
    os.makedirs("logs", exist_ok=True)
    with open("logs/cnn_lstm_history.json", "w") as f:
        json.dump({"train_loss": train_losses, "val_acc": val_accuracies}, f)

    # Plot training performance
    plot_training_curves(train_losses, val_accuracies)

if __name__ == "__main__":
    set_seed(42)
    train_model()