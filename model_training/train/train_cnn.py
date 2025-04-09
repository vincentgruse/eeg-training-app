"""
===============================================================================
CNN Model Training Script for EEG Motor Imagery Classification
===============================================================================

This script trains a 2D convolutional neural network (CNN) to classify EEG
motor imagery data (left-hand vs. right-hand) using spatial information
encoded in EEG channel data.

Key Features:
- Loads normalized EEG data from all subjects
- Reshapes input to 4D tensor [batch, 1, 64, 641] for CNN input
- Trains CNN with early stopping based on validation accuracy
- Saves best model weights to `models/cnn_model.pt`
- Logs training history to `logs/cnn_history.json`
- Saves training loss and validation accuracy plots to `charts/cnn/`

Intended Use:
- For use in comparative modeling pipelines (CNN vs. LSTM vs. CNN-LSTM)
- Suitable for deployment once model performance is finalized

Usage:
    python -m train.train_cnn
-------------------------------------------------------------------------------
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from models.cnn_model import EEGCNN
from train.utils import load_all_preprocessed_data, set_seed
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

# Training configuration
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
VALID_SPLIT = 0.2
PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_training_curves(train_losses, val_accuracies):
    """
    Generate and save training loss and validation accuracy plots
    to the `charts/cnn/` directory.
    """
    os.makedirs("charts/cnn", exist_ok=True)
    
    # Plot training loss
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN - Train Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("charts/cnn/train_loss.png")
    plt.close()
    
    # Plot validation accuracy
    plt.figure()
    plt.plot(val_accuracies, label="Val Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN - Validation Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("charts/cnn/val_accuracy.png")
    plt.close()
    
def train_model():
    """
    Main training loop for the CNN model. Loads and preprocesses data,
    performs validation split, trains using mini-batch SGD (Adam),
    and saves the best model and metrics to disk.
    """
    print("Loading data...")
    X, y = load_all_preprocessed_data("preprocessed")
    
    # Reshape for Conv2D input: [batch, channel=1, height=64, width=641]
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [N, 1, 64, 641]
    y = torch.tensor(y, dtype=torch.long)

    # Normalize globally
    X = (X - X.mean()) / X.std()
    
    # Create training and validation datasets
    dataset = TensorDataset(X, y)
    val_size = int(len(dataset) * VALID_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model, optimizer, loss
    model = EEGCNN().to(DEVICE)
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

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        # Save best model if validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "models/cnn_model.pt")
            print(f"New best model saved (val acc = {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break
            
    # Save metrics history to file
    os.makedirs("logs", exist_ok=True)
    with open("logs/cnn_history.json", "w") as f:
        json.dump({"train_loss": train_losses, "val_acc": val_accuracies}, f)
    
    # Save training visualizations
    plot_training_curves(train_losses, val_accuracies)

if __name__ == "__main__":
    set_seed(42)
    train_model()
