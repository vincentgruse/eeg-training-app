"""
train_lstm_2class.py

This script trains an LSTM model for binary EEG classification: FORWARD vs STOP.

Dependencies:
    json, torch, matplotlib, numpy, os, EEGLSTM, EEGDataset, sklearn
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from model_training.models.lstm import EEGLSTM
from model_training.utils.dataset_loader import EEGDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Training configuration.
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path configuration.
X_PATH = "model_training/preprocessed_data/2class/X_raw.npy"
Y_PATH = "model_training/preprocessed_data/2class/y_raw.npy"
MODEL_PATH = "model_training/outputs/models/lstm_2class.pt"
PLOT_DIR = "model_training/outputs/plots/lstm"
METRICS_DIR = "model_training/outputs/metrics/lstm"

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ==============================================================================
# DATA
# ==============================================================================
dataset = EEGDataset(X_PATH, Y_PATH)

if dataset.X.shape[1] == 8:
    dataset.X = dataset.X.transpose(0, 2, 1)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ==============================================================================
# MODEL
# ==============================================================================
model = EEGLSTM(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
train_losses, val_losses, val_accuracies = [], [], []
best_acc = 0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Model validation.
    model.eval()
    correct = 0
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    val_acc = correct / len(val_ds)
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_acc)
    print(f"[INFO] Epoch {epoch+1}: Train Loss={train_losses[-1]:.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[INFO] New best model saved (val acc = {val_acc:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

# ==============================================================================
# EVALUATION
# ==============================================================================
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FORWARD", "STOP"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - LSTM 2-Class")
plt.savefig(f"{PLOT_DIR}/lstm_2class_confusion.png")
plt.close()

# Accuracy Plot.
plt.plot(val_accuracies, label="Val Acc")
plt.title("Validation Accuracy - LSTM 2-Class")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid()
plt.legend()
plt.savefig(f"{PLOT_DIR}/lstm_2class_accuracy.png")
plt.close()

# Loss Plot.
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve - LSTM 2-Class")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.legend()
plt.savefig(f"{PLOT_DIR}/lstm_2class_loss.png")
plt.close()

# Save metrics as JSON.
metrics = {
    "val_accuracy": best_acc,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "val_accuracies": val_accuracies
}

json_path = os.path.join(METRICS_DIR, f"{MODEL_PATH.split('/')[-1].replace('.pt', '')}_metrics.json")
with open(json_path, "w") as f:
    json.dump(metrics, f)
print(f"[INFO] Metrics saved to {json_path}")