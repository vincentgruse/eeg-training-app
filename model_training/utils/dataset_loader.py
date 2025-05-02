"""
dataset_loader.py

This module defines a PyTorch-compatible Dataset class (`EEGDataset`) to load
EEG data stored in NumPy arrays for both 2-class and 5-class classification tasks.

Dependencies:
    numpy, torch
"""

import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    """
    Custom PyTorch Dataset to load EEG features and labels.
    """
    def __init__(self, data_path_X, data_path_y):
        self.X = np.load(data_path_X).astype(np.float32)
        self.y = np.load(data_path_y).astype(np.int64)

        assert len(self.X) == len(self.y), "Mismatch between data and labels"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def get_dataloader(data_dir, batch_size=64, shuffle=True):
    """
    Returns a PyTorch DataLoader for the EEGDataset given a directory.

    Args:
        data_dir (str): Directory containing X_raw.npy and y_raw.npy
        batch_size (int): Size of each training batch
        shuffle (bool): Whether to shuffle data each epoch

    Returns:
        DataLoader: PyTorch DataLoader ready for training or validation
    """
    X_path = f"{data_dir}/X_raw.npy"
    y_path = f"{data_dir}/y_raw.npy"
    dataset = EEGDataset(X_path, y_path)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    """
    Returns a PyTorch DataLoader for the EEGDataset given a directory.
    """
    loader = get_dataloader("model_training/preprocessed_data/5class")
    for X_batch, y_batch in loader:
        print("Batch X:", X_batch.shape)
        print("Batch y:", y_batch.shape)
        break