"""
===============================================================================
Utility Functions for EEG Motor Imagery Classification Pipeline
===============================================================================

This file provides helper functions used throughout the training and evaluation
scripts in the EEG-BCI project. These utilities handle:

1. Loading all preprocessed data (from per-subject .npy files) into memory-efficient
   NumPy arrays. This supports efficient training and evaluation across subjects.

2. Setting random seeds across NumPy, PyTorch (CPU and GPU), and Python for full
   experiment reproducibility.

Usage:
    from train.utils import load_all_preprocessed_data, set_seed
-------------------------------------------------------------------------------
"""

import os
import numpy as np
import torch
import random

def load_all_preprocessed_data(preprocessed_dir):
    """
    Loads and concatenates preprocessed EEG data from all subjects.

    Assumes each subject folder (e.g. preprocessed/S001) contains:
        - X.npy: EEG epochs (shape: [n_epochs, 64, 641])
        - y.npy: Class labels (0 for left-hand, 1 for right-hand)

    Returns:
        X_total (np.ndarray): Concatenated EEG data across subjects
        y_total (np.ndarray): Concatenated labels
    """
    all_X, all_y = [], []
    
    # Loop over each subject directory
    for subject in sorted(os.listdir(preprocessed_dir)):
        subj_path = os.path.join(preprocessed_dir, subject)
        if not os.path.isdir(subj_path):
            continue
        
        # Path to EEG data and label files
        x_path = os.path.join(subj_path, "X.npy")
        y_path = os.path.join(subj_path, "y.npy")
        
        # Load data if both files exist
        if os.path.exists(x_path) and os.path.exists(y_path):
            # mmap_mode allows loading large arrays without consuming RAM up front
            X = np.load(x_path, mmap_mode="r")
            y = np.load(y_path)
            all_X.append(X)
            all_y.append(y)

    # Combine all subjects into a single dataset
    X_total = np.concatenate(all_X, axis=0)
    y_total = np.concatenate(all_y, axis=0)

    # Print class balance for diagnostics
    unique, counts = np.unique(y_total, return_counts=True)
    print("Class balance:", dict(zip(unique, counts)))

    return X_total, y_total

def set_seed(seed=42):
    """
    Set fixed seed across common random libraries to ensure reproducibility.
    
    Args:
        seed (int): The seed value to apply (default: 42)
    """
    torch.manual_seed(seed)             # For PyTorch on CPU
    torch.cuda.manual_seed_all(seed)    # For PyTorch on all GPUs
    np.random.seed(seed)                # For NumPy
    random.seed(seed)                   # For Pyhon's built in RNG