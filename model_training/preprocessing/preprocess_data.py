"""
preprocess_data.py

This script is intended to process raw EEG data from collected CSVs located in
`model_training/data/`. It will perform:
1. Bandpass Filtering (8-30 Hz)
2. Sliding Window Segmentation (3s, 50% overlap)
3. Saves two outputs for both 5-class and 2-class models:
    - Raw windows: [samples, 8, 750]
    - Feature vectors: [samples, 8, num_features]
    
Dependences: numpy, pandas, pathlib, sklearn, tqdm
"""

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Path configurations.
RAW_DATA_DIR = Path("model_training/data")
OUT_DIR_5 = Path("model_training/preprocessed_data/5class")
OUT_DIR_2 = Path("model_training/preprocessed_data/2class")

# EEG configurations.
EEG_CHANNELS = [f"EXG Channel {i}" for i in range(8)]
WINDOW_SIZE = 250     # 1 second
STEP_SIZE = 25         # 90% overlap (maximize data)
SAMPLING_RATE = 250

# Label encodings.
LABEL_MAP_5 = {
    'BACKWARD': 0,
    'FORWARD': 1,
    'LEFT': 2,
    'RIGHT': 3,
    'STOP': 4
}

LABEL_MAP_2 = {
    'FORWARD': 0,
    'STOP': 1
}

# ==============================================================================
# SLIDING WINDOW
# ==============================================================================
def sliding_window(data: np.ndarray, window_size: int, step_size: int):
    """
    Apply a sliding window across a continuous EEG recording.

    Returns:
        Array of shape [num_windows, window_size, num_channels]
    """
    windows = []
    for start in range(0, data.shape[0] - window_size + 1, step_size):
        window = data[start:start + window_size]
        windows.append(window)
    return np.stack(windows)

# ==============================================================================
# NORMALIZATION
# ==============================================================================
def normalize_windows(windows: np.ndarray):
    """
    Z-score normalize each EEG window independently.
    """
    scaler = StandardScaler()
    normed = np.array([scaler.fit_transform(win) for win in windows])
    return normed

# ==============================================================================
# PREPROCESSING BLOCK
# ==============================================================================
def process_all_csv_files():
    """
    Processes all labeled EEG files in /data, applies segmentation and normalization,
    and outputs preprocessed numpy arrays for model training.
    """
    X_raw_5, y_raw_5 = [], []
    X_raw_2, y_raw_2 = [], []

    for label_dir in RAW_DATA_DIR.iterdir():
        if not label_dir.is_dir():
            continue
        label_name = label_dir.name.upper()

        for csv_file in tqdm(label_dir.glob("*.csv"), desc=f"{label_name}"):
            try:
                df = pd.read_csv(csv_file)
                eeg = df[EEG_CHANNELS].values
                
                # Normalize and segment EEG signal.
                windows = sliding_window(eeg, WINDOW_SIZE, STEP_SIZE)
                windows = normalize_windows(windows)

                # 5-class
                if label_name in LABEL_MAP_5:
                    label_5 = LABEL_MAP_5[label_name]
                    X_raw_5.append(windows)
                    y_raw_5.extend([label_5] * len(windows))

                # 2-class (only FORWARD & STOP)
                if label_name in LABEL_MAP_2:
                    label_2 = LABEL_MAP_2[label_name]
                    X_raw_2.append(windows)
                    y_raw_2.extend([label_2] * len(windows))

            except Exception as e:
                print(f"[ERROR] Failed to process {csv_file}: {e}")

    # Save outputs
    OUT_DIR_5.mkdir(parents=True, exist_ok=True)
    OUT_DIR_2.mkdir(parents=True, exist_ok=True)

    np.save(OUT_DIR_5 / "X_raw.npy", np.concatenate(X_raw_5))
    np.save(OUT_DIR_5 / "y_raw.npy", np.array(y_raw_5))
    np.save(OUT_DIR_2 / "X_raw.npy", np.concatenate(X_raw_2))
    np.save(OUT_DIR_2 / "y_raw.npy", np.array(y_raw_2))

    print("[INFO] Preprocessing complete.")

if __name__ == "__main__":
    print("[INFO] Preprocessing Data")
    process_all_csv_files()