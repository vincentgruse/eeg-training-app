"""
===============================================================================
EEG Preprocessing Script for PhysioNet EEG Motor Imagery Dataset (EEGMMIDB)
===============================================================================

This script preprocesses raw EEG data for all available subjects in the dataset.
It performs the following operations:
- Loads .edf EEG recordings for selected motor imagery runs (R07, R08, R11, R12)
- Applies bandpass filtering (8–30 Hz) and resampling (to 160 Hz)
- Extracts left- and right-hand motor imagery epochs using MNE annotations
- Segments each trial into 4-second epochs
- Saves preprocessed data per subject as NumPy arrays: X.npy (EEG data), y.npy (labels)

Data shape after preprocessing: 
  X: [n_epochs, 64, 641] — EEG channels × time samples per epoch
  y: [n_epochs]          — 0 (left hand) or 1 (right hand)

Output is saved in the following structure:
  preprocessed/SXXX/X.npy
  preprocessed/SXXX/y.npy

Designed for: EEG-BCI motor imagery classification pipelines
Compatible with: PhysioNet EEG Motor Movement/Imagery Dataset 
(https://physionet.org/content/eegmmidb/1.0.0/)
-------------------------------------------------------------------------------
"""

import os
import numpy as np
import mne
from tqdm import tqdm

# --- Configuration Settings ---
RAW_DATA_DIR = "data"                       # Root directory ontaining all SXXX subjects
OUTPUT_DIR = "preprocessed"                 # Destination folder for processed EEG data
RUNS_TO_USE = ["R07", "R08", "R11", "R12"]  # Motor imagery runs.
SAMPLING_RATE = 160                         # Downsample EEG to 160 Hz
EPOCH_LENGTH = 4.0                          # Epoch duration in seconds
BANDPASS_LOWER = 8                          # Lower cutoff for band pass filter
BANDPASS_UPPER = 30                         # Upper cutoff for band pass filter

# Mapping task annotations to class labels.
LABEL_MAP = {
    "T1": 0,    # Left-Hand Imagery
    "T2": 1     # Right-Hand Imagery
}

def preprocess_subject(subject_dir, output_dir):
    """
    Preprocess EEG data for a single subject and save NumPy arrays.
    """
    all_epochs = []
    all_labels = []

    for run_code in RUNS_TO_USE:
        # Compose full file path: e.g., data/S001/S001R07.edf
        edf_path = os.path.join(subject_dir, f"{os.path.basename(subject_dir)}{run_code}.edf")
        if not os.path.exists(edf_path):
            continue

        # Load raw EEG data (64 channels).
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick("eeg")  # Updated from deprecated pick_types
        raw.filter(BANDPASS_LOWER, BANDPASS_UPPER, fir_design='firwin', verbose=False)
        raw.resample(SAMPLING_RATE, verbose=False)

        # Extract annotation events (T1 = left, T2 = right)
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Keep only T1/T2 events for motor imagery
        valid_event_ids = {k: v for k, v in event_id.items() if k in LABEL_MAP}
        if not valid_event_ids:
            continue

        try:
            # Create epochs from raw signal based on event timings
            epochs = mne.Epochs(
                raw,
                events,
                event_id=valid_event_ids,
                tmin=0.0,
                tmax=EPOCH_LENGTH,
                baseline=None,
                preload=True,
                verbose=False
            )
        except Exception as e:
            print(f"Skipping {edf_path} due to Epochs error: {e}")
            continue

        data = epochs.get_data()        # shape: (n_epochs, 64, 641)
        labels = epochs.events[:, 2]    # Extract event code IDs (e.g., 2, 3)

        # Remap numerical event codes to class labels (0 = Left, 1 = Right)
        inv_map = {v: LABEL_MAP[k] for k, v in valid_event_ids.items()}
        mapped_labels = np.vectorize(inv_map.get)(labels)

        all_epochs.append(data)
        all_labels.append(mapped_labels)

    if all_epochs:
        # Concatenate all epochs from different runs
        X = np.concatenate(all_epochs, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Save processed data per subject
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "X.npy"), X)
        np.save(os.path.join(output_dir, "y.npy"), y)
        return len(X)
    
    return 0

def main():
    """
    Preprocess all subjects in RAW_DATA_DIR and save to OUTPUT_DIR.
    """
    subjects = sorted([d for d in os.listdir(RAW_DATA_DIR) if d.startswith("S") and os.path.isdir(os.path.join(RAW_DATA_DIR, d))])
    print(f"Fount {len(subjects)} subjects. Processing...")
    
    total_epochs = 0
    for subj in tqdm(subjects):
        subject_dir = os.path.join(RAW_DATA_DIR, subj)
        output_dir = os.path.join(OUTPUT_DIR, subj)
        n_epochs = preprocess_subject(subject_dir, output_dir)
        total_epochs += n_epochs
        
    print(f"\nFinished preprocessing. Total usable epochs: {total_epochs}")
    
if __name__ == "__main__":
    main()