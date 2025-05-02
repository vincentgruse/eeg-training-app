# EEG-Based Brain-Computer Interface for Robot Control

This project is a full-stack Brain-Computer Interface (BCI) pipeline that translates a user's EEG signals into directional commands — like FORWARD or STOP — to control a robot in real-time. We use deep learning models trained on raw EEG windows to classify user intent, and then deploy these models on a Raspberry Pi that sends predictions to a robot over serial in structured JSON format. The pipeline is modular, efficient, and built for both offline evaluation and real-time execution.

# How to Run

## Step 1: Preprocessing EEG Data
Script: `python -m model_training/preprocessing/preprocess_data.py`

This script takes raw EEG data from the CSV files stored under:
`model_training/data/*.csv`

Each of these files contains multichannel EEG samples for a single direction. This script applise basic filtering, segments each recording using a sliding window, outputs numpy arrays for both
2 class: FORWARD vs STOP and 5 class: BACKWARD, FORWARD, LEFT, RIGHT, STOP.

Outputs are placed in:
`model_training/preprocessed_data/*/X_raw.npy`
`model_training/preprocessed_data/*/X_raw.npy`

## Step 2: Training Models

We decided to train three different architectures on both class tasks to compare performance.

1. **CNN**: Spatial Pattern Extraction
2. **LSTM**: Temporal Modeling
3. **CNN-LSTM**: Hybrid Combination of Spatial Pattern Extraction and Temporal Modeling

Scripts:
`python -m model_training/train/train_2class.py`
`python -m model_training/train/train_5class.py`
`python -m model_training/train/train_cnn_lstm_2class.py`
`python -m model_training/train/train_cnn_lstm_5class.py`
`python -m model_training/train/train_lstm_2class.py`
`python -m model_training/train/train_lstm_5class.py`

Each script loads raw EEG windows and trains the model using early stopping to prevent overfitting. Validation/Training loss and accuracy are logged during training and models are saved as .pt files to be exported onto the robot. Metrics are also stored to the outputs directory.

## Step 3: Model Evaluation

Script: `python -m model_training/evaluation/evaluate_model_performance.py`

This script will load all .json metric files and performance plots. It will then print a summary table comparing validation accuracies of each model to compare performance. This gives a complete picture of how each model performs both on 2 class and 5 class tasks.

## Step 4: Real-Time Inference and Deployment

Script: `python -m model_training/deployment/pi_inference.py`

This script loads the best trained model and accepts one of three inputs:
  - `--mode simulate`: generate random EEG data for testing
  - `--mode file`: read from .npy file and infer each window
  - `--mode live`: receive streaming EEG via TCP socket
  - `--simulate-live`: simulates a live EEG device using random data over a socket

The output will be a console log that has a prediction and a confidence score: 
- Console logs (`Predicted: FORWARD (conf: 0.89)`) and a JSON equivalent
- {"class": "FORWARD", "confidence": 0.89}`

Example command:
```
bash
python -m model_training.deployment.pi_inference \
  --model outputs/models/cnn_lstm_2class.pt \
  --model-type cnn_lstm \
  --mode live \
  --simulate-live \
  --serial --port /dev/ttyUSB0
```

## Step 5: Robot Receiver

File: robot_control.ino

This is a lightweight Arduino sketch that should run on the robot side. It listens over serial for JSON and parses each prediction. It will also map the predictions to motor actions.