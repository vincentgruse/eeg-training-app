#!/bin/bash

# Update these variables as needed
MODEL="model_training/outputs/models/cnn_lstm_2class.pt"
MODEL_TYPE="cnn_lstm"
DATAFILE="model_training/preprocessed_data/2class/X_raw.npy"
PORT="/dev/ttyUSB0"

echo "🔁 Running EEG inference from file..."
python -m model_training.deployment.pi_inference \
  --model "$MODEL" \
  --model-type "$MODEL_TYPE" \
  --mode file \
  --file "$DATAFILE" \
  --serial --port "$PORT"
