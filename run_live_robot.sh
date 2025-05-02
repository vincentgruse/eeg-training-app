#!/bin/bash

MODEL="outputs/models/cnn_lstm_2class.pt"
MODEL_TYPE="cnn_lstm"
PORT="/dev/ttyUSB0"

echo "🚀 Starting robot inference (live mode, simulated EEG)..."

python -m model_training.deployment.pi_inference \
  --model "$MODEL" \
  --model-type "$MODEL_TYPE" \
  --mode live \
  --simulate-live \
  --serial --port "$PORT"
