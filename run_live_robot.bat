@echo off
SET MODEL=outputs/models/cnn_lstm_2class.pt
SET MODEL_TYPE=cnn_lstm
SET PORT=COM3

echo Starting robot inference with simulated live EEG...

python -m model_training.deployment.pi_inference ^
  --model %MODEL% ^
  --model-type %MODEL_TYPE% ^
  --mode live ^
  --simulate-live ^
  --serial --port %PORT%

pause