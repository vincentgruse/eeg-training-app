@echo off
REM Change these paths as needed
SET MODEL=model_training/outputs/models/cnn_lstm_2class.pt
SET MODEL_TYPE=cnn_lstm
SET DATAFILE=model_training/preprocessed_data/2class/X_raw.npy
SET PORT=COM3

echo Running EEG inference from file...
python -m model_training.deployment.pi_inference ^
  --model %MODEL% ^
  --model-type %MODEL_TYPE% ^
  --mode file ^
  --file %DATAFILE% ^
  --serial --port %PORT%

pause