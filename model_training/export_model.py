"""
===============================================================================
Model Export Script for EEG Classification (TorchScript)
===============================================================================

This script exports a trained PyTorch model (CNN, LSTM, or CNN-LSTM) into a
TorchScript `.pt` format for deployment. TorchScript allows you to run
PyTorch models without requiring Python code — suitable for C++ applications,
mobile apps, or embedded systems like Raspberry Pi or Jetson Nano.

This script supports:
- Command-line selection of model type (`--model cnn|lstm|cnn_lstm`)
- Automatic loading of the correct model architecture and weights
- Generation of a TorchScript file in `models/<model_name>_scripted.pt`

Supported models:
- CNN:    Trained on EEG shape [1, 1, 64, 641]
- LSTM:   Trained on reshaped [1, 641, 64]
- CNN-LSTM: Follows same input shape as CNN

Usage:
    python export_model.py --model cnn
    python export_model.py --model lstm
    python export_model.py --model cnn_lstm

Result:
    models/cnn_scripted.pt (or lstm_scripted.pt / cnn_lstm_scripted.pt)
-------------------------------------------------------------------------------
"""

import argparse
import os
import torch
import torch.nn as nn

# Import all supported model architectures.
from models.cnn_model import EEGCNN
from models.lstm_model import EEGLSTM
from models.cnn_lstm_model import EEGCNNLSTM

# Define expected input shapes per model.
INPUT_SHAPES = {
    "cnn": (1, 1, 64, 641),
    "lstm": (1, 641, 64),
    "cnn_lstm": (1, 1, 64, 641)
}

def export_model(model_name: str, model_class: nn.Module, checkpoint_path: str, input_shape: tuple):
    """
    Load model weights and export to TorchScript format.
    """
    print(f"Exporting {model_name.upper()} model...")
    
    # Initialize model and load trained weights onto CPU
    model = model_class().to("cpu")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    # Create dummy input to trace computation graph
    dummy_input = torch.randn(*input_shape)

    # Export to TorchScript using tracing
    scripted = torch.jit.trace(model, dummy_input)
    torchscript_path = f"models/{model_name}_scripted.pt"
    scripted.save(torchscript_path)
    print(f"TorchScript saved: {torchscript_path}")

if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Export a trained EEG model to TorchScript and ONNX")
    parser.add_argument("--model", type=str, choices=["cnn", "lstm", "cnn_lstm"], required=True,
        help="Model to export: cnn | lstm | cnn_lstm")
    args = parser.parse_args()

    # Mapping user input to model class and checkpoint path
    models_map = {
        "cnn": (EEGCNN, "models/cnn_model.pt"),
        "lstm": (EEGLSTM, "models/lstm_model.pt"),
        "cnn_lstm": (EEGCNNLSTM, "models/cnn_lstm_model.pt")
    }

    # Retrieve model class and input shape for selected model
    model_class, checkpoint = models_map[args.model]
    shape = INPUT_SHAPES[args.model]
    
    # Run export
    export_model(args.model, model_class, checkpoint, shape)
