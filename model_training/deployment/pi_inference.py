"""
pi_inference.py

This script is the final stage of the EEG-based BCI pipeline. It loads a trained
deep learning model (CNN, LSTM, or CNN-LSTM) and performs real-time inference
on EEG data for directional command prediction (e.g., FORWARD, STOP).

Dependencies:
    torch, numpy, argparse, serial, time, os, json, socket, threading
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import serial
import time
import os
import json
import socket
import threading

# Import model definitions.
from model_training.models.cnn import EEGCNN
from model_training.models.lstm import EEGLSTM
from model_training.models.cnn_lstm import EEGCNNLSTM

# Label and shape configuration.
LABELS_2CLASS = ['FORWARD', 'STOP']
LABELS_5CLASS = ['BACKWARD', 'FORWARD', 'LEFT', 'RIGHT', 'STOP']
SIMULATED_INPUT_SHAPE = (1, 1, 8, 250)
LIVE_PORT = 5000

def load_model(model_path, model_type):
    """
    Loads a CNN, LSTM, or CNN-LSTM model from the .pt file for inference.
    """
    print(f"[INFO] Loading Model: {model_path}")
    num_classes = 2 if '2class' in model_path else 5

    if model_type == "cnn":
        model = EEGCNN(num_classes=num_classes)
    elif model_type == "lstm":
        model = EEGLSTM(num_classes=num_classes)
    elif model_type == "cnn_lstm":
        model = EEGCNNLSTM(num_classes=num_classes)
    else:
        raise ValueError("[ERROR] Invalid model type. Use cnn, lstm, or cnn_lstm.")

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    labels = LABELS_2CLASS if num_classes == 2 else LABELS_5CLASS
    return model, labels

# ==============================================================================
# INPUT MODES
# ==============================================================================
def get_simulated_input():
    """
    Generate a fake EEG window for testing.
    Shape: [1, 1, 8, 250]
    """
    return np.random.randn(*SIMULATED_INPUT_SHAPE).astype(np.float32)

def get_file_inputs(file_path):
    """
    Load EEG windows from a preprocessed .npy file.
    Produces one window at a time.
    """
    data = np.load(file_path)
    for window in data:
        if window.shape == (8, 250):
            window = window.reshape(1, 1, 8, 250)
        elif window.shape == (250, 8):
            window = window.T.reshape(1, 1, 8, 250)
        yield window.astype(np.float32)

def get_live_input_stream(host='0.0.0.0', port=LIVE_PORT):
    """
    Wait for a TCP connection and receive live EEG windows from an external source.
    Each EEG window should be sent as newline terminated JSON.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"[INFO] Waiting for EEG stream on {host}:{port}...")
    conn, addr = server.accept()
    print(f"[INFO] Successfully connected to {addr}")

    while True:
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in chunk:
                break

        try:
            eeg_window = json.loads(data.decode().strip())
            eeg_array = np.array(eeg_window, dtype=np.float32)
            if eeg_array.shape == (8, 250):
                eeg_array = eeg_array.reshape(1, 1, 8, 250)
            elif eeg_array.shape == (250, 8):
                eeg_array = eeg_array.T.reshape(1, 1, 8, 250)
            else:
                print("[INFO] Invalid EEG window shape:", eeg_array.shape)
                continue
            yield eeg_array
        except Exception as e:
            print("[INFO] Error decoding EEG input:", e)

# ==============================================================================
# SIMULATE LIVE STREAMING
# ==============================================================================
def launch_fake_live_sender(host="127.0.0.1", port=LIVE_PORT, interval=1.0):
    """
    Simulates a live EEG device by sending random windows over a TCP socket.
    This runs in a background thread for testing live mode without hardware.
    """
    def send_loop():
        time.sleep(2)  # give server time to start
        try:
            sock = socket.socket()
            sock.connect((host, port))
            print(f"[INFO] Simulated EEG sender connected to {host}:{port}")
            while True:
                eeg = np.random.randn(8, 250).tolist()
                msg = json.dumps(eeg) + "\n"
                sock.sendall(msg.encode())
                time.sleep(interval)
        except Exception as e:
            print(f"[ERROR] Simulated sender failed: {e}")
    t = threading.Thread(target=send_loop, daemon=True)
    t.start()

# ==============================================================================
# SERIAL SETUP
# ==============================================================================
def init_serial(port, baud):
    """
    Initialize serial connection to the robot.
    """
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"[INFO] Serial connected on {port} @ {baud} baud.")
        return ser
    except Exception as e:
        print(f"[INFO] Serial connection failed: {e}")
        return None

# ==============================================================================
# MAIN INFERENCE LOOP
# ==============================================================================
def inference_loop(model, labels, model_type, mode, file_path=None, use_serial=False, port="COM3", baud=9600):
    """
    Continuously receives EEG input, performs inference, and sends results via serial (if enabled).
    """
    ser = init_serial(port, baud) if use_serial else None

    # Chooses the appropriate input generator.
    if mode == "simulate":
        generator = lambda: iter([get_simulated_input()] * 100000)
    elif mode == "file":
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Invalid file: {file_path}")
        generator = lambda: get_file_inputs(file_path)
    elif mode == "live":
        generator = lambda: get_live_input_stream()
    else:
        raise ValueError("[ERROR] Invalid mode. Choose simulate, file, or live.")

    print(f"[INFO] Inference started using {model_type.upper()} model in '{mode}' mode.")

    try:
        for eeg in generator():
            input_tensor = torch.tensor(eeg, dtype=torch.float32)

            # Used because the LSTM expects [batch, sequence length, features]
            if model_type == "lstm":
                input_tensor = input_tensor.squeeze(1).squeeze(1)

            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).numpy()[0]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])

            prediction = labels[pred_idx]
            result = {
                "class": prediction,
                "confidence": round(confidence, 4)
            }

            print(f"[INFO] {json.dumps(result)}")

            if ser:
                ser.write((json.dumps(result) + "\n").encode())

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("[INFO] Inference stopped by user.")
        if ser:
            ser.close()

# ==============================================================================
# MAIN CLIENT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--model-type", type=str, choices=["cnn", "lstm", "cnn_lstm"], required=True)
    parser.add_argument("--mode", type=str, choices=["simulate", "file", "live"], required=True)
    parser.add_argument("--file", type=str, help="Path to .npy file for --mode file")
    parser.add_argument("--serial", action="store_true", help="Enable serial output")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (e.g., /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=9600, help="Baud rate")
    parser.add_argument("--simulate-live", action="store_true", help="Use internal fake live EEG generator")

    args = parser.parse_args()

    # Starts simulated live stream if flagged.
    if args.mode == "live" and args.simulate_live:
        launch_fake_live_sender(host="127.0.0.1")

    model, labels = load_model(args.model, args.model_type)
    inference_loop(
        model=model,
        labels=labels,
        model_type=args.model_type,
        mode=args.mode,
        file_path=args.file,
        use_serial=args.serial,
        port=args.port,
        baud=args.baud
    )