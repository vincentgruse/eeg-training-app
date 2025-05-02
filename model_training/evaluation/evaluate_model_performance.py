"""
evaluate_model_performance.py

This script compares the performance of all trained models (CNN, LSTM, CNN-LSTM)
across two classification tasks (2-class and 5-class). It loads saved metrics and
visualization plots, then displays and saves summary figures.

Dependencies:
    os, json, matplotlib
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Directory, models and task configuration.
METRICS_DIR = "model_training/outputs/metrics/"
PLOT_DIR = "model_training/outputs/plots/"
OUTPUT_DIR = "model_training/outputs/plots/evaluation"
MODELS = ["cnn", "lstm", "cnn_lstm"]
TASKS = ["5class", "2class"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# LOAD METRICS
# ==============================================================================
def load_final_accuracy(model, task):
    """
    Load the final validation accuracy for a given model-task pair from JSON.
    """
    json_file = f"{model}/{model}_{task}_metrics.json"
    path = os.path.join(METRICS_DIR, json_file)
    if not os.path.exists(path):
        print(f"[ERROR] Missing: {json_file}")
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("val_accuracy", None)

# ==============================================================================
# IMAGE HELPER FUNCTIONS
# ==============================================================================
def load_plot_image(model, filename):
    """
    Load a saved .png plot for a specific model and filename.
    """
    filename = f"{model}/{filename}"
    path = os.path.join(PLOT_DIR, filename)
    if not os.path.exists(path):
        print(f"[ERROR] Missing file: {filename}, {path}")
        return None
    return mpimg.imread(path)

def plot_side_by_side(task, metric):
    """
    Plot and save a side-by-side comparison of all models for a given task and metric.
    """
    plt.figure(figsize=(15, 4))
    for idx, model in enumerate(MODELS):
        key = f"{model}_{task}"
        img = load_plot_image(model, f"{key}_{metric}.png")
        if img is not None:
            plt.subplot(1, len(MODELS), idx + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{model.upper()} - {task.upper()}")
    plt.suptitle(f"{metric.replace('_', ' ').title()} Comparison ({task.upper()})", fontsize=16)
    plt.tight_layout()
    save_file = os.path.join(OUTPUT_DIR, f"compare_{metric}_{task}.png")
    plt.savefig(save_file)
    print(f"[INFO] Successfully saved: {save_file}")

# ==============================================================================
# ACCURACY TABLE
# ==============================================================================
def print_accuracy_table():
    """
    Print a formatted summary table of final validation accuracies.
    """
    print("\n[INFO] Final Validation Accuracies:")
    print("-" * 40)
    print(f"{'Model':<20} | {'Accuracy'}")
    print("-" * 40)
    for task in TASKS:
        for model in MODELS:
            key = f"{model}_{task}"
            acc = load_final_accuracy(model, task)
            if acc is not None:
                print(f"{model.upper()} {task:<11} | {acc:.4f}")
            else:
                print(f"{model.upper()} {task:<11} | MISSING")
    print("-" * 40)

# ==============================================================================
# MAIN CLIENT
# ==============================================================================
if __name__ == "__main__":
    print_accuracy_table()

    print("\n[INFO] Accuracy Comparison for 5-Class Models")
    plot_side_by_side("5class", "accuracy")

    print("\n[INFO] Accuracy Comparison for 2-Class Models")
    plot_side_by_side("2class", "accuracy")

    print("\n[INFO] Loss Curve Comparison for 5-Class Models")
    plot_side_by_side("5class", "loss")

    print("\n[INFO] Loss Curve Comparison for 2-Class Models")
    plot_side_by_side("2class", "loss")

    print("\n[INFO] Confusion Matrices (5-Class)")
    plot_side_by_side("5class", "confusion")

    print("\n[INFO] Confusion Matrices (2-Class)")
    plot_side_by_side("2class", "confusion")