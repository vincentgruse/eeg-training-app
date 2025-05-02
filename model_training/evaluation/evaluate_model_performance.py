import os
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ------------------ CONFIG ------------------ #
METRICS_DIR = "model_training/outputs/metrics/"
PLOT_DIR = "model_training/outputs/plots/"
OUTPUT_DIR = "model_training/outputs/plots/evaluation"
MODELS = ["cnn", "lstm", "cnn_lstm"]
TASKS = ["5class", "2class"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------ LOAD METRICS ------------------ #
def load_final_accuracy(model, task):
    json_file = f"{model}/{model}_{task}_metrics.json"
    path = os.path.join(METRICS_DIR, json_file)
    if not os.path.exists(path):
        print(f"⚠️ Missing: {json_file}")
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("val_accuracy", None)

# ------------------ IMAGE HELPERS ------------------ #
def load_plot_image(model, filename):
    filename = f"{model}/{filename}"
    path = os.path.join(PLOT_DIR, filename)
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {filename}, {path}")
        return None
    return mpimg.imread(path)

def plot_side_by_side(task, metric):
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
    print(f" Saved: {save_file}")

# ------------------ ACCURACY TABLE ------------------ #
def print_accuracy_table():
    print("\n🧠 Final Validation Accuracies:")
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

# ------------------ MAIN ------------------ #
if __name__ == "__main__":
    print_accuracy_table()

    print("\n📊 Accuracy Comparison for 5-Class Models")
    plot_side_by_side("5class", "accuracy")

    print("\n📊 Accuracy Comparison for 2-Class Models")
    plot_side_by_side("2class", "accuracy")

    print("\n📉 Loss Curve Comparison for 5-Class Models")
    plot_side_by_side("5class", "loss")

    print("\n📉 Loss Curve Comparison for 2-Class Models")
    plot_side_by_side("2class", "loss")

    print("\n📷 Confusion Matrices (5-Class)")
    plot_side_by_side("5class", "confusion")

    print("\n📷 Confusion Matrices (2-Class)")
    plot_side_by_side("2class", "confusion")
