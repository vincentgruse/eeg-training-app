import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, data_path_X, data_path_y):
        self.X = np.load(data_path_X).astype(np.float32)
        self.y = np.load(data_path_y).astype(np.int64)

        assert len(self.X) == len(self.y), "Mismatch between data and labels"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def get_dataloader(data_dir, batch_size=64, shuffle=True):
    X_path = f"{data_dir}/X_raw.npy"
    y_path = f"{data_dir}/y_raw.npy"
    dataset = EEGDataset(X_path, y_path)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Example usage (can be commented out when importing as a module)
if __name__ == "__main__":
    loader = get_dataloader("model_training/preprocessed_data/5class")
    for X_batch, y_batch in loader:
        print("Batch X:", X_batch.shape)
        print("Batch y:", y_batch.shape)
        break