import os
import h5py
import torch
from torch.utils.data import Dataset

class EMGDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.dataset_dir = data_dir  # directly use the fake dataset folder
        self.files = [
            f for f in os.listdir(self.dataset_dir)
            if f.endswith(".hdf5")
        ]
        print(f"[INFO] Found {len(self.files)} samples.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.dataset_dir, filename)

        with h5py.File(filepath, 'r') as f:
            emg = f['emg'][:].T
            pose = f['pose'][:].T

        return torch.tensor(emg, dtype=torch.float32), torch.tensor(pose[:, 0], dtype=torch.float32)
