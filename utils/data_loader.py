import os
import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset

class EMGDataset(Dataset):
    def __init__(self, metadata_csv, data_dir, split="train"):
        self.data_dir = data_dir
        self.meta = pd.read_csv(metadata_csv)

        # Keep only the desired split
        self.meta = self.meta[self.meta["split"] == split].copy()

        # Get all available .hdf5 files (should match your actual dataset, e.g., 2022-12-06)
        available_files = set(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))

        # Keep only metadata rows where the filename exists
        self.meta = self.meta[self.meta["filename"].isin(available_files)]
        self.meta.reset_index(drop=True, inplace=True)

        print(f"[INFO] Found {len(self.meta)} valid samples for split='{split}'")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        filepath = os.path.join(self.data_dir, row["filename"])

        with h5py.File(filepath, 'r') as f:
            emg = f["emg"][:].T  # shape: (channels, time)
            pose = f["pose"][:].T  # shape: (joints, time)

        return torch.tensor(emg, dtype=torch.float32), torch.tensor(pose[:, 0], dtype=torch.float32)
