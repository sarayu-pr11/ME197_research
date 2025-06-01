import os
import h5py
import torch
import pandas as pd
from torch.utils.data import Dataset

class EMGDataset(Dataset):
    def __init__(self, metadata_csv, data_dir, split="train"):
        """
        Loads EMG and pose data from HDF5 files based on metadata.

        Args:
            metadata_csv (str): Path to metadata CSV file.
            data_dir (str): Folder containing .hdf5 data files.
            split (str): Dataset split to use: 'train', 'val', or 'test'.
        """
        self.data_dir = data_dir
        self.meta = pd.read_csv(metadata_csv)

        print("[DEBUG] Loaded CSV with shape:", self.meta.shape)
        print("[DEBUG] CSV Columns:", self.meta.columns.tolist())
        print("[DEBUG] Unique 'split' values:", self.meta['split'].unique())

        # Ensure filename has correct extension
        self.meta["filename"] = self.meta["filename"].astype(str)
        self.meta["filename"] = self.meta["filename"].apply(lambda x: x if x.endswith(".hdf5") else x + ".hdf5")

        # Filter metadata by desired split
        self.meta = self.meta[self.meta["split"] == split].copy()
        print(f"[DEBUG] Metadata filtered to split='{split}', rows remaining: {len(self.meta)}")

        # Get available .hdf5 files in directory
        available_files = set(f for f in os.listdir(data_dir) if f.endswith(".hdf5"))
        print("[DEBUG] HDF5 files found in directory:", len(available_files))
        print("[DEBUG] Example available files:", list(available_files)[:5])

        # Check how many metadata filenames match available HDF5 files
        matching_rows = self.meta["filename"].isin(available_files)
        print("[DEBUG] Matching metadata filenames:", matching_rows.sum())
        print("[DEBUG] Example metadata filenames:", self.meta["filename"].unique()[:5])

        # Filter metadata for only files that actually exist
        self.meta = self.meta[matching_rows].copy()
        self.meta.reset_index(drop=True, inplace=True)

        print(f"[INFO] Final dataset contains {len(self.meta)} samples for split='{split}'")

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = os.path.join(self.data_dir, row["filename"])

        with h5py.File(file_path, "r") as f:
            emg = f["emg"][:].T  # (channels, time)
            pose = f["pose"][:].T  # (joints, time)

        return torch.tensor(emg, dtype=torch.float32), torch.tensor(pose[:, 0], dtype=torch.float32)
