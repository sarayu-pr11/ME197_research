import os
import torch
import pandas as pd
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class EMGDataset(Dataset):
    def __init__(self, metadata_csv, data_dir, split="train"):
        self.data_dir = data_dir
        self.meta = pd.read_csv(metadata_csv)
      #  print("[DEBUG] Loaded CSV with shape:", self.meta.shape)
      #  print("[DEBUG] CSV Columns:", list(self.meta.columns))

       # print("[DEBUG] Unique 'split' values:", self.meta['split'].unique())
        self.meta = self.meta[self.meta["split"] == split].copy()
      #  print(f"[DEBUG] Metadata filtered to split='{split}', rows remaining:", len(self.meta))

        # Find valid HDF5 files in the directory
        available_files = {
            f for f in os.listdir(data_dir)
            if f.endswith(".hdf5")
        }
        print("[DEBUG] HDF5 files found in directory:", len(available_files))
      #  print("[DEBUG] Example available files:", list(available_files)[:5])

        self.meta["filename_with_ext"] = self.meta["filename"].astype(str) + ".hdf5"
        self.meta = self.meta[self.meta["filename_with_ext"].isin(available_files)].copy()
        print("[DEBUG] Matching metadata filenames:", len(self.meta))
      #  print("[DEBUG] Example metadata filenames:", self.meta["filename"].unique()[:5])

        self.file_list = self.meta["filename_with_ext"].tolist()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        fpath = os.path.join(self.data_dir, fname)

        try:
            with h5py.File(fpath, "r") as f:
                # Try accessing 'data' dataset directly
                if "data" in f:
                    data = f["data"][:]
                else:
                    # Find first actual dataset inside the file, even if nested
                    def find_dataset(group):
                        for key in group:
                            item = group[key]
                            if isinstance(item, h5py.Dataset):
                                return item[:]
                            elif isinstance(item, h5py.Group):
                                result = find_dataset(item)
                                if result is not None:
                                    return result
                        return None

                    data = find_dataset(f)
                    if data is None:
                        raise ValueError(f"No dataset found inside {fpath}")

                emg = torch.tensor(np.array([x["emg"] for x in data]), dtype=torch.float32)  # (T, 16)
                pose = torch.tensor(np.array([x["joint_angles"] for x in data]), dtype=torch.float32)  # (T, 20)
                return emg, pose

        except Exception as e:
            print(f"[ERROR] Failed to read file {fpath}: {e}")
            raise e

def pad_collate(batch):
    emg_list, pose_list = zip(*batch)
    emg_padded = pad_sequence(emg_list, batch_first=True)    # (B, T_max, 16)
    pose_padded = pad_sequence(pose_list, batch_first=True)  # (B, T_max, 20)
    return emg_padded.transpose(1, 2), pose_padded.transpose(1, 2)

# Optional: for direct testing of the loader
if __name__ == "__main__":
    dataset = EMGDataset("data/emg2pose_dataset_mini/metadata.csv", "data/emg2pose_dataset_mini", split="val")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=pad_collate)

    for emg, pose in loader:
        print("EMG batch shape:", emg.shape)
        print("Pose batch shape:", pose.shape)
        break
