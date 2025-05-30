import pandas as pd
import os

# Paths
metadata_path = "data/emg2pose_dataset_mini/metadata.csv"
data_dir = "data"
output_path = "data/metadata_filtered.csv"

# Load metadata
meta = pd.read_csv(metadata_path)

# Fix: append '.hdf5' to the filename column so it matches actual file names
meta["filename"] = meta["filename"].astype(str) + ".hdf5"

# List actual .hdf5 files in your data folder
available_files = set(os.listdir(data_dir))

# Keep only metadata rows where the .hdf5 file exists
meta_filtered = meta[meta["filename"].isin(available_files)].copy()

# Save to new CSV
meta_filtered.to_csv(output_path, index=False)

print(f"[INFO] Saved filtered metadata with {len(meta_filtered)} rows to {output_path}")
