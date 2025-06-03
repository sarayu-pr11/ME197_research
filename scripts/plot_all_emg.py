import matplotlib
matplotlib.use("Agg")  # Headless backend

import matplotlib.pyplot as plt
import h5py
import numpy as np
import os

def plot_emg_channels(h5_file_path, output_dir="emg_plots"):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_file_path, "r") as f:
        # Navigate to EMG dataset
        group = f.get("emg2pose/timeseries") or f.get("data") or next(iter(f.values()))
        data = group[:]

        # Extract EMG
        emg_signals = np.stack([x["emg"] for x in data], axis=0)  # shape: (T, 16)

        plt.figure(figsize=(15, 10))
        offset = 5
        for i in range(emg_signals.shape[1]):
            signal = emg_signals[:, i]
            plt.plot(signal + i * offset, label=f"Channel {i+1}", linewidth=0.6)

        plt.title("EMG Channels")
        plt.axis("off")
        filename = os.path.basename(h5_file_path).replace(".hdf5", ".png")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        print(f"[INFO] Saved plot to {filename}")

if __name__ == "__main__":
    data_dir = "data/emg2pose_dataset_mini"
    for fname in os.listdir(data_dir):
        if fname.endswith(".hdf5"):
            plot_emg_channels(os.path.join(data_dir, fname))
