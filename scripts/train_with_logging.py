import os
import torch
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn


class EMGDataset(Dataset):
    def __init__(self, metadata_csv, data_dir, split="train"):
        self.data_dir = data_dir
        self.meta = pd.read_csv(metadata_csv)
        self.meta = self.meta[self.meta["split"] == split].copy()
        self.meta["filename_with_ext"] = self.meta["filename"].astype(str) + ".hdf5"
        files_on_disk = set(os.listdir(data_dir))
        self.meta = self.meta[self.meta["filename_with_ext"].isin(files_on_disk)].copy()
        self.file_list = self.meta["filename_with_ext"].tolist()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        fpath = os.path.join(self.data_dir, fname)
        try:
            with h5py.File(fpath, "r") as f:
                dataset = list(f.values())[0]
                emg = torch.tensor([x["emg"] for x in dataset], dtype=torch.float32)
                pose = torch.tensor([x["joint_angles"] for x in dataset], dtype=torch.float32)
                return emg, pose
        except Exception as e:
            print(f"[ERROR] Failed to load {fpath}: {e}")
            raise e


def pad_collate(batch):
    emg_list, pose_list = zip(*batch)
    emg_padded = pad_sequence(emg_list, batch_first=True)
    pose_padded = pad_sequence(pose_list, batch_first=True)
    return emg_padded.transpose(1, 2), pose_padded.transpose(1, 2)


class SimpleConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 20, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.encoder(x)


def train_with_logging():
    dataset = EMGDataset("data/emg2pose_dataset_mini/metadata.csv", "data/emg2pose_dataset_mini", split="val")
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=pad_collate)

    model = SimpleConvModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_history = []
    for epoch in range(5):
        total_loss = 0
        for emg, pose in loader:
            optimizer.zero_grad()
            pred = model(emg)
            loss = criterion(pred, pose)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    # Plot Loss Curve
    plt.figure()
    plt.plot(range(1, len(loss_history)+1), loss_history, marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.close()

    # Prediction vs Ground Truth Plot
    model.eval()
    with torch.no_grad():
        for emg, pose in loader:
            pred = model(emg)
            break

    joint = 0
    plt.figure(figsize=(10, 4))
    plt.plot(pose[0, joint].cpu().numpy(), label="True")
    plt.plot(pred[0, joint].cpu().numpy(), label="Predicted")
    plt.title(f"Joint {joint} Pose Prediction")
    plt.xlabel("Timestep")
    plt.ylabel("Angle")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"joint_{joint}_prediction_vs_truth.png")
    plt.close()


if __name__ == "__main__":
    train_with_logging()
