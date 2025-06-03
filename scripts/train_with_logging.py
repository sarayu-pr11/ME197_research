import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend to avoid display errors

import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from models.emg_pose_model import EMGtoPoseModel
from utils.data_loader import EMGDataset, pad_collate


def train_with_logging():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = EMGDataset(
        metadata_csv="data/emg2pose_dataset_mini/metadata.csv",
        data_dir="data/emg2pose_dataset_mini",
        split="val"
    )
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=pad_collate)

    # Initialize model, loss, optimizer
    model = EMGtoPoseModel().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Store loss values for plotting
    losses = []

    # Training loop (1 epoch for demonstration)
    for emg, pose in loader:
        emg = emg.to(device)
        pose = pose.to(device)

        # Fix input shape: (batch, seq_len, channels) → (batch, channels, seq_len)
        emg = emg.permute(0, 2, 1)

        optimizer.zero_grad()
        pred = model(emg)
        pose_last = pose[:, :, -1]  # take last timestep
        loss = criterion(pred, pose_last)
        loss.backward()
        optimizer.step()

        print(f"[INFO] Loss: {loss.item():.4f}")
        losses.append(loss.item())

    # Plot loss
    plt.plot(losses)
    plt.title("Loss per Batch")
    plt.xlabel("Batch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    print("[INFO] Saved loss plot to 'loss_plot.png'")

if __name__ == "__main__":
    train_with_logging()
