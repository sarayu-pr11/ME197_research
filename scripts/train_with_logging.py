import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models.emg_pose_model import EMGPoseNet
from utils.data_loader import EMGDataset

def train_with_logging():
    # Initialize dataset and dataloader
    dataset = EMGDataset(
        metadata_csv="data/emg2pose_dataset_mini/metadata.csv",
        data_dir="data/emg2pose_dataset_mini",
        split="val"  # or "val" depending on your CSV
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize model and training components
    model = EMGPoseNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(log_dir="runs/emg_experiment")

    # Training loop
    model.train()
    for epoch in range(10):
        total_loss = 0
        for emg, pose in loader:
            pred = model(emg)
            loss = criterion(pred, pose)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "emg_model.pth")
    writer.close()

if __name__ == "__main__":
    train_with_logging()
