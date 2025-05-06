import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.emg_pose_model import EMGPoseNet
from utils.data_loader import EMGDataset

def train():
    dataset = EMGDataset("data/metadata.csv", "data", split="train")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = EMGPoseNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()
