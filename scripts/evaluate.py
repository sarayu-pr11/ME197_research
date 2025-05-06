import torch
from torch.utils.data import DataLoader
from models.emg_pose_model import EMGPoseNet
from utils.data_loader import EMGDataset
import torch.nn.functional as F

def evaluate():
    dataset = EMGDataset("data/metadata.csv", "data", split="val")
    loader = DataLoader(dataset, batch_size=8)

    model = EMGPoseNet()
    model.load_state_dict(torch.load("emg_model.pth"))
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for emg, pose in loader:
            pred = model(emg)
            loss = F.mse_loss(pred, pose)
            total_loss += loss.item()

    print(f"Validation MSE: {total_loss / len(loader):.4f}")

if __name__ == "__main__":
    evaluate()
