import torch
import torch.nn as nn

class EMGPoseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, 128, 1)
            nn.Flatten(),             # (B, 128)
            nn.Linear(128, 20)        # Output pose dimension (20 joints)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)
