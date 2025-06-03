import torch
import torch.nn as nn

class EMGtoPoseModel(nn.Module):
    def __init__(self, input_channels=16, output_channels=20):
        super(EMGtoPoseModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(128, output_channels)

    def forward(self, x):
        # x: (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)  # -> (batch_size, channels, seq_len)
        x = self.encoder(x)
        return self.fc(x)  # -> (batch_size, output_channels)
