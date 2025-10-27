import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR3DCNN(nn.Module):
    def __init__(self, input_channels=165, feature_dim=128):
        super(SimCLR3DCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.projector = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dim
        features = self.encoder(x).view(x.size(0), -1)
        return F.normalize(self.projector(features), dim=1)
