import torch.nn as nn
import torch.nn.utils as utils

class LipschitzRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            utils.spectral_norm(nn.Linear(in_dim, 64)),
            nn.ReLU(),
            utils.spectral_norm(nn.Linear(64, 1))
        )

    def forward(self, x):
        return self.model(x)
