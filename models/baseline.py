import torch.nn as nn

class BaselineRegressor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)
