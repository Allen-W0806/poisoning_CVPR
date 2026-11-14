import numpy as np
import torch
from torch import nn


class EditPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 8)
        self.encoder.requires_grad_(False)
        self.head = nn.Linear(8, 3)

    def forward(self, features):
        return self.head(torch.tanh(self.encoder(features)))

    def distribution(self, images, device):
        features = np.stack([np.asarray(i, dtype=np.float32).mean((0, 1)) / 255 for i in images])
        return torch.distributions.Categorical(logits=self(torch.as_tensor(features, device=device)))
