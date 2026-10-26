"""FEN and DMN networks from paper Table 1 and Section 2."""
import copy
import torch
from torch import nn


class FusionEncoder(nn.Module):
    """G_g -> 32 -> 16 -> 1; Eq. (7)/(10)."""
    def __init__(self, n_variables: int):
        super().__init__()
        self.layer1 = nn.Linear(n_variables, 32)
        self.layer2 = nn.Linear(32, 16)
        self.bottleneck = nn.Linear(16, 1)

    def forward(self, x):
        h = torch.relu(self.layer1(x))
        dimension_reduction = torch.relu(self.layer2(h))
        feature = self.bottleneck(dimension_reduction)
        return dimension_reduction, feature


class FusionDecoder(nn.Module):
    """1 -> 16 -> 32 -> G_g; Eq. (8)."""
    def __init__(self, n_variables: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(1, 16), nn.ReLU(),
                                    nn.Linear(16, 32), nn.ReLU(),
                                    nn.Linear(32, n_variables))

    def forward(self, z):
        return self.layers(z)


class FusionNetwork(nn.Module):
    def __init__(self, n_variables: int):
        super().__init__()
        self.query = FusionEncoder(n_variables)
        self.key = copy.deepcopy(self.query)
        self.decoder = FusionDecoder(n_variables)
        for p in self.key.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        q_d, q_z = self.query(x)
        with torch.no_grad():
            k_d, k_z = self.key(x)
        return self.decoder(q_z), q_d, q_z, k_d, k_z

    @torch.no_grad()
    def update_key(self, momentum: float):
        # Paper Eq. (13): phi <- mu * phi + (1-mu) * theta.
        for k, q in zip(self.key.parameters(), self.query.parameters()):
            k.lerp_(q, 1.0 - momentum)


class DiscriminativeMappingNetwork(nn.Module):
    """C -> 32 (tanh) -> C; decoder C -> 32 (tanh) -> C (Table 1)."""
    def __init__(self, n_groups: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_groups, 32), nn.Tanh(),
                                     nn.Linear(32, n_groups))
        self.decoder = nn.Sequential(nn.Linear(n_groups, 32), nn.Tanh(),
                                     nn.Linear(32, n_groups))

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)
