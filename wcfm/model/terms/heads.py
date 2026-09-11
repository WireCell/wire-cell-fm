"""The DINO projection head: `n_layers` of `Linear -> GELU` (no activation on the last), L2
normalise, then a bias-free `Linear`. It runs on the `Voxels` feature tensor directly through
WarpConvNet's modules, with no dense materialisation."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.activations import GELU
from warpconvnet.nn.modules.mlp import Linear
from warpconvnet.nn.modules.sequential import Sequential


class L2Normalize(nn.Module):
    def forward(self, x: Voxels) -> Voxels:
        feats = F.normalize(x.feature_tensor, dim=-1)
        return Voxels(
            batched_coordinates=x.batched_coordinates,
            batched_features=CatFeatures(feats, x.offsets),
            offsets=x.offsets,
        )


class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(Linear(in_dim if i == 0 else hidden_dim, hidden_dim, bias=True))
            if i < n_layers - 1:
                layers.append(GELU())
        self.mlp = Sequential(*layers)
        self.normalize = L2Normalize()
        self.last = Linear(hidden_dim, out_dim, bias=False)
        self.out_dim = int(out_dim)

    def forward(self, x: Voxels) -> Voxels:
        return self.last(self.normalize(self.mlp(x)))
