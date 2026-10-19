"""The DINO projection head: `n_layers` of `Linear -> GELU` (no activation on the last), L2
normalise, then a bias-free `Linear` whose rows, the prototypes, are renormalised to unit length
on every forward. Each output is the cosine between a feature and a prototype, so no prototype
can win the softmax by growing its norm. It runs on the `Voxels` feature tensor directly through
WarpConvNet's modules, with no dense materialisation."""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.activations import GELU
from warpconvnet.nn.modules.mlp import Linear
from warpconvnet.nn.modules.sequential import Sequential


class DINOProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int = 4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(Linear(in_dim if i == 0 else hidden_dim, hidden_dim, bias=True))
            if i < n_layers - 1:
                layers.append(GELU())
        self.mlp = Sequential(*layers)
        # Applied through `forward` with its rows normalised, never called as a module. The
        # stored weight keeps its norm; the EMA and weight decay act on it, the forward does not
        # see it. `tests/test_dino_head.py` pins the invariance.
        self.last = Linear(hidden_dim, out_dim, bias=False)
        self.out_dim = int(out_dim)

    def forward(self, x: Voxels) -> Voxels:
        h = F.normalize(self.mlp(x).feature_tensor, dim=-1)
        w = F.normalize(self.last.block.weight, dim=1)
        return x.replace(batched_features=F.linear(h, w))
