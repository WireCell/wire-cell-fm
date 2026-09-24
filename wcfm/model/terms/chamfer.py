"""The chamfer term: rebuild the points of every masked group from its decoded token.

PoLAr-MAE masks whole groups, so what is missing is a set of points rather than a value at a
known coordinate. `increase_dim` maps a decoded token to `group_max_points` predicted points of
`channels` values each, `(x, y, z, log_q)` in the group's own frame: coordinates relative to
the centre over the radius, charge absolute. The predicted points carry no slot order, so the
loss is the bidirectional Chamfer distance against the group's real points, both sides cut to
the group's real point count, averaged over the masked groups of the batch. This term learns
where the points are; `EnergyTerm` learns how much charge each one carries.

`OccupancyTerm` asks the where-question of the pixel MAE, where the candidate coordinates are
enumerated up front and the answer is a yes or no per candidate.
"""


from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from ..backbones.polarmae import MaskedTokens, PolarMAEBackbone
from ..backbones.polarmae.ops import chamfer_distance
from .base import GroupTerm, TermOutput


class ChamferTerm(GroupTerm):
    def __init__(self, *, weight: float = 1.0, channels: int = 4):
        super().__init__(weight=weight)
        if channels not in (3, 4):
            raise ValueError(
                f"channels must be 3 (positions) or 4 (positions and charge), got {channels}"
            )
        self.channels = int(channels)
        self.increase_dim: nn.Conv1d | None = None
        self.group_max_points = 0

    def build(self, backbone: PolarMAEBackbone) -> None:
        self.group_max_points = int(backbone.grouping.group_max_points)
        self.increase_dim = nn.Conv1d(backbone.out_dim, self.channels * self.group_max_points, 1)

    def head_forward(self, tokens: MaskedTokens) -> Tensor:
        """`(M, K, channels)` predicted points for the `M` masked groups of the batch."""
        assert self.increase_dim is not None, "build() was not called"
        x = tokens.decoded[tokens.masked]  # (M, D)
        up = self.increase_dim(x.transpose(0, 1)).transpose(0, 1)  # a 1x1 conv over (D, M)
        return up.reshape(x.shape[0], self.group_max_points, self.channels)

    def compute(self, tokens: MaskedTokens, head_out: Tensor, ctx: Any) -> TermOutput:
        g = tokens.bundle.groups
        target = g.groups[tokens.masked][..., : self.channels]
        lengths = g.point_mask[tokens.masked].sum(-1)
        if head_out.shape[0] == 0:
            loss = head_out.sum() * 0.0
        else:
            loss = chamfer_distance(head_out.float(), target.float(), lengths, lengths)
        return TermOutput(loss=loss, scalars={"loss_chamfer": float(loss.detach())})
