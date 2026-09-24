"""The energy term: predict the log-charge of every point of a masked group from its token.

PoLAr-MAE masks whole groups after tokenisation, so the decoder emits one vector per masked
group and nothing per point. A token alone cannot say which of the group's points carries
which charge, so `equivariant_patch_encoder` embeds the group's real point positions, in slot
order, into a vector the size of a token; concatenated with the decoded token it goes through
`energy_decoder`, a 1x1 convolution to one `log_q` per slot. The loss is the mean squared
error over the real points of the masked groups, one flat mean over the batch: a large event
weighs more, and a pixel that falls in two masked groups counts twice. Head, loss and reduction
are the PoLAr-MAE reference's, and "energy" is its name for `log_q`; changing any of them
breaks the comparison with its `loss/train_energy` curve.

`ChargeTerm` asks the same question of the pixel MAE, where masking removes single pixels and
the decoder emits a feature at each: a 1x1 head there, and an L1 normalised per image.
"""


from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..backbones.polarmae import MaskedTokens, PolarMAEBackbone
from ..backbones.polarmae.tokenizer import MaskedMiniPointNet
from .base import GroupTerm, TermOutput


class EnergyTerm(GroupTerm):
    def __init__(self, *, weight: float = 1.0):
        super().__init__(weight=weight)
        self.equivariant_patch_encoder: MaskedMiniPointNet | None = None
        self.energy_decoder: nn.Conv1d | None = None

    def build(self, backbone: PolarMAEBackbone) -> None:
        D = int(backbone.out_dim)
        self.equivariant_patch_encoder = MaskedMiniPointNet(3, D, equivariant=True)
        self.energy_decoder = nn.Conv1d(2 * D, int(backbone.grouping.group_max_points), 1)

    def head_forward(self, tokens: MaskedTokens) -> Tensor:
        """`(M, K)` predicted `log_q` per slot of the `M` masked groups."""
        assert self.equivariant_patch_encoder is not None and self.energy_decoder is not None
        g = tokens.bundle.groups
        groups = g.groups[tokens.masked]  # (M, K, 4)
        mask = g.point_mask[tokens.masked].unsqueeze(1)  # (M, 1, K)
        positions = self.equivariant_patch_encoder(groups[..., :3], mask)  # (M, D)
        inp = torch.cat([positions, tokens.decoded[tokens.masked]], dim=1)  # (M, 2D)
        return self.energy_decoder(inp.transpose(0, 1)).transpose(0, 1)

    def compute(self, tokens: MaskedTokens, head_out: Tensor, ctx: Any) -> TermOutput:
        g = tokens.bundle.groups
        mask = g.point_mask[tokens.masked]  # (M, K)
        if head_out.shape[0] == 0 or not bool(mask.any()):
            loss = head_out.sum() * 0.0
        else:
            target = g.groups[tokens.masked][..., -1]
            loss = F.mse_loss(head_out[mask].float(), target[mask].float())
        return TermOutput(loss=loss, scalars={"loss_energy": float(loss.detach())})
