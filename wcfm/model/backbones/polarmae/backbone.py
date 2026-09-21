"""`PolarMAEBackbone`: PoLAr-MAE's tokenizer, ViT encoder, decoder and mask token as a `Backbone`.

Two surfaces. `forward(xs, inject, taps)` is the `Backbone` contract: every pixel of `xs` is a
point, every group is a visible token, and the encoder's output is carried back to the pixels by
an inverse-distance average over the `upsample_k` nearest group centres, so `out` sits on the
input's coordinates and offsets. `tokenize`, `encode` and `decode` are the pieces a training
module composes with a token mask in between; the decoder and `mask_token` are held here so a
checkpoint carries them whichever module wrote it.

A pixel `(channel, tick)` with normalised charge `q` becomes
`((channel - center[0]) * scale, (tick - center[1]) * scale, 0, q)`. `group_radius_px` is in
pixels and is multiplied by `scale` before it reaches the grouping.

`inject_roles` is empty, so the objectives that reinject coordinates refuse this backbone at
validation; the DINO and distillation objectives and extraction take it as it is.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.types.voxels import Voxels

from ..base import Backbone, FeatureBundle, Injection
from .ops import knn_points
from .tokenizer import Groups, PointcloudGrouping, PointcloudTokenizer
from .transformer import NORMS, LearnedPositionalEncoder, Transformer

__all__ = ["PolarMAEBackbone", "TokenBundle", "random_token_mask"]

LOCAL_DIM = 256
"""Width of the mini-PointNet's per-point feature, fixed by `MaskedMiniPointNet.first_conv`."""


@dataclass
class TokenBundle:
    """A tokenised batch: what `encode` and `decode` consume."""

    tokens: Tensor
    """`(B, T, D)`, zero where `groups.emb_mask` is False."""
    pos: Tensor
    """`(B, T, D)` positional encoding of the group centres."""
    groups: Groups

    @property
    def emb_mask(self) -> Tensor:
        return self.groups.emb_mask

    @property
    def lengths(self) -> Tensor:
        """Real tokens per event, `(B,)`."""
        return self.groups.emb_mask.sum(1)


@torch.no_grad()
def random_token_mask(lengths: Tensor, T: int, ratio: float) -> tuple[Tensor, Tensor]:
    """`(masked, visible)`, both `(B, T)` bool, masking `int(ratio * lengths[b])` of the first
    `lengths[b]` positions of each row uniformly at random. Positions past `lengths[b]` are in
    neither."""
    B = lengths.shape[0]
    device = lengths.device
    valid = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    if ratio == 0:
        return torch.zeros_like(valid), valid
    scores = torch.rand(B, T, device=device).masked_fill(~valid, float("inf"))
    _, order = scores.sort(dim=1)
    n_mask = (ratio * lengths).to(torch.int64)
    pick = torch.arange(T, device=device).unsqueeze(0) < n_mask.unsqueeze(1)
    rows = torch.arange(B, device=device).unsqueeze(1).expand(B, T)
    masked = torch.zeros_like(valid)
    visible = torch.zeros_like(valid)
    masked[rows, order] = pick & valid
    visible[rows, order] = (~pick) & valid
    return masked, visible


class PolarMAEBackbone(Backbone):
    """Tokenizer, positional encoder, encoder, decoder and mask token.

    Module names follow the PoLAr-MAE state dict, so its weights load under a prefix rename:
    `tokenizer.embedding.*`, `pos_embed.pos_enc.*`, `encoder.blocks.*`, `decoder.blocks.*`,
    `decoder.norm.*`, `mask_token`.

    `out` is the encoder's token feature at every pixel, `out_dim == embed_dim`. The one tap,
    `local`, is the mini-PointNet's per-point feature averaged over the groups a pixel falls in,
    zero for a pixel in no group; its stride is 1. `norm` is documented on
    `wcfm.model.backbones.polarmae.transformer`.
    """

    TAPS: ClassVar[tuple[str, ...]] = ("local",)
    TAP_STRIDE: ClassVar[dict[str, int]] = {"local": 1}
    INJECT_TAPS: ClassVar[tuple[str, ...]] = ()
    GRAD_GROUPS: ClassVar[dict[str, tuple[str, ...]]] = {
        "tokenizer": ("tokenizer.",),
        "pos_embed": ("pos_embed.",),
        "encoder": ("encoder.",),
        "decoder": ("decoder.",),
        "mask_token": ("mask_token",),
    }

    def __init__(
        self,
        *,
        center: Iterable[float] = (480.0, 563.0, 0.0),
        scale: float = 1.0 / 600.0,
        group_radius_px: float = 5.0,
        num_init_groups: int = 256,
        context_length: int = 512,
        group_max_points: int = 32,
        group_upscale_points: int = 256,
        overlap_factor: float = 0.5,
        reduction_method: str = "fps",
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.05,
        drop_path: float = 0.25,
        decoder_depth: int = 4,
        norm: str = "layer",
        upsample_k: int = 5,
    ):
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        center = tuple(float(c) for c in center)
        if len(center) != 3:
            raise ValueError(f"center must have three entries, got {center}")
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32), persistent=False)
        self.scale = float(scale)
        self.group_radius_px = float(group_radius_px)
        self.out_dim = int(embed_dim)
        self.inject_roles = ()
        self.upsample_k = int(upsample_k)
        self.norm = norm

        grouping = PointcloudGrouping(
            num_groups=num_init_groups,
            group_max_points=group_max_points,
            group_radius=self.group_radius_px * self.scale,
            group_upscale_points=group_upscale_points,
            overlap_factor=overlap_factor,
            context_length=context_length,
            reduction_method=reduction_method,
        )
        self.tokenizer = PointcloudTokenizer(grouping=grouping, num_channels=4, token_dim=embed_dim)
        self.pos_embed = LearnedPositionalEncoder(embed_dim)
        self.encoder = Transformer(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=drop_path,
            postnorm=False,
            norm=norm,
        )
        self.decoder = Transformer(
            embed_dim=embed_dim,
            depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            drop_path=drop_path,
            postnorm=True,
            norm=norm,
        )
        self.mask_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02, a=-0.02, b=0.02)

    @property
    def grouping(self) -> PointcloudGrouping:
        return self.tokenizer.grouping

    def tap_dim(self, tap: str) -> int:
        if tap != "local":
            raise ValueError(f"{type(self).__name__} has no tap {tap!r}; it has {self.TAPS}")
        return LOCAL_DIM

    # ------------------------------------------------------------------ point clouds

    def points_from(self, xs: Voxels) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor]]:
        """A padded `(B, N, 4)` cloud in normalised units, the real count per event, and the
        `(event, slot)` of every input row so a `(B, N, ...)` result maps back onto `xs`."""
        coords = xs.coordinate_tensor
        feats = xs.feature_tensor
        device = coords.device
        offsets = xs.offsets.to(device=device, dtype=torch.int64)
        counts = offsets[1:] - offsets[:-1]
        B = int(counts.shape[0])
        M = int(coords.shape[0])
        N = int(counts.max()) if B and M else 0
        event = torch.repeat_interleave(torch.arange(B, device=device), counts)
        slot = torch.arange(M, device=device) - offsets[:-1][event]
        xyz = feats.new_zeros((M, 3), dtype=torch.float32)
        xyz[:, :2] = coords[:, :2].to(torch.float32)
        points = feats.new_zeros((B, max(N, 1), 4), dtype=torch.float32)
        points[event, slot, :3] = (xyz - self.center) * self.scale
        points[event, slot, 3] = feats[:, 0].to(torch.float32)
        return points, counts, (event, slot)

    def tokenize(self, points: Tensor, lengths: Tensor) -> TokenBundle:
        tokens, groups = self.tokenizer(points, lengths)
        return TokenBundle(tokens=tokens, pos=self.pos_embed(groups.centers), groups=groups)

    def encode(self, bundle: TokenBundle, visible: Tensor | None = None) -> Tensor:
        """The encoder over `visible` tokens, every real token when `visible` is None."""
        mask = bundle.emb_mask if visible is None else visible
        return self.encoder(bundle.tokens, bundle.pos, mask)

    def decode(
        self, bundle: TokenBundle, encoded: Tensor, visible: Tensor, masked: Tensor
    ) -> Tensor:
        """The decoder over every real token, `mask_token` standing in at the masked ones."""
        corrupted = encoded * visible.unsqueeze(-1).to(encoded.dtype) + self.mask_token.to(
            encoded.dtype
        ) * masked.unsqueeze(-1).to(encoded.dtype)
        return self.decoder(corrupted, bundle.pos, bundle.emb_mask)

    def upsample(
        self, tokens: Tensor, bundle: TokenBundle, points: Tensor, lengths: Tensor
    ) -> Tensor:
        """Token features to every point, `(B, N, D)`: an inverse-distance average over the
        `upsample_k` nearest real group centres of the point's own event. An event with fewer
        centres than `upsample_k` averages over the ones it has, so the result does not depend
        on which events share the batch."""
        dists, idx = knn_points(
            points[..., :3],
            bundle.groups.centers,
            lengths1=lengths,
            lengths2=bundle.lengths,
            K=self.upsample_k,
            return_sorted=False,
        )
        eps = torch.finfo(torch.float32).eps
        w = 1.0 / (dists + eps)
        w = w / w.sum(2, keepdim=True).clamp_min(eps)
        B, N, K = idx.shape
        D = tokens.shape[-1]
        gathered = torch.gather(
            tokens.float().unsqueeze(1).expand(-1, N, -1, -1),
            2,
            idx.unsqueeze(-1).expand(-1, -1, -1, D),
        )
        return (gathered * w.unsqueeze(-1)).sum(2)

    def local_features(self, bundle: TokenBundle, n_points: int) -> tuple[Tensor, Tensor]:
        """`(local, covered)`: the mini-PointNet's per-point feature averaged over the groups
        each point belongs to, `(B, N, 256)`, and `(B, N)` bool for the points in at least one
        group. Uncovered points are zero."""
        g = bundle.groups
        B, T, K, _ = g.groups.shape
        device = g.groups.device
        valid = g.point_mask & g.emb_mask.unsqueeze(-1)
        with torch.autocast(device_type=device.type, enabled=False):
            feature = self.tokenizer.embedding.local(
                g.groups[g.emb_mask].float(), g.point_mask[g.emb_mask].unsqueeze(1)
            )
        feature = feature * g.point_mask[g.emb_mask].unsqueeze(1).to(feature.dtype)
        per_slot = feature.new_zeros((B, T, LOCAL_DIM, K))
        per_slot[g.emb_mask] = feature
        per_slot = per_slot.permute(0, 1, 3, 2)
        rows = (torch.arange(B, device=device).view(B, 1, 1) * n_points + g.idx.clamp(min=0))[valid]
        buf = feature.new_zeros((B * n_points, LOCAL_DIM))
        cnt = feature.new_zeros((B * n_points,))
        buf.index_add_(0, rows, per_slot[valid])
        cnt.index_add_(0, rows, torch.ones_like(rows, dtype=feature.dtype))
        local = (buf / cnt.clamp(min=1).unsqueeze(-1)).view(B, n_points, LOCAL_DIM)
        return local, cnt.gt(0).view(B, n_points)

    # ------------------------------------------------------------------- the contract

    @staticmethod
    def _voxels(xs: Voxels, feats: Tensor) -> Voxels:
        return Voxels(
            batched_coordinates=xs.batched_coordinates,
            batched_features=CatFeatures(feats, xs.offsets),
            offsets=xs.offsets,
        )

    def forward(
        self, xs: Voxels, inject: Injection | None = None, taps: Iterable[str] = ()
    ) -> FeatureBundle:
        taps = self.check_request(inject, taps)
        points, lengths, (event, slot) = self.points_from(xs)
        bundle = self.tokenize(points, lengths)
        encoded = self.encode(bundle)
        out = self.upsample(encoded, bundle, points, lengths)[event, slot]
        values: dict[str, Tensor] = {}
        if "local" in taps:
            local, _ = self.local_features(bundle, points.shape[1])
            values["local"] = local[event, slot]
        return FeatureBundle(
            out=self._voxels(xs, out),
            taps={t: self._voxels(xs, values[t]) for t in taps},
            injected=None,
        )
