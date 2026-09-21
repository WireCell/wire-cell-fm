"""Grouping, and the mini-PointNet that turns a group into a token.

A point cloud arrives as `(B, N, 4)`, `(x, y, z, log_q)` in normalised units, with `lengths`
marking the real points of each event. `PointcloudGrouping` picks the group centres with
`cnms` over every point, gathers each centre's ball, reduces it to `group_max_points` points and
expresses the group's coordinates relative to its centre in units of the radius.
`MaskedMiniPointNet` maps each group to one token; its batch norm ignores padded slots through
the mask, its max-pool does not, which is why a padded slot repeats the group's first point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .ops import ball_query, cnms, grid_ball_query, masked_gather, sample_farthest_points

__all__ = [
    "Groups",
    "MaskedBatchNorm1d",
    "MaskedMiniPointNet",
    "PointOrderEncoder",
    "PointcloudGrouping",
    "PointcloudTokenizer",
]


@dataclass
class Groups:
    """One tokenisation of a padded batch, `T` groups of `K` points each."""

    groups: Tensor
    """`(B, T, K, 4)`: xyz relative to the centre over the radius, `log_q` absolute. Padded
    slots repeat the group's first point; padded groups are zero."""
    centers: Tensor
    """`(B, T, 3)` in normalised units."""
    emb_mask: Tensor
    """`(B, T)` bool, the groups that become tokens."""
    point_mask: Tensor
    """`(B, T, K)` bool, the real points of each group."""
    idx: Tensor
    """`(B, T, K)` index of each point into the padded batch, -1 where the slot is padding."""


class PointcloudGrouping(nn.Module):
    """Centres by `cnms`, members by ball query, `group_max_points` of them by farthest point
    sampling (`reduction_method="fps"`) or by charge (`"energy"`).

    Every retained centre with at least one point becomes a token.

    `context_length` caps the centres per event; the `T` of a `Groups` is the batch's largest
    retained count or that cap, whichever is smaller, so it varies from batch to batch. Retained
    centres keep ascending index order, so a truncation drops the highest indices, which on a
    `(channel, tick)` cloud sorted by channel is one side of the image.

    `pitch` is the lattice spacing of the cloud in its own units, the backbone's `scale` for a
    pixel cloud. With it both neighbour queries go through `grid_ball_query`, which returns
    what `ball_query` returns at a cost that does not grow with the square of the event. It
    requires distinct lattice sites per event; `grid_ball_query` says what happens otherwise.
    """

    def __init__(
        self,
        *,
        num_groups: int,
        group_max_points: int,
        group_radius: float,
        group_upscale_points: int,
        overlap_factor: float,
        context_length: int,
        reduction_method: str = "fps",
        pitch: float | None = None,
    ):
        super().__init__()
        if reduction_method not in ("fps", "energy"):
            raise ValueError(
                f"reduction_method must be 'fps' or 'energy', got {reduction_method!r}"
            )
        self.num_groups = int(num_groups)
        self.group_max_points = int(group_max_points)
        self.group_radius = float(group_radius)
        self.group_upscale_points = int(group_upscale_points)
        self.overlap_factor = float(overlap_factor)
        self.context_length = int(context_length)
        self.reduction_method = reduction_method
        self.pitch = None if pitch is None else float(pitch)

    def _query(self, p1: Tensor, p2: Tensor, K: int, radius: float, l1: Tensor, l2: Tensor):
        if self.pitch is None:
            return ball_query(p1, p2, K=K, radius=radius, lengths1=l1, lengths2=l2)
        return grid_ball_query(
            p1, p2, K=K, radius=radius, pitch=self.pitch, lengths1=l1, lengths2=l2
        )

    @torch.no_grad()
    def forward(self, points: Tensor, lengths: Tensor) -> Groups:
        B, N, C = points.shape
        xyz = points[..., :3].float()
        centers, n_centers = cnms(
            xyz,
            radius=self.group_radius,
            overlap_factor=self.overlap_factor,
            K=self.num_groups,
            lengths=lengths,
            pitch=self.pitch,
        )
        # `cnms` returns every candidate; the batch is padded to its longest event's centres,
        # not to `context_length`, so the encoder runs over the tokens that exist.
        T = min(self.context_length, max(int(n_centers.max()), 1))
        centers = centers[:, :T]
        n_centers = n_centers.clamp_max(T)
        idx = self._query(
            centers, xyz, self.group_upscale_points, self.group_radius, n_centers, lengths
        )
        # Columns past the fullest group hold only -1; dropping them changes nothing and
        # shrinks what farthest point sampling iterates over.
        idx = idx[:, :, : max(int(idx.ge(0).sum(2).max()), self.group_max_points)]
        idx = self._reduce(points, idx)
        K = self.group_max_points
        point_lengths = idx.ge(0).sum(2)
        point_mask = torch.arange(K, device=idx.device).view(1, 1, K) < point_lengths.unsqueeze(-1)
        emb_mask = point_lengths > 0

        filled = torch.where(idx.lt(0), idx[..., :1].expand_as(idx), idx)
        groups = masked_gather(points, filled)
        groups = groups.clone()
        groups[..., :3] = (groups[..., :3] - centers.unsqueeze(2)) / self.group_radius
        groups = groups * emb_mask.unsqueeze(-1).unsqueeze(-1).to(groups.dtype)
        return Groups(
            groups=groups, centers=centers, emb_mask=emb_mask, point_mask=point_mask, idx=idx
        )

    def _reduce(self, points: Tensor, idx: Tensor) -> Tensor:
        """`(B, G, K_big)` ball members to `(B, G, K)`, -1 padded."""
        B, G, Kbig = idx.shape
        K = self.group_max_points
        if self.reduction_method == "energy":
            q = points[..., 3].unsqueeze(1).expand(-1, G, -1)
            energies = torch.gather(q, 2, idx.clamp(min=0))
            energies = energies.masked_fill(idx.lt(0), float("-inf"))
            top, top_idx = energies.topk(K, dim=2)
            out = torch.gather(idx, 2, top_idx)
            return out.masked_fill(torch.isinf(top), -1)
        grouped = masked_gather(points, idx).reshape(B * G, Kbig, points.shape[-1])
        fps = sample_farthest_points(grouped, lengths=idx.ge(0).sum(2).reshape(B * G), K=K)
        fps = fps.reshape(B, G, K)
        out = torch.gather(idx, 2, fps.clamp(min=0))
        return out.masked_fill(fps.lt(0), -1)

    def extra_repr(self) -> str:
        return (
            f"num_groups={self.num_groups}, group_max_points={self.group_max_points}, "
            f"group_radius={self.group_radius:g}, "
            f"group_upscale_points={self.group_upscale_points}, "
            f"overlap_factor={self.overlap_factor}, context_length={self.context_length}, "
            f"reduction_method={self.reduction_method!r}, pitch={self.pitch}"
        )


class MaskedBatchNorm1d(nn.Module):
    """Batch norm over `(B, C, L)` whose statistics exclude the slots where `mask` is False.

    The running statistics are per rank under DDP, as with `nn.BatchNorm1d`.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B, C, L = x.shape
        if mask is None:
            mask = x.new_ones((B, 1, L))
        mask = mask.to(x.dtype)
        if self.training:
            n = mask.sum().clamp(min=1)
            mean = (x * mask).sum(dim=(0, 2)) / n
            centered = x - mean.view(1, C, 1)
            var = ((centered * mask) ** 2).sum(dim=(0, 2)) / n
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean, var = self.running_mean, self.running_var
            centered = x - mean.view(1, C, 1)
        x = centered / torch.sqrt(var + self.eps).view(1, C, 1) * mask
        return x * self.weight.view(1, C, 1) + self.bias.view(1, C, 1)


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of an integer position, `(N,) -> (N, dim)`."""

    def __init__(self, dim: int):
        super().__init__()
        self.emb_dim = int(dim)

    def forward(self, ts: Tensor) -> Tensor:
        half = self.emb_dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=ts.device) * -emb)
        emb = ts[:, None].to(emb.dtype) * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class PointOrderEncoder(nn.Module):
    """An embedding of each point's slot index in its group, `(B, N, C) -> (1, N, dim)`."""

    def __init__(self, dim: int):
        super().__init__()
        self.time_embed = nn.Sequential(TimeEmbedding(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, points: Tensor) -> Tensor:
        inp = torch.arange(points.shape[1], device=points.device)
        return self.time_embed(inp).unsqueeze(0)


class MaskedMiniPointNet(nn.Module):
    """Two shared-weight 1x1 convolution stacks with a max-pool between and after them.

    `local(points, mask)` is the first stack alone, `(M, 256, S)`: one feature per point before
    any pooling, which is what the backbone's `local` tap scatters back to the pixels.
    `equivariant=True` adds a slot-order embedding to it, so the energy head can read a group's
    points in a fixed order.
    """

    def __init__(self, channels: int, feature_dim: int, equivariant: bool = False):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(channels, 128, 1, bias=False),
            MaskedBatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False),
            MaskedBatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, feature_dim, 1),
        )
        self.equivariant = bool(equivariant)
        if self.equivariant:
            self.position_encoder = PointOrderEncoder(256)

    @staticmethod
    def _run(stack: nn.Sequential, feature: Tensor, mask: Tensor) -> Tensor:
        for layer in stack:
            feature = (
                layer(feature, mask) if isinstance(layer, MaskedBatchNorm1d) else layer(feature)
            )
        return feature

    def local(self, points: Tensor, mask: Tensor) -> Tensor:
        """`(M, S, C)` points and `(M, 1, S)` mask to `(M, 256, S)` per-point features."""
        feature = self._run(self.first_conv, points.transpose(2, 1), mask)
        if self.equivariant:
            feature = feature + self.position_encoder(points).transpose(2, 1)
        return feature

    def forward(self, points: Tensor, mask: Tensor) -> Tensor:
        """`(M, S, C)` or `(B, G, S, C)` points to one feature per group."""
        reshape = points.ndim == 4
        if reshape:
            B, G, S, C = points.shape
            points = points.reshape(B * G, S, C)
            mask = mask.reshape(B * G, 1, S)
        feature = self.local(points, mask)
        pooled = feature.max(dim=2, keepdim=True).values
        feature = torch.cat([pooled.expand(-1, -1, feature.shape[2]), feature], dim=1)
        feature = self._run(self.second_conv, feature, mask)
        out = feature.max(dim=2).values
        return out.reshape(B, G, -1) if reshape else out


class PointcloudTokenizer(nn.Module):
    """`PointcloudGrouping` followed by `MaskedMiniPointNet`: `(B, N, 4)` to `(B, T, token_dim)`.

    The PointNet runs with autocast disabled, in the parameters' dtype, whatever precision the
    surrounding forward uses.
    """

    def __init__(self, *, grouping: PointcloudGrouping, num_channels: int, token_dim: int):
        super().__init__()
        self.grouping = grouping
        self.token_dim = int(token_dim)
        self.embedding = MaskedMiniPointNet(num_channels, token_dim)

    def forward(self, points: Tensor, lengths: Tensor) -> tuple[Tensor, Groups]:
        g = self.grouping(points, lengths)
        with torch.autocast(device_type=points.device.type, enabled=False):
            out = self.embedding(
                g.groups[g.emb_mask].float(), g.point_mask[g.emb_mask].unsqueeze(1)
            )
        tokens = out.new_zeros((g.groups.shape[0], g.groups.shape[1], self.token_dim))
        tokens[g.emb_mask] = out
        return tokens, g
