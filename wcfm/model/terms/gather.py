"""Coordinate matching between sparse outputs.

Both functions match by coordinate, never by row position: nothing in the sparse convolution
API promises rows come back in the order they went in, and a target misaligned by one row
trains perfectly happily while meaning nothing. Each voxel gets a batch-unique flat key
`b * S + y * W + x`, so one sort and one `searchsorted` do the work of B per-image
intersections.

`gather_at_coords` raises on an unmatched request by default. Dropping a miss silently means a
term scores whatever happened to overlap and discards the rest of what it enumerated, with
nothing in the metrics to say so. A term that genuinely expects misses -- the occupancy term
intersecting against what was injected -- asks for `on_miss="drop"` and gets back the count of
what it dropped.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from warpconvnet.geometry.types.voxels import Voxels

OnMiss = Literal["raise", "drop"]


def match_and_gather(
    s_out: Voxels,
    s_backbone: Voxels,
    t_out: Voxels,
    masked_coords_per_batch: list[Tensor] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    """Student and teacher features at the coordinates both hold.

    Returns `(s_feats, s_bb_feats, t_feats, counts, is_masked)`: features at the intersection
    in batch-major order, the per-image matched counts, and -- when `masked_coords_per_batch`
    is given -- a boolean per matched row saying whether the student received a mask token
    there. `is_masked` is `None` when no masked coordinates were supplied or nothing matched.
    """
    B = len(s_out.offsets) - 1
    device = s_out.feature_tensor.device
    s_coords, t_coords = s_out.coordinate_tensor, t_out.coordinate_tensor

    def _empty():
        return (
            s_out.feature_tensor.new_zeros(0, s_out.feature_tensor.shape[1]),
            s_backbone.feature_tensor.new_zeros(0, s_backbone.feature_tensor.shape[1]),
            t_out.feature_tensor.new_zeros(0, t_out.feature_tensor.shape[1]),
            torch.zeros(B, dtype=torch.int64, device=device),
            None,
        )

    if s_coords.shape[0] == 0 or t_coords.shape[0] == 0:
        return _empty()

    W = int(max(s_coords[:, 0].max().item(), t_coords[:, 0].max().item())) + 1
    H = int(max(s_coords[:, 1].max().item(), t_coords[:, 1].max().item())) + 1
    S = H * W

    def _keys(coords, offsets):
        counts = (offsets[1:] - offsets[:-1]).to(device)
        b_idx = torch.repeat_interleave(torch.arange(B, device=device), counts)
        return b_idx * S + coords[:, 1].long() * W + coords[:, 0].long(), b_idx

    s_keys, b_s = _keys(s_coords, s_out.offsets)
    t_keys, _ = _keys(t_coords, t_out.offsets)
    t_sorted, t_order = t_keys.sort()
    pos = torch.searchsorted(t_sorted, s_keys).clamp(max=t_sorted.shape[0] - 1)
    valid = t_sorted[pos] == s_keys
    s_idx = valid.nonzero(as_tuple=False).squeeze(1)
    if s_idx.numel() == 0:
        return _empty()
    t_idx = t_order[pos[s_idx]]
    counts = torch.bincount(b_s[s_idx], minlength=B)

    s_feats = s_out.feature_tensor[s_idx]
    s_bb_feats = s_backbone.feature_tensor[s_idx]
    t_feats = t_out.feature_tensor[t_idx]

    is_masked = None
    if masked_coords_per_batch is not None:
        m_per_image = list(masked_coords_per_batch)[:B]
        m_counts = torch.tensor([m.shape[0] for m in m_per_image], dtype=torch.int64, device=device)
        if int(m_counts.sum()) > 0:
            m_coords = torch.cat(m_per_image, dim=0).to(device)
            b_m = torch.repeat_interleave(torch.arange(B, device=device), m_counts)
            m_keys = b_m * S + m_coords[:, 1].long() * W + m_coords[:, 0].long()
            m_sorted, _ = m_keys.sort()
            matched_keys = s_keys[s_idx]
            mp = torch.searchsorted(m_sorted, matched_keys).clamp(max=m_sorted.shape[0] - 1)
            is_masked = m_sorted[mp] == matched_keys
        else:
            is_masked = torch.zeros(s_idx.shape[0], dtype=torch.bool, device=device)
    return s_feats, s_bb_feats, t_feats, counts, is_masked


def gather_at_coords(
    vox: Voxels,
    coords_per_batch: list[Tensor],
    targets_per_batch: list[Tensor] | None = None,
    *,
    on_miss: OnMiss = "raise",
) -> tuple[Tensor, Tensor | None, Tensor, int]:
    """Read a one-channel prediction at requested coordinates, per image.

    Returns `(pred [M], target [M] or None, counts [B], n_missed)`. Predictions and targets
    stay aligned row for row. A requested coordinate absent from `vox` raises under
    `on_miss="raise"` -- the caller claimed it injected there -- and is dropped along with its
    target under `"drop"`, with the number dropped returned so the caller can count it.
    """
    device = vox.feature_tensor.device
    B = len(vox.offsets) - 1
    req = list(coords_per_batch)[:B]
    n_req = torch.tensor([c.shape[0] for c in req], dtype=torch.int64, device=device)
    out_coords = vox.coordinate_tensor
    total_req = int(n_req.sum())

    if total_req == 0:
        empty = vox.feature_tensor.new_zeros(0)
        return (
            empty,
            empty if targets_per_batch is not None else None,
            torch.zeros(B, dtype=torch.int64, device=device),
            0,
        )
    if out_coords.shape[0] == 0:
        if on_miss == "raise":
            raise RuntimeError(
                f"{total_req} coordinates were requested from an output with no rows at all"
            )
        empty = vox.feature_tensor.new_zeros(0)
        return (
            empty,
            empty if targets_per_batch is not None else None,
            torch.zeros(B, dtype=torch.int64, device=device),
            total_req,
        )

    req_coords = torch.cat(req, dim=0).to(device)
    req_b = torch.repeat_interleave(torch.arange(B, device=device), n_req)
    W = int(max(int(out_coords[:, 0].max()), int(req_coords[:, 0].max()))) + 1
    H = int(max(int(out_coords[:, 1].max()), int(req_coords[:, 1].max()))) + 1
    S = H * W
    o_counts = (vox.offsets[1:] - vox.offsets[:-1]).to(device)
    o_b = torch.repeat_interleave(torch.arange(B, device=device), o_counts)
    o_keys = o_b * S + out_coords[:, 1].long() * W + out_coords[:, 0].long()
    r_keys = req_b * S + req_coords[:, 1].long() * W + req_coords[:, 0].long()

    o_sorted, o_order = o_keys.sort()
    pos = torch.searchsorted(o_sorted, r_keys).clamp(max=o_sorted.shape[0] - 1)
    hit = o_sorted[pos] == r_keys
    n_missed = int((~hit).sum())
    if n_missed and on_miss == "raise":
        miss = req_coords[~hit][:5].tolist()
        raise RuntimeError(
            f"{n_missed} of {total_req} requested coordinates have no prediction "
            f"(first: {miss}). The term asked at coordinates the backbone did not emit; "
            "score the set reported in FeatureBundle.injected, or pass on_miss='drop' if the "
            "miss is expected and counted."
        )

    pred = vox.feature_tensor.reshape(-1)[o_order[pos[hit]]]
    counts = torch.bincount(req_b[hit], minlength=B)
    target = None
    if targets_per_batch is not None:
        tgt = torch.cat([t.reshape(-1) for t in list(targets_per_batch)[:B]], dim=0).to(device)
        target = tgt[hit]
    return pred, target, counts, n_missed
