"""Sparse-aware masking for the student views: the base class and the pixel, block and region
maskers.

Every masker answers the same call and returns the same `MaskResult`:

    result = masker(voxels)
    result.student          # Voxels: what survived
    result.keep             # bool over the INPUT rows, batch order -- for gathering truth
    result.masked_coords    # per image [N_b, 2]: what was removed
    result.masked_feats     # per image [N_b, F]: the features that were there (charge target)
    result.cand_coords      # per image, or None: occupancy candidates (region masker only)
    result.occ_targets      # per image, or None: their labels

`masked_feats` is always collected: it is a boolean index of a tensor already in hand, so
making it optional would cost every caller a branch to save nothing. A term that needs the
target declares `requires_masking` and finds it there.

`keep` is what lets the view carry truth. Cropping and masking subset and reorder pixels, so
per-pixel labels are gathered along with them rather than bolted on afterwards.

Every masker guarantees at least one voxel survives per non-empty image, and an empty image
produces aligned, empty entries in every list, so `student.offsets` stays length `B + 1`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.types.voxels import Voxels


@dataclass
class MaskResult:
    student: Voxels
    keep: Tensor
    masked_coords: list[Tensor]
    masked_feats: list[Tensor]
    cand_coords: list[Tensor] | None = None
    occ_targets: list[Tensor] | None = None

    @property
    def n_masked(self) -> int:
        return int(sum(c.shape[0] for c in self.masked_coords))


# --------------------------------------------------------------------------------- helpers


def batch_index(voxels: Voxels) -> Tensor:
    """Per-voxel batch id `[Ntot]`, on the voxel device, derived from offsets."""
    counts = (voxels.offsets[1:] - voxels.offsets[:-1]).to(
        device=voxels.coordinate_tensor.device, dtype=torch.int64
    )
    return torch.repeat_interleave(torch.arange(counts.shape[0], device=counts.device), counts)


def segment_rank(batch_idx_sorted: Tensor, counts_per_b: Tensor) -> Tensor:
    """0-based rank within each batch segment, for an array already grouped by batch."""
    seg_start = torch.cat([counts_per_b.new_zeros(1), counts_per_b.cumsum(0)[:-1]])
    M = batch_idx_sorted.shape[0]
    return torch.arange(M, device=batch_idx_sorted.device) - seg_start[batch_idx_sorted]


def assemble_masked_voxels(
    voxels: Voxels, batch_idx: Tensor, masked: Tensor
) -> tuple[Voxels, list[Tensor], list[Tensor]]:
    """A per-voxel boolean mask into `(student, masked_coords, masked_feats)`.

    The input is batch-ordered, so a boolean index preserves that grouping and the student's
    offsets are the cumulative kept counts. Splitting the dropped rows per image costs the one
    host sync in this function.
    """
    B = len(voxels.offsets) - 1
    coords, feats = voxels.coordinate_tensor, voxels.feature_tensor
    coord_dim, feat_dim = coords.shape[1], feats.shape[1]
    keep = ~masked

    keep_counts = torch.bincount(batch_idx[keep], minlength=B)
    new_offsets = torch.cat([keep_counts.new_zeros(1), keep_counts.cumsum(0)]).cpu()
    student = Voxels(
        batched_coordinates=IntCoords(coords[keep], offsets=new_offsets),
        batched_features=CatFeatures(feats[keep], offsets=new_offsets),
        offsets=new_offsets,
    )
    split_sizes = torch.bincount(batch_idx[masked], minlength=B).tolist()
    masked_coords = list(torch.split(coords[masked], split_sizes))
    masked_feats = list(torch.split(feats[masked], split_sizes))
    if len(masked_coords) != B:  # splitting a 0-row tensor yields one piece, not B
        masked_coords = [coords.new_zeros(0, coord_dim) for _ in range(B)]
        masked_feats = [feats.new_zeros(0, feat_dim) for _ in range(B)]
    return student, masked_coords, masked_feats


def cap_negatives(
    cand_b: Tensor,
    occ: Tensor,
    B: int,
    neg_per_pos: float | None = None,
    max_neg: int | None = None,
) -> Tensor:
    """Sub-sample the empty candidates to bound memory, keeping every positive.

    The per-image negative budget is the smallest of whichever caps are set: `neg_per_pos`
    keeps K empties per positive (class balance fixed, memory scales with the busiest image);
    `max_neg` keeps at most N empties per image (memory additive, the realised ratio floats).
    Returns a boolean keep-mask aligned to `cand_b`. With no cap set nothing is dropped.
    """
    is_pos = occ > 0.5
    keep = torch.ones(cand_b.shape[0], dtype=torch.bool, device=cand_b.device)
    neg_idx = (~is_pos).nonzero(as_tuple=False).squeeze(1)
    if neg_idx.numel() == 0:
        return keep

    budget = None
    if neg_per_pos is not None:
        n_pos = torch.bincount(cand_b[is_pos], minlength=B)
        budget = (float(neg_per_pos) * n_pos.double()).round().long()
    if max_neg is not None:
        abs_cap = torch.full((B,), int(max_neg), dtype=torch.int64, device=cand_b.device)
        budget = abs_cap if budget is None else torch.minimum(budget, abs_cap)
    if budget is None:
        return keep

    nb = cand_b[neg_idx]
    neg_counts = torch.bincount(nb, minlength=B)
    order = (nb.double() + torch.rand(neg_idx.shape[0], device=cand_b.device).double()).argsort()
    rank = segment_rank(nb[order], neg_counts)
    keep[neg_idx[order][~(rank < budget[nb[order]])]] = False
    return keep


def label_candidates(
    cand_coords: Tensor,
    cand_b: Tensor,
    masked_coords: Tensor,
    masked_b: Tensor,
    B: int,
    image_w: int,
    image_h: int,
    stride: int = 1,
    neg_per_pos: float | None = None,
    max_neg: int | None = None,
) -> tuple[list[Tensor], list[Tensor]]:
    """Label occupancy candidates and split them per image.

    A candidate is positive exactly when it coincides with a voxel masking removed -- the
    label is membership in the removed set and nothing else is consulted, so a candidate
    cannot be positive because of structure the student can still see. `stride > 1` maps
    both onto a coarser grid by floor division and deduplicates, turning the label into a
    block-OR over that cell. Returns coordinates in those coarse units.
    """
    device = cand_coords.device
    s = int(stride) if stride else 1

    if s > 1:
        cand_coords = torch.div(cand_coords, s, rounding_mode="floor")
        if masked_coords.shape[0] > 0:
            masked_coords = torch.div(masked_coords, s, rounding_mode="floor")
        W = (int(image_w) + s - 1) // s
        H = (int(image_h) + s - 1) // s
        k = cand_b * (W * H + W) + cand_coords[:, 1].long() * W + cand_coords[:, 0].long()
        srt = torch.argsort(k, stable=True)
        first = torch.ones(k.shape[0], dtype=torch.bool, device=device)
        first[1:] = k[srt][1:] != k[srt][:-1]
        uniq = torch.zeros(k.shape[0], dtype=torch.bool, device=device)
        uniq[srt[first]] = True
        cand_coords, cand_b = cand_coords[uniq], cand_b[uniq]
    else:
        W, H = int(image_w), int(image_h)
    STRIDE = W * H + W  # exceeds any in-image key y*W + x

    order = torch.argsort(cand_b, stable=True)
    cand_b, cand_coords = cand_b[order], cand_coords[order]

    cand_key = cand_b * STRIDE + cand_coords[:, 1].long() * W + cand_coords[:, 0].long()
    if masked_coords.shape[0] > 0:
        m_key = masked_b * STRIDE + masked_coords[:, 1].long() * W + masked_coords[:, 0].long()
        occ = torch.isin(cand_key, m_key).float()
    else:
        occ = torch.zeros(cand_coords.shape[0], device=device)

    if neg_per_pos is not None or max_neg is not None:
        keep = cap_negatives(cand_b, occ, B, neg_per_pos=neg_per_pos, max_neg=max_neg)
        cand_b, cand_coords, occ = cand_b[keep], cand_coords[keep], occ[keep]

    counts = torch.bincount(cand_b, minlength=B).tolist()
    cand_list = list(torch.split(cand_coords, counts))
    occ_list = list(torch.split(occ, counts))
    if len(cand_list) != B:
        cand_list = [cand_coords.new_zeros(0, 2) for _ in range(B)]
        occ_list = [occ.new_zeros(0) for _ in range(B)]
    return cand_list, occ_list


# ---------------------------------------------------------------------------------- maskers


class Masker(ABC):
    """The shape every masker has. A subclass decides which rows go, and this class turns that
    decision into a `MaskResult` the same way every time."""

    # True for a masker whose geometry is defined on the full canvas and therefore cannot run
    # on a crop. `SslModule.validate` reads it.
    requires_full_canvas: ClassVar[bool] = False
    # True for a masker that enumerates occupancy candidates.
    builds_candidates: bool = False

    @abstractmethod
    def _select(self, voxels: Voxels, batch_idx: Tensor) -> tuple[Tensor, object | None]:
        """A boolean `masked` over all rows, plus anything the subclass needs afterwards."""

    def _extras(
        self, voxels: Voxels, batch_idx: Tensor, masked: Tensor, state: object | None
    ) -> tuple[list[Tensor] | None, list[Tensor] | None]:
        return None, None

    def __call__(self, voxels: Voxels) -> MaskResult:
        B = len(voxels.offsets) - 1
        coords, feats = voxels.coordinate_tensor, voxels.feature_tensor
        if coords.shape[0] == 0:
            empty_c = [coords.new_zeros(0, coords.shape[1]) for _ in range(B)]
            empty_f = [feats.new_zeros(0, feats.shape[1]) for _ in range(B)]
            cands = empty_c if self.builds_candidates else None
            occ = (
                [coords.new_zeros(0).float() for _ in range(B)] if self.builds_candidates else None
            )
            return MaskResult(
                student=voxels,
                keep=torch.ones(0, dtype=torch.bool, device=coords.device),
                masked_coords=empty_c,
                masked_feats=empty_f,
                cand_coords=cands,
                occ_targets=occ,
            )
        batch_idx = batch_index(voxels)
        masked, state = self._select(voxels, batch_idx)
        student, masked_coords, masked_feats = assemble_masked_voxels(voxels, batch_idx, masked)
        cands, occ = self._extras(voxels, batch_idx, masked, state)
        return MaskResult(
            student=student,
            keep=~masked,
            masked_coords=masked_coords,
            masked_feats=masked_feats,
            cand_coords=cands,
            occ_targets=occ,
        )


class PixelMasker(Masker):
    """Independent per-voxel dropout at `ratio`, per image; at least one voxel survives."""

    def __init__(self, ratio: float = 0.5):
        if not 0.0 <= ratio < 1.0:
            raise ValueError(f"ratio must be in [0, 1), got {ratio}")
        self.ratio = float(ratio)

    def _select(self, voxels: Voxels, batch_idx: Tensor) -> tuple[Tensor, None]:
        B = len(voxels.offsets) - 1
        device = voxels.coordinate_tensor.device
        masked = torch.zeros(voxels.coordinate_tensor.shape[0], dtype=torch.bool, device=device)
        for b in range(B):
            start, end = int(voxels.offsets[b]), int(voxels.offsets[b + 1])
            N = end - start
            if N == 0:
                continue
            n_keep = max(1, N - int(N * self.ratio))
            perm = torch.randperm(N, device=device)
            masked[start + perm[n_keep:]] = True
        return masked, None


class BlockMasker(Masker):
    """Windows of `[+-win_ch, +-win_tick]` around K random active voxels, K estimated to cover
    `ratio` of the voxels: `E[covered] = 1 - (1 - p)^K` with the per-block coverage rate `p`
    learned by EMA within a call. Because blocks overlap, the masked fraction varies around
    `ratio`."""

    def __init__(self, ratio: float = 0.5, win_ch: int = 5, win_tick: int = 5):
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        self.ratio = float(ratio)
        self.win_ch = int(win_ch)
        self.win_tick = int(win_tick)
        self._block_area = (2 * self.win_ch + 1) * (2 * self.win_tick + 1)
        self._p_eff: float | None = None

    def _select(self, voxels: Voxels, batch_idx: Tensor) -> tuple[Tensor, None]:
        # Reset per call so the EMA from one crop type does not contaminate the next; it still
        # converges within a call across the batch, which is when it helps.
        self._p_eff = None
        B = len(voxels.offsets) - 1
        coords = voxels.coordinate_tensor
        device = coords.device
        masked = torch.zeros(coords.shape[0], dtype=torch.bool, device=device)
        for b in range(B):
            start, end = int(voxels.offsets[b]), int(voxels.offsets[b + 1])
            N = end - start
            if N == 0:
                continue
            coords_i = coords[start:end]
            p_formula = min(self._block_area, N) / N
            p = self._p_eff if self._p_eff is not None else p_formula
            p = max(1e-6, min(p, 1.0 - 1e-6))
            K = min(math.ceil(math.log(1.0 - self.ratio) / math.log(1.0 - p)), N - 1)
            if K <= 0:
                continue
            centers = coords_i[torch.randperm(N, device=device)[:K]]
            diff = coords_i.unsqueeze(1) - centers.unsqueeze(0)
            mask_bool = (
                (diff[..., 0].abs() <= self.win_ch) & (diff[..., 1].abs() <= self.win_tick)
            ).any(dim=1)
            if mask_bool.all():  # guarantee a survivor
                mask_bool[torch.randint(0, N, (1,), device=device)] = False
            n_masked = int(mask_bool.sum())
            actual = n_masked / N
            if 0.0 < actual < 1.0:
                p_measured = 1.0 - (1.0 - actual) ** (1.0 / K)
                self._p_eff = (
                    p_measured if self._p_eff is None else 0.9 * self._p_eff + 0.1 * p_measured
                )
            masked[start:end] = mask_bool
        return masked, None


class RegionMasker(Masker):
    """Whole cells of a fixed grid over the canvas, `"wipe"` (empty them) or `"randomize"`
    (thin them at `r2`). Only cells holding at least one active voxel are eligible.

    This is the one masker that knows the geometry of what it removed, so it can enumerate
    occupancy candidates: every pixel of a wiped cell, labelled positive exactly where charge
    was. Under `"wipe"` that labelling is exact, because only cells left completely empty are
    enumerated. Under `"randomize"` the survivors inside a chosen cell are active pixels the
    label calls empty, so that flavour leaks by construction and is refused as a reconstruction
    target upstream.

    The grid is defined on the canvas rather than on the data, which is why it takes `image_w`
    and `image_h` and why it cannot run on a crop.
    """

    requires_full_canvas: ClassVar[bool] = True

    def __init__(
        self,
        image_w: int,
        image_h: int,
        cell_w: int = 70,
        cell_h: int = 100,
        r1: float = 0.5,
        r2: float = 0.75,
        flavor: str = "wipe",
        wipe_max: float = 0.75,
        build_candidates: bool = False,
        cand_stride: int = 2,
        neg_per_pos: float | None = None,
        max_neg: int | None = None,
    ):
        if flavor not in ("wipe", "randomize"):
            raise ValueError(f"unknown flavor {flavor!r}; expected 'wipe' or 'randomize'")
        # A partial edge cell densifies to a different candidate count than an interior one
        # and silently skews the positive rate.
        if image_w % cell_w or image_h % cell_h:
            raise ValueError(
                f"cell {cell_w}x{cell_h} does not tile a {image_w}x{image_h} canvas evenly"
            )
        # An odd cell puts a wiped cell's coarse footprint half a cell out of step with the
        # grid, so a candidate can land on a coarse cell that still holds surviving charge.
        if cell_w % cand_stride or cell_h % cand_stride:
            raise ValueError(
                f"cell {cell_w}x{cell_h} must divide by the candidate stride {cand_stride}"
            )
        self.image_w, self.image_h = int(image_w), int(image_h)
        self.cell_w, self.cell_h = int(cell_w), int(cell_h)
        self.r1, self.r2 = float(r1), float(r2)
        self.flavor = flavor
        self.wipe_max = float(wipe_max)
        self.builds_candidates = bool(build_candidates)
        self.cand_stride = int(cand_stride)
        self.neg_per_pos = neg_per_pos
        self.max_neg = max_neg
        self.n_cols = self.image_w // self.cell_w
        self.n_rows = self.image_h // self.cell_h
        self.n_cells = self.n_cols * self.n_rows

    def _densify(self, take_cell: Tensor, ucells: Tensor, device, coord_dtype):
        """Every pixel of each selected cell, at full resolution: `(coords [N, 2], b [N])`."""
        taken = ucells[take_cell]
        if taken.shape[0] == 0:
            return (
                torch.zeros(0, 2, dtype=coord_dtype, device=device),
                torch.zeros(0, dtype=torch.int64, device=device),
            )
        cb = torch.div(taken, self.n_cells, rounding_mode="floor")
        local = taken - cb * self.n_cells
        base_x = (local % self.n_cols) * self.cell_w
        base_y = torch.div(local, self.n_cols, rounding_mode="floor") * self.cell_h
        ox, oy = torch.meshgrid(
            torch.arange(self.cell_w, device=device),
            torch.arange(self.cell_h, device=device),
            indexing="ij",
        )
        ox, oy = ox.reshape(-1), oy.reshape(-1)
        cand_x = (base_x[:, None] + ox[None, :]).reshape(-1)
        cand_y = (base_y[:, None] + oy[None, :]).reshape(-1)
        cand_b = cb[:, None].expand(-1, ox.shape[0]).reshape(-1)
        return torch.stack([cand_x, cand_y], dim=1).to(coord_dtype), cand_b

    def _select(self, voxels: Voxels, batch_idx: Tensor) -> tuple[Tensor, tuple]:
        B = len(voxels.offsets) - 1
        coords = voxels.coordinate_tensor
        device = coords.device
        Ntot = coords.shape[0]
        N = torch.bincount(batch_idx, minlength=B)

        cx = torch.div(coords[:, 0], self.cell_w, rounding_mode="floor")
        cy = torch.div(coords[:, 1], self.cell_h, rounding_mode="floor")
        cell_global = batch_idx * self.n_cells + (cy * self.n_cols + cx)
        ucells, inv, ucounts = torch.unique(cell_global, return_inverse=True, return_counts=True)
        cell_batch = torch.div(ucells, self.n_cells, rounding_mode="floor")
        n_u = ucells.shape[0]
        n_active = torch.bincount(cell_batch, minlength=B)

        ckey = torch.rand(n_u, device=device)
        corder = (cell_batch.double() + ckey.double()).argsort()
        cb_sorted = cell_batch[corder]
        rank_sorted = segment_rank(cb_sorted, n_active)

        if self.flavor == "wipe":
            csum = ucounts[corder].cumsum(0)
            seg_start = torch.cat([n_active.new_zeros(1), n_active.cumsum(0)[:-1]])
            before = torch.cat([csum.new_zeros(1), csum])[seg_start[cb_sorted]]
            cum_within = csum - before
            cap = self.wipe_max * N.double()
            take_sorted = (cum_within <= cap[cb_sorted]) | (rank_sorted == 0)
            take_cell = torch.zeros(n_u, dtype=torch.bool, device=device)
            take_cell[corder] = take_sorted
            masked = take_cell[inv]

            # The forced first cell can overshoot the ceiling on its own; release the surplus
            # at random so the student is never left with nothing to encode.
            n_cap = cap.long()
            mc = torch.bincount(batch_idx[masked], minlength=B)
            surplus = (mc - n_cap).clamp(min=0)
            if bool((surplus > 0).any()):
                m_idx = masked.nonzero(as_tuple=False).squeeze(1)
                mb = batch_idx[m_idx]
                morder = (
                    mb.double() + torch.rand(m_idx.shape[0], device=device).double()
                ).argsort()
                mrank = segment_rank(mb[morder], mc)
                masked[m_idx[morder][mrank < surplus[mb[morder]]]] = False

            # Densify only cells that ended up completely empty: the release above can put
            # visible charge back inside a taken cell, and a candidate there would label a
            # pixel the student can still see as empty.
            has_survivor = torch.zeros(n_u, dtype=torch.bool, device=device)
            has_survivor[inv[~masked]] = True
            dense_cell = take_cell & ~has_survivor
        else:
            n_sel = torch.minimum(
                n_active, (self.r1 * n_active.double()).round().long().clamp(min=1)
            )
            sel_sorted = rank_sorted < n_sel[cb_sorted]
            sel_cell = torch.zeros(n_u, dtype=torch.bool, device=device)
            sel_cell[corder] = sel_sorted
            dense_cell = sel_cell
            masked = sel_cell[inv] & (torch.rand(Ntot, device=device) < self.r2)
        return masked, (dense_cell, ucells)

    def _extras(self, voxels, batch_idx, masked, state):
        if not self.builds_candidates:
            return None, None
        dense_cell, ucells = state
        coords = voxels.coordinate_tensor
        B = len(voxels.offsets) - 1
        cand_coords, cand_b = self._densify(dense_cell, ucells, coords.device, coords.dtype)
        m_idx = masked.nonzero(as_tuple=False).squeeze(1)
        return label_candidates(
            cand_coords,
            cand_b,
            coords[m_idx],
            batch_idx[m_idx],
            B,
            self.image_w,
            self.image_h,
            stride=self.cand_stride,
            neg_per_pos=self.neg_per_pos,
            max_neg=self.max_neg,
        )
