""" Common operation for handling WarpConvNet's `Voxels`.
    Definition of the 'Batch' dataclass, which is the common return type of all three readers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.types.voxels import Voxels


def voxels_from(coords: Tensor, feats: Tensor, offsets: Tensor) -> Voxels:
    """Build a batched `Voxels` from concatenated coords, features and CSR offsets.

    A wrapper around WarpConvNet's `Voxels` constructor. `offsets` is the CSR row pointer:
    length `B + 1`, `offsets[0] == 0`, and `offsets[-1] == coords.shape[0]`. It is passed to
    the coordinates, the features and the container alike.
    """
    if coords.shape[0] != feats.shape[0]:
        raise ValueError(
            f"coords and feats disagree on row count: {coords.shape[0]} vs {feats.shape[0]}"
        )
    if int(offsets[0]) != 0 or int(offsets[-1]) != coords.shape[0]:
        raise ValueError(
            f"offsets must run 0..{coords.shape[0]}, got {int(offsets[0])}..{int(offsets[-1])}"
        )
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


def upcast(coarse: Voxels, fine: Voxels, factor: int) -> Tensor:
    """`coarse`'s features at every row of `fine`, in `fine`'s row order: each row takes the
    features of its parent, the `coarse` row in the same image at the fine coordinate
    floor-divided by `factor`.

    A strided convolution emits a voxel under every input voxel, so a fine map and an encoder
    stage computed from it always have the parents; a row without one raises rather than
    leaving a pixel with no feature.
    """
    fc = fine.coordinate_tensor
    feats = coarse.feature_tensor
    if fc.shape[0] == 0:
        return feats.new_zeros(0, feats.shape[1])
    device = fc.device
    fc = torch.div(fc, factor, rounding_mode="floor") if factor > 1 else fc
    cc = coarse.coordinate_tensor
    if cc.shape[0] == 0:
        raise RuntimeError(f"{fc.shape[0]} rows have no parent: the coarse map is empty")
    W = int(max(int(fc[:, 0].max()), int(cc[:, 0].max()))) + 1
    H = int(max(int(fc[:, 1].max()), int(cc[:, 1].max()))) + 1

    def keys(c: Tensor, offsets: Tensor) -> Tensor:
        counts = (offsets[1:] - offsets[:-1]).to(device)
        b = torch.repeat_interleave(torch.arange(counts.shape[0], device=device), counts)
        return b * (H * W) + c[:, 1].long() * W + c[:, 0].long()

    c_sorted, c_order = keys(cc, coarse.offsets).sort()
    f_keys = keys(fc, fine.offsets)
    pos = torch.searchsorted(c_sorted, f_keys).clamp(max=c_sorted.shape[0] - 1)
    n_missed = int((c_sorted[pos] != f_keys).sum())
    if n_missed:
        raise RuntimeError(f"{n_missed} of {f_keys.shape[0]} rows have no parent at {factor}x")
    return feats[c_order[pos]]


def offsets_from_counts(counts: Tensor | list[int]) -> Tensor:
    """CSR offsets from per-sample row counts. The other half of the duplicated incantation."""
    if not isinstance(counts, Tensor):
        counts = torch.tensor(counts, dtype=torch.int64)
    counts = counts.to(torch.int64)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)])


@dataclass(frozen=True)
class Batch:
    """What every backend yields: pixels, and the truth that came with them.

    Voxels: sparse representation native of WarpConvNet. 
    Meta: a dict of truth info (tensors and other objects), depending on how much
          truth is returned.

    """

    voxels: Voxels
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Any]:
        return iter((self.voxels, self.meta))

    @property
    def batch_size(self) -> int:
        return len(self.voxels.offsets) - 1

    def to(self, device: torch.device | str) -> Batch:
        """Move pixels to a device. `meta` tensors move too; lists and strings do not."""
        moved = {
            k: v.to(device) if isinstance(v, Tensor) else v
            for k, v in self.meta.items()
        }
        return Batch(self.voxels.to(device), moved)
