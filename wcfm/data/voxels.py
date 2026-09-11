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
