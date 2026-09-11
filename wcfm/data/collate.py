"""Collation for the map-style backends.

`collate` is the DataLoader's `collate_fn`: it takes the `(voxels, meta)` singletons a map-style
dataset yields and concatenates them into one `Batch(voxels, meta)`. A `Batch` always carries
meta, so there is no meta-less variant: a consumer that wants one field projects it out itself,
and the engine never branches on the shape of a batch.

`collate_meta` is the meta half on its own, because the sharded reader assembles its voxels
itself and still needs meta batched the same way.

The sharded reader does not come through `collate`: it is an `IterableDataset` that yields
batches already assembled, driven with `DataLoader(batch_size=None)`. See build.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from wcfm.data.voxels import Batch, offsets_from_counts, voxels_from

# Event-level truth: one value per image, so a crop does not disturb it.
_EVENT_LONG = ("label", "nu_pdg", "nu_ccnc", "nu_intType")
_EVENT_FLOAT = ("nu_energy",)
_EVENT_STACK = ("vertex_xyz",)
_EVENT_PASSTHROUGH = ("event_key",)
# Per-pixel truth: CSR-aligned to /coords, so cropping and masking DO reorder it. Kept as a
# per-sample list rather than concatenated, because the consumer that gathers it needs the
# per-sample boundaries anyway and a flat tensor would hide them.
_PIXEL = ("pixel_labels", "pixel_energyfrac", "pixel_trackid", "pixel_truth_q")
# Public, because the model's augment stage gathers exactly these along with the pixels a
# crop or mask selected -- they are CSR-aligned to the coordinates, so a reorder of one is a
# reorder of the other.
PIXEL_TRUTH_KEYS = _PIXEL


def collate(items: Sequence[tuple[Any, dict]]) -> Batch:
    """Collate `(voxels, meta)` singletons into one `Batch`.

    Each input `Voxels` holds exactly one sample. Truth tiers absent from the source are absent
    from the result, and a source carrying no truth at all yields no truth keys, so a consumer
    checks for a key rather than for a flag.
    """
    voxels_list, metas = zip(*items, strict=True)

    coords = [v.coordinate_tensor for v in voxels_list]
    feats = [v.feature_tensor for v in voxels_list]
    offsets = offsets_from_counts([c.shape[0] for c in coords])
    voxels = voxels_from(torch.cat(coords, dim=0), torch.cat(feats, dim=0), offsets)

    return Batch(voxels, collate_meta(metas))


def collate_meta(metas: Sequence[dict]) -> dict[str, Any]:
    """Stack a sequence of per-sample meta dicts into one batched dict.

    Shared by `collate` and by the sharded reader, which assembles its own voxels but needs meta
    in exactly this shape. Two implementations of it would drift.
    """
    if not metas or not metas[0]:
        return {}
    first = metas[0]
    out: dict[str, Any] = {}
    for key in _EVENT_LONG:
        if key in first:
            out[key] = torch.tensor([m[key] for m in metas], dtype=torch.long)
    for key in _EVENT_FLOAT:
        if key in first:
            out[key] = torch.tensor([m[key] for m in metas], dtype=torch.float32)
    for key in _EVENT_STACK:
        if key in first:
            out[key] = torch.stack([_as_tensor(m[key]) for m in metas], dim=0)
    for key in _EVENT_PASSTHROUGH:
        if key in first:
            out[key] = [m[key] for m in metas]
    for key in _PIXEL:
        if key in first:
            out[key] = [m[key] for m in metas]
    return out


def _as_tensor(x: Any) -> Tensor:
    return x if isinstance(x, Tensor) else torch.as_tensor(x, dtype=torch.float32)
