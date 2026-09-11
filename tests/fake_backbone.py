"""A CPU-runnable ``Backbone`` and the batches to feed it, so ``SslModule`` can be tested
through the real ``Trainer`` without a sparse convolution.

``LinearBackbone`` is an MLP over the per-voxel features with the **real** injection helper
(``inject_into_skip``) in front of it, so what the tests exercise about injection -- the
projection, the dedupe against the skip, the role tokens, the reported ``injected`` set -- is
the production code, and only the convolutions are stood in for. It lives in ``tests/`` for
the same reason ``toy.py`` does: it knows what a backbone is, which no framework module may.

``make_batch`` builds a real ``Batch`` -- ``Voxels`` plus meta -- with one charge channel that
is strictly positive (so the log transform is defined) and a per-pixel truth array that
**encodes the coordinate** (``x * 1000 + y``), which is what makes a misaligned gather
detectable rather than plausible.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.types.voxels import Voxels

from wcfm.data.voxels import Batch, offsets_from_counts, voxels_from
from wcfm.model.backbones import Backbone, FeatureBundle, Injection, inject_into_skip


class LinearBackbone(Backbone):
    TAPS = ("enc0", "hidden")
    TAP_STRIDE = {"enc0": 1, "hidden": 1}
    INJECT_TAPS = ("enc0",)
    GRAD_GROUPS = {"net": ("net.",), "head": ("head.",), "tokens": ("tokens.",)}

    def __init__(
        self,
        in_dim: int = 1,
        hidden: int = 8,
        out_dim: int = 8,
        inject_roles: Iterable[str] = ("masked",),
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.inject_roles = tuple(inject_roles)
        self.net = nn.Linear(in_dim, hidden)
        self.head = nn.Linear(hidden, out_dim)
        self.tokens = nn.ParameterDict(
            {f"enc0__{r}": nn.Parameter(torch.randn(in_dim) * 0.02) for r in self.inject_roles}
        )
        self._tap_width = {"enc0": int(in_dim), "hidden": int(hidden)}

    def tap_dim(self, tap: str) -> int:
        return self._tap_width[tap]

    def forward(self, xs: Voxels, inject: Injection | None = None, taps=()) -> FeatureBundle:
        taps = self.check_request(inject, taps)
        injected = []
        skip = xs
        if inject is not None:
            tokens = {r: self.tokens[f"enc0__{r}"] for r in self.inject_roles}
            skip, injected = inject_into_skip(xs, "enc0", 1, inject, tokens)
        feats = skip.feature_tensor.float()
        hidden = torch.relu(self.net(feats))
        out = self.head(hidden)

        def vox(f: Tensor) -> Voxels:
            return Voxels(
                batched_coordinates=skip.batched_coordinates,
                batched_features=CatFeatures(f, skip.offsets),
                offsets=skip.offsets,
            )

        values = {"enc0": feats, "hidden": hidden}
        return FeatureBundle(
            out=vox(out),
            taps={t: vox(values[t]) for t in taps},
            injected=Injection(injected) if inject is not None else None,
        )


# ------------------------------------------------------------------------------ batches


def coord_label(coords: Tensor) -> Tensor:
    """The per-pixel truth used throughout: a value that names its own coordinate."""
    return coords[:, 0].long() * 1000 + coords[:, 1].long()


def make_batch(
    counts=(40, 30),
    *,
    width: int = 64,
    height: int = 48,
    seed: int = 0,
    pixel_truth: bool = True,
    blob: bool = False,
) -> Batch:
    """A batch of images with unique coordinates and positive charge. ``blob=True`` gathers
    the pixels around the centre so a cropper has somewhere to aim."""
    g = torch.Generator().manual_seed(seed)
    coords, feats, labels = [], [], []
    for n in counts:
        if n == 0:
            c = torch.zeros(0, 2, dtype=torch.int64)
        elif blob:
            centre = torch.tensor([width // 2, height // 2])
            xy = (centre + torch.randn(n * 2, 2, generator=g) * (min(width, height) / 6)).round()
            xy[:, 0].clamp_(0, width - 1)
            xy[:, 1].clamp_(0, height - 1)
            c = torch.unique(xy.long(), dim=0)[:n]
        else:
            c = torch.stack(
                [
                    torch.randint(0, width, (n,), generator=g),
                    torch.randint(0, height, (n,), generator=g),
                ],
                dim=1,
            )
            c = torch.unique(c, dim=0)
        coords.append(c.to(torch.int32))
        feats.append(torch.rand(c.shape[0], 1, generator=g) * 100.0 + 1.0)
        labels.append(coord_label(c).numpy())
    offsets = offsets_from_counts([c.shape[0] for c in coords])
    voxels = voxels_from(torch.cat(coords), torch.cat(feats), offsets)
    meta = {
        "event_key": [f"ev{i}" for i in range(len(counts))],
        "label": torch.arange(len(counts), dtype=torch.long),
    }
    if pixel_truth:
        meta["pixel_labels"] = labels
    return Batch(voxels, meta)


class _Batches(Dataset):
    def __init__(self, n: int, **kw):
        self.batches = [make_batch(seed=i, **kw) for i in range(n)]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, i):
        return self.batches[i]


def batch_loader(steps: int = 4, **kw) -> DataLoader:
    """Pre-built ``Batch`` objects under ``batch_size=None``, the sharded backend's shape."""
    return DataLoader(_Batches(steps, **kw), batch_size=None, shuffle=False)


def rows_of(vox: Voxels, b: int) -> Tensor:
    return vox.coordinate_tensor[int(vox.offsets[b]) : int(vox.offsets[b + 1])]


def as_pairs(coords: Tensor) -> set[tuple[int, int]]:
    return {(int(x), int(y)) for x, y in coords.tolist()}


__all__ = [
    "LinearBackbone",
    "as_pairs",
    "batch_loader",
    "coord_label",
    "make_batch",
    "rows_of",
    "np",
]
