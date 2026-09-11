"""Views carry truth; scope is a selector.

`Augment` turns one `Batch` into a `ViewPlan`: crops (or the whole image as the single view),
then a mask per student view. Every `View` carries the student's voxels, the same crop before
masking (`clean`, what the teacher sees for a global view), the `MaskResult` if any, and
`meta`, the truth gathered along with the pixels. Cropping and masking subset and reorder rows,
so per-pixel labels are indexed by the same selection in the same order, and
`tests/test_model_ssl.py` asserts the alignment.

`ViewPlan` also carries the source batch, so a term that needs the full uncropped image has
somewhere to get it: `View.clean` is the crop, not the image.

A term comparing two student views is not built. `Term.compute` receives one bundle and one
view index. Every bundle in a step is alive at once, since the view loop runs inside
`SslModule.forward`, so nothing prevents such a term -- it would be a different
`TrainingModule` rather than a flag here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor
from warpconvnet.geometry.types.voxels import Voxels

from wcfm.data.collate import PIXEL_TRUTH_KEYS
from wcfm.data.voxels import Batch

from .cropping import Cropper
from .masking import Masker, MaskResult


@dataclass
class View:
    voxels: Voxels
    """What the student encodes: the crop, masked."""
    clean: Voxels
    """The same crop before masking: what the teacher encodes when this view is global."""
    is_global: bool
    mask: MaskResult | None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def masked_coords(self) -> list[Tensor] | None:
        return self.mask.masked_coords if self.mask is not None else None

    @property
    def batch_size(self) -> int:
        return len(self.voxels.offsets) - 1


@dataclass
class ViewPlan:
    views: list[View]
    source: Batch
    n_global: int

    @property
    def globals(self) -> list[View]:
        return self.views[: self.n_global]

    @property
    def masked(self) -> bool:
        return any(v.mask is not None for v in self.views)

    @property
    def n_views(self) -> int:
        return len(self.views)


def select_meta(meta: dict[str, Any], index: list[Tensor] | None) -> dict[str, Any]:
    """The truth for a subset of rows. Event-level keys pass through; per-pixel keys are
    indexed per image by `index` (`None` means every row)."""
    if index is None:
        return dict(meta)
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if key in PIXEL_TRUTH_KEYS and isinstance(value, list):
            out[key] = [_take(arr, idx) for arr, idx in zip(value, index, strict=True)]
        else:
            out[key] = value
    return out


def _take(arr: Any, idx: Tensor) -> Any:
    if isinstance(arr, Tensor):
        return arr[idx.to(arr.device)]
    return arr[idx.cpu().numpy()]


def _keep_index(keep: Tensor, offsets: Tensor) -> list[Tensor]:
    """Per-image row indices that survived a mask, from the boolean over all rows."""
    out: list[Tensor] = []
    for b in range(len(offsets) - 1):
        s, e = int(offsets[b]), int(offsets[b + 1])
        out.append(keep[s:e].nonzero(as_tuple=False).squeeze(1))
    return out


def _compose_index(outer: list[Tensor], inner: list[Tensor]) -> list[Tensor]:
    """`outer[b][inner[b]]`: rows of the source that survived the crop and then the mask."""
    return [o[i.to(o.device)] for o, i in zip(outer, inner, strict=True)]


class Augment:
    """Crops, then a mask per student view. Either half may be `None`."""

    def __init__(self, cropper: Cropper | None = None, masker: Masker | None = None):
        self.cropper = cropper
        self.masker = masker

    @property
    def n_global(self) -> int:
        return self.cropper.n_global if self.cropper is not None else 1

    @property
    def n_views(self) -> int:
        return self.cropper.n_crops if self.cropper is not None else 1

    def __call__(self, batch: Batch) -> ViewPlan:
        voxels, meta = batch.voxels, batch.meta
        if self.cropper is not None:
            crops = self.cropper(voxels)
            pieces = [(c.voxels, c.index, c.is_global) for c in crops]
        else:
            pieces = [(voxels, None, True)]

        views: list[View] = []
        for clean, index, is_global in pieces:
            mask: MaskResult | None = None
            student = clean
            view_index = index
            if self.masker is not None:
                mask = self.masker(clean)
                student = mask.student
                kept = _keep_index(mask.keep, clean.offsets)
                view_index = kept if index is None else _compose_index(index, kept)
            views.append(
                View(
                    voxels=student,
                    clean=clean,
                    is_global=is_global,
                    mask=mask,
                    meta=select_meta(meta, view_index),
                )
            )
        return ViewPlan(views=views, source=batch, n_global=self.n_global)

    def __repr__(self) -> str:
        return f"Augment(cropper={self.cropper!r}, masker={self.masker!r})"


__all__ = ["Augment", "View", "ViewPlan", "select_meta"]
