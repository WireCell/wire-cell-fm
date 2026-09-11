"""Activity-aware multi-crop.

Selected voxels keep their original coordinates, with no translation to a crop-local origin,
so the intersection between two crops is found by coordinate matching on the backbone outputs.
That is safe because sparse convolutions compute positions as `coord // stride` regardless of
the origin.

Each crop comes back as a `CropResult` carrying the per-image row indices it selected, so the
truth riding along with the pixels -- `pixel_labels` and its siblings, CSR-aligned to the
coordinates -- is gathered along with them. A supervised per-pixel term cannot be written
without that.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.types.voxels import Voxels


@dataclass
class CropResult:
    voxels: Voxels
    """The crop, batched over the B images."""
    index: list[Tensor]
    """Per image: the rows of that image's slice of the source that this crop selected."""
    is_global: bool


def _sample_crop_wh(img_w, img_h, scale_range, aspect_range) -> tuple[int, int]:
    area = img_w * img_h
    scale = random.uniform(*scale_range)
    target_area = scale * area
    aspect = math.exp(random.uniform(math.log(aspect_range[0]), math.log(aspect_range[1])))
    crop_w = int(round(math.sqrt(target_area * aspect)))
    crop_h = int(round(math.sqrt(target_area / aspect)))
    return max(1, min(crop_w, img_w)), max(1, min(crop_h, img_h))


def _propose_box(anchor_xy, crop_w, crop_h, img_w, img_h) -> tuple[int, int, int, int]:
    ax, ay = anchor_xy
    left = max(0, min(int(round(ax - crop_w / 2)), img_w - crop_w))
    top = max(0, min(int(round(ay - crop_h / 2)), img_h - crop_h))
    return left, top, left + crop_w, top + crop_h


def _fallback_centre_box(img_w, img_h, scale_range, aspect_range) -> tuple[int, int, int, int]:
    crop_w, crop_h = _sample_crop_wh(img_w, img_h, scale_range, aspect_range)
    left = (img_w - crop_w) // 2
    top = (img_h - crop_h) // 2
    return left, top, left + crop_w, top + crop_h


def _in_box(coords: Tensor, box: tuple[int, int, int, int]) -> Tensor:
    left, top, right, bottom = box
    return (
        (coords[:, 0] >= left)
        & (coords[:, 0] < right)
        & (coords[:, 1] >= top)
        & (coords[:, 1] < bottom)
    )


class Cropper:
    """`n_global + n_local` crops per image, anchored on a blurred activity heatmap."""

    def __init__(
        self,
        image_w: int,
        image_h: int,
        n_global: int = 2,
        n_local: int = 4,
        global_scale=(0.4, 1.0),
        local_scale=(0.05, 0.2),
        aspect_ratio=(0.75, 1.333),  # main()'s default, which every archived run took
        blur_sigma_px: float = 10.0,
        heatmap_power: float = 1.0,
        min_active_pixels: int = 10,
        max_attempts: int = 50,
    ):
        if n_global < 1:
            raise ValueError("n_global must be >= 1: the teacher encodes the global views")
        self.image_w, self.image_h = int(image_w), int(image_h)
        self.n_global, self.n_local = int(n_global), int(n_local)
        self.global_scale = tuple(float(s) for s in global_scale)
        self.local_scale = tuple(float(s) for s in local_scale)
        self.aspect_ratio = tuple(float(a) for a in aspect_ratio)
        self.blur_sigma_px = float(blur_sigma_px)
        self.heatmap_power = float(heatmap_power)
        self.min_active_pixels = int(min_active_pixels)
        self.max_attempts = int(max_attempts)
        self._n_crops = self.n_global + self.n_local
        self._init_blur_kernel(self.blur_sigma_px)

    @property
    def n_crops(self) -> int:
        return self._n_crops

    def _init_blur_kernel(self, sigma: float) -> None:
        if sigma <= 0:
            self._blur_kernel_h = self._blur_kernel_v = None
            self._blur_pad = 0
            return
        radius = int(3 * sigma + 0.5)
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k1d = torch.exp(-0.5 * (x / sigma) ** 2)
        k1d = k1d / k1d.sum()
        self._blur_kernel_h = k1d.reshape(1, 1, 1, -1)
        self._blur_kernel_v = k1d.reshape(1, 1, -1, 1)
        self._blur_pad = radius

    def _blur(self, A: Tensor) -> Tensor:
        if self._blur_kernel_h is None:
            return A.clone()
        x = A.unsqueeze(1)
        kh = self._blur_kernel_h.to(x.device, x.dtype)
        kv = self._blur_kernel_v.to(x.device, x.dtype)
        pad = min(self._blur_pad, x.shape[-1] - 1)
        x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), kh[..., : 2 * pad + 1])
        pad_v = min(self._blur_pad, x.shape[-2] - 1)
        x = F.conv2d(F.pad(x, (0, 0, pad_v, pad_v), mode="reflect"), kv[:, :, : 2 * pad_v + 1])
        return x.squeeze(1)

    def __call__(self, voxels: Voxels) -> list[CropResult]:
        B = len(voxels.offsets) - 1
        device = voxels.coordinate_tensor.device
        W, H = self.image_w, self.image_h

        A = torch.zeros(B, H, W, device=device)
        for b in range(B):
            s, e = int(voxels.offsets[b]), int(voxels.offsets[b + 1])
            if e > s:
                c = voxels.coordinate_tensor[s:e]
                A[b, c[:, 1].long().clamp(0, H - 1), c[:, 0].long().clamp(0, W - 1)] = 1.0
        Hp = self._blur(A).clamp(min=0)
        del A
        if self.heatmap_power != 1.0:
            Hp = Hp**self.heatmap_power
        Hp_flat = Hp.view(B, -1)
        row_sums = Hp_flat.sum(dim=1, keepdim=True)
        uniform = torch.ones(1, Hp_flat.shape[1], device=device) / Hp_flat.shape[1]
        Hp_flat = torch.where(row_sums > 1e-12, Hp_flat, uniform)
        n_samples = self._n_crops * self.max_attempts
        anchor_flat = torch.multinomial(Hp_flat, n_samples, replacement=True)
        anchor_x = (anchor_flat % W).cpu()
        anchor_y = (anchor_flat // W).cpu()
        del Hp, Hp_flat, anchor_flat

        crop_idx: list[list[Tensor]] = [[] for _ in range(self._n_crops)]
        for b in range(B):
            s, e = int(voxels.offsets[b]), int(voxels.offsets[b + 1])
            coords_b = voxels.coordinate_tensor[s:e]
            N = e - s
            cursor = 0
            for k in range(self._n_crops):
                scale = self.global_scale if k < self.n_global else self.local_scale
                if N == 0:
                    crop_idx[k].append(torch.zeros(0, dtype=torch.long, device=device))
                    continue
                kidx = None
                for _ in range(self.max_attempts):
                    ax, ay = int(anchor_x[b, cursor]), int(anchor_y[b, cursor])
                    cursor += 1
                    cw, ch = _sample_crop_wh(W, H, scale, self.aspect_ratio)
                    mask = _in_box(coords_b, _propose_box((ax, ay), cw, ch, W, H))
                    if int(mask.sum()) >= self.min_active_pixels:
                        kidx = mask.nonzero(as_tuple=False).squeeze(1)
                        break
                if kidx is None:
                    mask = _in_box(coords_b, _fallback_centre_box(W, H, scale, self.aspect_ratio))
                    kidx = mask.nonzero(as_tuple=False).squeeze(1)
                    if kidx.numel() == 0:
                        kidx = torch.zeros(1, dtype=torch.long, device=device)
                crop_idx[k].append(kidx)

        crops: list[CropResult] = []
        for k in range(self._n_crops):
            parts_c, parts_f = [], []
            for b in range(B):
                s = int(voxels.offsets[b])
                kidx = crop_idx[k][b]
                parts_c.append(voxels.coordinate_tensor[s + kidx])
                parts_f.append(voxels.feature_tensor[s + kidx])
            counts = torch.tensor([c.shape[0] for c in parts_c], dtype=torch.int64)
            offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
            new_coords = torch.cat(parts_c, dim=0)
            new_feats = torch.cat(parts_f, dim=0)
            crops.append(
                CropResult(
                    voxels=Voxels(
                        batched_coordinates=IntCoords(new_coords, offsets=offsets),
                        batched_features=CatFeatures(new_feats, offsets=offsets),
                        offsets=offsets,
                    ),
                    index=crop_idx[k],
                    is_global=k < self.n_global,
                )
            )
        return crops
