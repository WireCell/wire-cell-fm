"""Augmentation: the charge normalisation, crops, masks, and the `ViewPlan` they produce."""

from .cropping import Cropper, CropResult
from .masking import (
    BlockMasker,
    Masker,
    MaskResult,
    PixelMasker,
    RegionMasker,
    cap_negatives,
    label_candidates,
)
from .transforms import FeatureLogTransform
from .views import Augment, View, ViewPlan, select_meta

__all__ = [
    "Augment",
    "BlockMasker",
    "CropResult",
    "Cropper",
    "FeatureLogTransform",
    "MaskResult",
    "Masker",
    "PixelMasker",
    "RegionMasker",
    "View",
    "ViewPlan",
    "cap_negatives",
    "label_candidates",
    "select_meta",
]
