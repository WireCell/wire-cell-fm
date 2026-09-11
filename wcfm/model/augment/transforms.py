"""The charge normalisation, applied in the loop.

`FeatureLogTransform` runs inside `SslModule.training_step`, after the loader and before the
masker, in place. Two things depend on that order: the masker's `masked_feats` charge target
is taken from the normalised features, and `charge_loss` assumes its inputs are already in
log space. Moving this into the dataset or the collate -- the tidier-looking choice --
silently changes the charge target. `tests/test_model_ssl.py` asserts the order by checking
that the masker's target equals the transformed value at the masked coordinates.
"""

from __future__ import annotations

import math

from torch import Tensor
from warpconvnet.geometry.types.voxels import Voxels


class FeatureLogTransform:
    """Raw charge [ADC] to roughly `[-1, +1]` by log10 compression, in place.

        y = 2 * (log10(x + min_val) - log10(min_val))
              / (log10(max_val + min_val) - log10(min_val)) - 1

    `x = 0` maps to `-1`, and `x = max_val` maps to `+1` with no clipping above. The two
    constants are percentiles of a production's charge distribution and live on `data`.
    `enabled=False` makes this a no-op that still exists, so the config can say "no
    normalisation" without the module growing an `if`.
    """

    def __init__(self, min_val: float, max_val: float, enabled: bool = True) -> None:
        self.min_val = float(min_val)
        self.max_val = float(max_val)
        self.enabled = bool(enabled)
        y0 = math.log10(self.min_val)
        y1 = math.log10(self.max_val + self.min_val)
        self._y0 = y0
        self._scale = 2.0 / (y1 - y0)

    def value(self, x: Tensor) -> Tensor:
        """The transform as a pure function, for tests and for anything that must not mutate."""
        if not self.enabled:
            return x
        return (x + self.min_val).log10().sub(self._y0).mul(self._scale).add(-1.0)

    def __call__(self, xs: Voxels) -> Voxels:
        if self.enabled:
            feats = xs.feature_tensor
            feats.add_(self.min_val).log10_().sub_(self._y0).mul_(self._scale).add_(-1.0)
        return xs

    def __repr__(self) -> str:
        return (
            f"FeatureLogTransform(min_val={self.min_val:g}, max_val={self.max_val:g}, "
            f"enabled={self.enabled})"
        )
