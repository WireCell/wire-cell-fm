"""The raw-charge input: `[channel, tick, log_charge]` per pixel.

This is the number every probe's features are measured against. It depends on the extraction and
on the charge transform the run trained with, and never on the checkpoint's weights, which is
what makes it a baseline rather than a second model and what lets it double as a self-check: two
checkpoints extracted over the same events must score identically from it.

Split out of the probe suite for the same reason `taxonomy` and `geometry` were: extraction
needs it too. `probe_event` scores one pooled vector per event, and under `rows="pooled"` the
pixels it pools over are not written -- so extraction pools the raw baseline at the same time it
pools the features, while it still holds every row. A probe cannot reconstruct that afterwards
from a subset without silently pooling a different population.
"""

from __future__ import annotations

import numpy as np

__all__ = ["log_charge", "raw_charge_from", "raw_charge_kind_of"]


def log_charge(
    charge: np.ndarray, min_val: float | None = None, max_val: float | None = None
) -> np.ndarray:
    """Compress raw ADC charge to ~[-1, 1].

    With `min_val` and `max_val` this is `FeatureLogTransform`, exactly what the backbone was
    fed during training, so the charge column is the model's real input. Without them it is a
    parameter-free `log10(1 + q)`: monotone in charge and adequate, since the heads standardize
    their inputs anyway, but not comparable with a run scored through the trained transform.
    """
    q = np.clip(np.asarray(charge, dtype=np.float64), 0.0, None)
    if min_val is None or max_val is None:
        return np.log10(1.0 + q)
    y0 = np.log10(min_val)
    y1 = np.log10(max_val + min_val)
    return 2.0 * (np.log10(q + min_val) - y0) / (y1 - y0) - 1.0


def raw_charge_kind_of(params: dict | None) -> str:
    """`trained` or `log10_1p`, from the recorded transform parameters.

    Recorded with every result and checked by `merge`, because the two are different inputs: a
    `trained` delta is not comparable with a `log10_1p` one even for the same checkpoint.
    """
    params = params or {}
    return "trained" if params.get("kind") == "log" and "min_val" in params else "log10_1p"


def raw_charge_from(
    positions: np.ndarray, charges: np.ndarray, params: dict | None = None
) -> np.ndarray:
    """`[N, 3]` of `[channel, tick, log_charge]`, float32."""
    params = params or {}
    if raw_charge_kind_of(params) == "trained":
        lq = log_charge(charges, float(params["min_val"]), float(params["max_val"]))
    else:
        lq = log_charge(charges)
    positions = np.asarray(positions)
    return np.stack(
        [
            positions[:, 0].astype(np.float64),
            positions[:, 1].astype(np.float64),
            lq,
        ],
        axis=1,
    ).astype(np.float32)
