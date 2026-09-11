"""`load_backbone(path, source)`: load a trained backbone by branch, from inside `wcfm.model`.

The work is in `wcfm.eval.loading`, which does it without naming a model class, since
`wcfm/eval/` is a framework package and may not import this one. This module is the other
direction, which is allowed, and it exists so that a probe or a notebook already inside
`wcfm.model` gets one call and the vocabulary of branches with it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from wcfm.eval.loading import checkpoint_sha256, inference_sources, load_module

__all__ = ["checkpoint_sha256", "inference_sources", "load_backbone", "load_module"]


def load_backbone(path: Path | str, source: str = "student", map_location: Any = "cpu"):
    """A `(voxels, taps) -> FeatureBundle` callable for one branch, plus the checkpoint.

    `source` is `student` or `teacher`; a run trained with `model/teacher=none` refuses the
    latter loudly rather than handing back features from initialisation weights.

    One branch at a time is the convenience. The charge transform is applied in place per
    call, so scoring both branches over one dataset means `SslModule.inference_step`, which
    is what `wcfm eval extract` uses.
    """
    module, ckpt = load_module(path, map_location=map_location)
    from wcfm.eval.loading import inference_step

    def run(voxels, taps=()):
        return inference_step(module, voxels, (source,), taps)[source]

    # Fail on an impossible `source` here, at load time, rather than at the first batch.
    if source not in inference_sources(module):
        raise ValueError(
            f"this run has no {source!r} branch; it has {list(inference_sources(module))}"
        )
    return run, ckpt
