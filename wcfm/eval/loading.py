"""From a checkpoint file back to the module that wrote it, without naming what that module is.

`wcfm.engine.checkpoint` reads the file and returns `model` as an opaque state dict, which is
as far as the engine goes: it does not know what is in the blob. Extraction needs the blob back
in a `nn.Module` it can push voxels through, and this is where that happens -- still without
naming a model class, because `wcfm/eval/` is a framework package and
`tests/test_import_graph.py` forbids it importing `wcfm.model`.

The architecture comes from the checkpoint's own `cfg`, instantiated by `_target_` -- the same
move `wcfm/cli/train.py` makes to build a model it may not name. It also makes the architecture
that produced a set of weights a property of the file, so "which `encoding_range` was this
trained at" stops being something a caller can supply wrongly.

Loading is strict. A tolerant `load_state_dict` yields a backbone that is part trained and part
freshly initialised, which extracts features that look entirely normal and score like noise,
and nothing downstream can tell. It is refused here, and the error names the file.

Branches (`student`, `teacher`) are model vocabulary and do not appear in this module. The
module is asked for them through the optional `inference_step` hook, `getattr`-style, exactly
as the metrics package asks for `grad_taxonomy`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

__all__ = ["checkpoint_sha256", "inference_sources", "inference_step", "load_module"]


def checkpoint_sha256(path: Path | str, _chunk: int = 1 << 20) -> str:
    """SHA-256 of the checkpoint file, streamed.

    This is what `provenance.json` records and what the DAG's PRE script compares to decide
    whether an extraction is stale. Comparing modification times instead is defeated by a re-run
    against an unchanged checkpoint, and waiting for a file to stop growing by a slow GPFS
    write.
    """
    h = hashlib.sha256()
    with open(Path(path), "rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path: Path | str, map_location: Any = "cpu"):
    """Rebuild the module a checkpoint was written from, weights loaded, in `eval()`.

    Returns `(module, checkpoint)`, so the caller keeps the epoch, step and cfg that came with
    it rather than re-reading the file to get them.
    """
    from hydra.utils import instantiate

    from wcfm.engine.checkpoint import load_checkpoint

    ckpt = load_checkpoint(path, map_location=map_location)
    cfg_model = (ckpt.cfg or {}).get("model")
    if not cfg_model:
        raise ValueError(
            f"{path} carries no `cfg.model`, so the architecture that produced its weights is "
            "unknown and cannot be rebuilt. Every checkpoint `wcfm train` writes carries one."
        )
    module = instantiate(cfg_model)
    try:
        module.load_state_dict(ckpt.model, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"the weights in {path} do not match the model its own `cfg.model` describes: "
            f"{exc}\nThis is refused rather than loaded loosely: a partly-initialised backbone "
            "extracts features that look normal and score like noise."
        ) from exc
    module.eval()
    return module, ckpt


def inference_sources(module) -> tuple[str, ...]:
    """The branches this module can be extracted from, via the optional hook."""
    hook = getattr(module, "inference_sources", None)
    if hook is None:
        raise TypeError(
            f"{type(module).__name__} does not implement `inference_sources()`, so there is no "
            "defined set of branches to extract. The hook is optional on the training contract "
            "and required to be evaluated offline."
        )
    return tuple(hook())


def inference_step(module, voxels, sources, taps=()):
    """One clean image through every requested branch: `{source: FeatureBundle}`."""
    hook = getattr(module, "inference_step", None)
    if hook is None:
        raise TypeError(
            f"{type(module).__name__} does not implement "
            "`inference_step(voxels, sources, taps)`, so it cannot be extracted from."
        )
    return hook(voxels, sources, taps)
