"""Checkpoint v2: one file, a flat schema, and `model` opaque to the framework.

`{schema_version, epoch, step, cfg, model, optimizer, rng, meta}` at
`checkpoints/checkpoint_epoch{N}.pt`, plus `latest.pt` for resuming a training.

Resume needs RNG state per rank, the loader position, the schedule step, and the module's
own `state_dict`, which carries whatever the model keeps outside its parameters. The
framework never looks inside that blob.

- The write is atomic: `torch.save` to a temporary path in the same directory, then
  `os.replace`. The preemption handler writes while a kill is in progress, and a truncated
  `latest.pt` would break the next `--resume auto`.
- RNG state is keyed per rank, and holds rank 0's alone, since only rank 0 writes. The
  keying stops a rank restoring another rank's stream, which would silently correlate their
  augmentation after a resume. A rank with no entry of its own reseeds from
  `seed + rank + 1000 * resume_epoch` (`Trainer._load`), reproducible from the config and
  distinct per rank. Gathering every rank's state into the file is left open.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 2

__all__ = [
    "SCHEMA_VERSION",
    "Checkpoint",
    "load_checkpoint",
    "resolve_resume",
    "rng_state",
    "save_checkpoint",
    "set_rng_state",
    "should_save",
]


@dataclass
class Checkpoint:
    """What a checkpoint holds. `model` and `optimizer` are opaque state dicts."""

    epoch: int
    step: int
    cfg: dict
    model: dict
    optimizer: dict
    rng: dict
    meta: dict = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "epoch": self.epoch,
            "step": self.step,
            "cfg": self.cfg,
            "model": self.model,
            "optimizer": self.optimizer,
            "rng": self.rng,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, blob: dict) -> Checkpoint:
        version = int(blob.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint schema_version={version}, expected {SCHEMA_VERSION}. A v0 file is "
                "a legacy torch.save from ml-dune-model, which this framework does not read: "
                "it pickles classes from the old `dino` package by reference, so loading one "
                "needs that repo importable. Convert it there."
            )
        return cls(
            epoch=int(blob["epoch"]),
            step=int(blob["step"]),
            cfg=blob.get("cfg", {}),
            model=blob["model"],
            optimizer=blob.get("optimizer", {}),
            rng=blob.get("rng", {}),
            meta=blob.get("meta", {}),
            schema_version=version,
        )


def rng_state(rank: int = 0) -> dict:
    """This rank's RNG state, keyed by rank so a resume restores each rank's own stream."""
    import random

    import numpy as np

    state: dict[str, Any] = {
        "rank": int(rank),
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state: dict | None) -> bool:
    """Restore RNG state saved by `rng_state`. Returns whether anything was restored.

    A checkpoint carrying no state for this rank leaves the streams where they are. The
    resume is degraded, and the caller records that it happened.
    """
    if not state:
        return False
    import random

    import numpy as np

    torch.set_rng_state(_as_byte_tensor(state["torch"]))
    random.setstate(_tuple_deep(state["python"]))
    np.random.set_state(_tuple_deep(state["numpy"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(_as_byte_tensor(state["cuda"]))
    return True


def _as_byte_tensor(x: Any) -> torch.Tensor:
    """`torch.set_rng_state` insists on a ByteTensor; a round trip through some
    serializers hands back a list or a tensor on the wrong device."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", torch.uint8)
    return torch.tensor(x, dtype=torch.uint8)


def _tuple_deep(x: Any) -> Any:
    """`random.setstate` and `np.random.set_state` both require tuples, and a YAML or
    JSON round trip turns the nested tuples into lists."""
    if isinstance(x, list | tuple):
        return tuple(_tuple_deep(i) for i in x)
    return x


def save_checkpoint(path: Path | str, ckpt: Checkpoint) -> Path:
    """Write atomically: temp file in the same directory, then `os.replace`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        torch.save(ckpt.to_dict(), tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def load_checkpoint(path: Path | str, map_location: Any = "cpu") -> Checkpoint:
    """Read a v2 checkpoint.

    `weights_only=False` is required: `cfg` and `meta` are plain Python containers and the
    module's blob may hold whatever it declared. The file was written by this project into a
    directory it owns.
    """
    blob = torch.load(Path(path), map_location=map_location, weights_only=False)
    return Checkpoint.from_dict(blob)


def resolve_resume(resume: str, checkpoint_dir: Path | str) -> Path | None:
    """Turn `run.resume` into a path, or `None` for a fresh run.

    `"auto"` picks up `latest.pt`, failing that the highest `checkpoint_epoch{N}.pt`, so a
    run evicted before its handler wrote still resumes from the last periodic checkpoint.
    `"none"` refuses to resume with a checkpoint present, which makes a from-scratch re-run
    a config change. Any other value is a path, and a missing one raises: a typo would
    otherwise start a 40-hour run from epoch 0 in silence.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if resume == "none":
        return None
    if resume == "auto":
        latest = checkpoint_dir / "latest.pt"
        if latest.exists():
            return latest
        by_epoch = sorted(
            checkpoint_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0),
        )
        return by_epoch[-1] if by_epoch else None
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"run.resume={resume!r} does not exist")
    return path


def should_save(epoch: int, total_epochs: int, save_every: int, save_at: list[int]) -> bool:
    """Cadence is an interval plus a list of epochs.

    `save_at: [10, 50, 100]` alongside `save_every` checkpoints every sweep point at the
    same epochs whatever its own `epochs` value, so two points are comparable at matching
    epochs. The final epoch always saves.
    """
    if epoch == total_epochs:
        return True
    if epoch in (int(e) for e in save_at or ()):
        return True
    return bool(save_every) and epoch % int(save_every) == 0
