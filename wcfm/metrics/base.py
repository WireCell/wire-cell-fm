"""The metrics protocol: what a collector is handed, and what it may not do.

The engine builds a `StepRecord` and hands it to the `Collector` objects named in
`metrics.collectors`. Three rules hold by construction rather than by care:

- Nothing inside `compute()` performs a collective. A collector declares `reduce` over its
  output keys and the engine reduces them uniformly. A cadence that is a pure function of the
  step is necessary and not sufficient: a body can still branch on data in front of a
  collective, which leaves the ranks in different places and hangs the job.
- Everything in `observables` is detached, and the record is built after backward. Holding a
  live handle pins the autograd graph.
- `named_parameters` is the unwrapped module's. `fabric.setup()` returns a wrapper whose
  parameter names are prefixed, so a gradient taxonomy written against unwrapped names
  matches nothing and reports zeros without raising. The engine keeps both handles and passes
  this one.

This module is the framework's half. `collectors.py` ships the five that know nothing about
any model: `Spectrum`, `GradNorm`, `Throughput`, `ArrayDump` and `TermGrad`. A model may ship
collectors of its own, named in config by `_target_` like any other component.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from torch import Tensor

__all__ = ["Cadence", "Collector", "StepRecord", "fires"]

Reduction = Literal["mean", "max", "sum"]
# Every collector takes a cadence from config, so this is a named type rather than three
# `# type: ignore`s on three `self.cadence = cadence` lines.
Cadence = int | Literal["step", "epoch"]


@dataclass
class StepRecord:
    """One step's worth of raw material. Built after backward, everything detached."""

    step: int
    epoch: int
    scalars: dict[str, float] = field(default_factory=dict)
    observables: dict[str, Tensor] = field(default_factory=dict)
    # The unwrapped module's `named_parameters`, left as a callable so the parameters are
    # walked only when a collector that declared "params" or "grads" fires.
    named_parameters: Callable[[], Iterator] = field(default_factory=lambda: iter(()))
    timing: dict[str, float] = field(default_factory=dict)
    # The module's gradient taxonomy, if it declared one: group name -> parameter-name
    # prefixes. The engine reads it through `getattr(module, "grad_taxonomy", None)`, so a
    # module that wants its gradients grouped does not have to widen `TrainingModule`.
    taxonomy: dict[str, tuple[str, ...]] | None = None
    term_grads: dict[str, Tensor] | None = None
    """Per-term gradient vectors over the shared parameters, already all-reduced.

    Present only on a step where a collector declaring `needs = {"term_grads"}` fires and the
    module honoured `ctx.extra["collect_term_gradients"]`. The module takes them with
    `torch.autograd.grad(..., retain_graph=True)` before the real backward, which leaves
    `.grad` untouched and fires no accumulator hook, so the vectors are rank-local until the
    module all-reduces them itself. They arrive here reduced, which is what lets a collector
    read them without performing a collective.
    """
    # Where a collector may write a `.npy` beside the stream. A collector cannot know the run
    # directory from config, so it is handed `metrics/arrays/` here. The engine fills this in
    # on the global-zero rank only, so a collector that writes arrays emits nothing on the
    # other ranks instead of racing them all onto one path. A collector handed `None` writes
    # nothing rather than guessing a path.
    arrays_dir: Path | None = None


class Collector(ABC):
    """A metric that fires on a cadence and declares how its outputs reduce across ranks.

    `cadence` is an integer number of steps, or `"step"` for every step, or `"epoch"` for once
    per epoch -- the default, and the right choice for a metric too expensive to run per step.
    An epoch-cadence collector fires from `Collection.on_epoch`, and its columns land in the
    `epoch` stream.

    `needs` names the observable keys the collector reads, plus the literals `"params"`,
    `"grads"` and `"term_grads"`. The engine skips a collector whose needs the record cannot
    satisfy rather than letting it raise mid-epoch.

    `"term_grads"` is also a request: it costs about one `autograd.grad` per term per view, so
    the engine asks the module for it only on a step where a collector declaring it fires.
    `needs` is a class attribute, so every rank asks on the same steps, which is what stops the
    collective the module performs from hanging the job.
    """

    cadence: Cadence = "epoch"
    needs: frozenset[str] = frozenset()
    reduce: dict[str, Reduction] = {}

    @abstractmethod
    def compute(self, rec: StepRecord) -> dict[str, float | np.ndarray]:
        """Pure and rank-local. Must not perform a collective."""

    def unmet(self, rec: StepRecord) -> set[str]:
        """Which declared needs this record cannot satisfy."""
        available = set(rec.observables)
        if rec.term_grads:
            available.add("term_grads")
        return {n for n in self.needs if n not in available and n not in ("params", "grads")}


def fires(collector: Collector, step: int, *, end_of_epoch: bool = False) -> bool:
    """Cadence as a pure function of the step -- no state, so two collectors on the same
    cadence always fire on the same steps and their columns line up in the stream."""
    cadence = collector.cadence
    if cadence == "epoch":
        return end_of_epoch
    if cadence == "step":
        return True
    cadence = int(cadence)
    return cadence > 0 and step % cadence == 0
