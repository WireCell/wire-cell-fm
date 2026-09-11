"""Running the collectors, and writing what they produce.

`Trainer` calls this twice per epoch: `on_step` after an optimizer step, `on_epoch` when the
epoch ends. Everything between -- selecting what fires, building the `StepRecord`, reducing
across ranks, writing the row -- happens here, so the loop holds nothing about metrics.

Two rules the collective sequence depends on, both of which hold by construction because
`cadence`, `needs` and `reduce` are class attributes on `Collector`:

- Every rank selects the same collectors on the same step. `fires` reads the step and the
  cadence, and nothing rank-local.
- Every rank reduces the same keys in the same order. The reduction covers what a collector
  declared, whatever it returned on this rank.

No branch on rank-local information may sit in front of a reduction. A collector whose needs
are unmet produces an empty result and still takes its turn.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from wcfm.engine.protocol import StepOutput
from wcfm.metrics.base import StepRecord, fires

log = logging.getLogger(__name__)

__all__ = ["Collection", "build_collectors"]


def build_collectors(metrics_cfg: Any) -> dict[str, Any]:
    """Whatever is in `metrics.collectors`, and nothing if it is empty.

    The engine registers them and knows none of their names. An empty dict is a valid
    configuration: the engine records lr and weight decay itself, since it already holds
    them and they are a diverging run's first question.
    """
    from hydra.utils import instantiate

    entries = OmegaConf.to_container(metrics_cfg.collectors, resolve=True) or {}
    return {str(name): instantiate(entry) for name, entry in entries.items()}  # type: ignore[union-attr]


class Collection:
    """The collectors of one run, and the machinery that fires them."""

    def __init__(
        self,
        collectors: dict[str, Any],
        *,
        fabric: Any,
        writer: Any,
        module: Any,
        run_dir: Path,
        step_cadence: int,
    ):
        self.collectors = collectors
        self.fabric = fabric
        self.writer = writer
        self.module = module
        self.run_dir = Path(run_dir)
        self.step_cadence = int(step_cadence)
        declared = getattr(module, "grad_taxonomy", None)
        self.taxonomy: dict[str, tuple[str, ...]] | None = declared() if declared else None
        # The last completed step's raw material, for the end-of-epoch firing: an
        # epoch-cadence collector summarises the epoch it just finished, and this is the
        # freshest observation in it. `None` until a step completes, which is also what makes
        # an epoch with no steps produce no epoch-cadence columns rather than stale ones.
        self._last_step_material: tuple[int, StepOutput, dict, dict] | None = None

    @property
    def world_size(self) -> int:
        return int(getattr(self.fabric, "world_size", 1))

    # ------------------------------------------------------------------ the two entry points

    def begin_epoch(self) -> None:
        """Forget the previous epoch's last step, so an empty epoch produces no columns."""
        self._last_step_material = None

    def wants_term_grads(self, step: int) -> bool:
        """Does a collector firing on this step need per-term gradients?

        Asked of `needs`, a class attribute, so every rank answers identically. The module
        all-reduces the vectors it produces, so a rank that decided differently would leave
        the others in a collective forever.

        The request exists because the measurement is expensive, one `autograd.grad` per term
        per view, so it is made only on a firing step.
        """
        return any("term_grads" in c.needs for _n, c in self._due_on_step(step))

    def on_step(
        self,
        epoch: int,
        step: int,
        out: StepOutput,
        applied: dict[str, float],
        timing: dict[str, float],
        grad_norm: float | None = None,
    ) -> None:
        """Build the record, dispatch, write one row. Reductions happen here, never in
        `compute()`."""
        self._last_step_material = (step, out, applied, timing)
        due = self._due_on_step(step)
        on_stream = self.step_cadence > 0 and step % self.step_cadence == 0
        if not due and not on_stream:
            return

        record = self._record(
            epoch,
            step,
            out,
            applied,
            timing,
            with_observables=bool(due),
            with_term_grads=any("term_grads" in c.needs for _n, c in due),
        )

        # Every scheduled value is a column on every logged step, written by the engine
        # because the engine is what holds them.
        row: dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            **{k: v for k, v in record.scalars.items()},
            **applied,
        }
        if grad_norm is not None:
            row["grad_norm"] = grad_norm
        row.update(self._dispatch(due, record))

        self.writer.write("step", row)

    def on_epoch(self, epoch: int) -> dict:
        """The end-of-epoch firing, for collectors whose cadence is `"epoch"`.

        The record is the last step's material: its observables and timing are the freshest
        the epoch has, and an epoch-cadence collector is summarising the epoch it just
        finished. Anything that needs more than the last step accumulates it in the collector,
        which is why `compute` is handed a record rather than a window.
        """
        due = [(n, c) for n, c in self.collectors.items() if c.cadence == "epoch"]
        if not due:
            return {}
        # Whether this rank completed a step is rank-local and `_dispatch` below reduces,
        # so a rank returning early here would hang the others in `all_reduce`. One scalar
        # reduce per epoch buys the guarantee.
        if not self._all_ranks_have_material():
            return {}
        assert self._last_step_material is not None
        step, out, applied, timing = self._last_step_material
        record = self._record(epoch, step, out, applied, timing, with_observables=True)
        return self._dispatch(due, record)

    # ------------------------------------------------------------------ selection

    def _due_on_step(self, step: int) -> list[tuple[str, Any]]:
        """`(name, collector)` for everything firing on this step, in registration order.

        `end_of_epoch=False`, so an `"epoch"`-cadence collector is excluded here. `on_epoch`
        selects on the cadence itself: `fires` answers `True` for a `"step"` cadence and for
        an integer cadence the step divides, whatever `end_of_epoch` says, so asking it at
        epoch end would fire those a second time and write a per-step quantity into the
        epoch record.
        """
        return [(name, c) for name, c in self.collectors.items() if fires(c, step)]

    def _all_ranks_have_material(self) -> bool:
        """Did every rank complete a step this epoch? Collective when it has to be."""
        local = 1.0 if self._last_step_material is not None else 0.0
        if self.world_size == 1:
            return local > 0.0
        flag = self.fabric.all_reduce(
            torch.tensor([local], device=self.fabric.device), reduce_op="min"
        )
        return bool(float(flag.item()) > 0.0)

    # ------------------------------------------------------------------ record and dispatch

    def _record(
        self,
        epoch: int,
        step: int,
        out: StepOutput,
        applied: dict[str, float],
        timing: dict[str, float],
        *,
        with_observables: bool,
        with_term_grads: bool = False,
    ) -> StepRecord:
        return StepRecord(
            step=step,
            epoch=epoch,
            scalars={**out.scalars, "n_samples": out.n_samples},
            observables=self.module.observables() if with_observables else {},
            named_parameters=self.module.named_parameters,  # UNWRAPPED
            timing=timing,
            # Optional hook, found by getattr. Without one, parameters are grouped by
            # the first token of their path.
            taxonomy=self.taxonomy,
            # Optional hook, found by getattr. A module that ignores the request produces
            # none, and the collector writes no columns.
            term_grads=(
                getattr(self.module, "last_term_gradients", None) if with_term_grads else None
            ),
            # Rank 0 only. A collector that writes its own files -- `ArrayDump` -- would
            # otherwise have every rank `np.save` to the same GPFS path under the same name,
            # concurrently, while only rank 0's pointer reached the stream. `None` is the
            # documented "no array directory available", and `ArrayDump` emits nothing for it
            # rather than guessing a path, so this needs no rank check inside the collector.
            arrays_dir=(self.run_dir / "metrics" / "arrays") if self.fabric.is_global_zero
            else None,
        )

    def _dispatch(self, due: list[tuple[str, Any]], record: StepRecord) -> dict[str, Any]:
        """Run the collectors that are firing and prefix their keys with their config names.

        Every rank arrives here with the same `due` list, because `fires` is a pure function
        of the step and the cadence. No branch in this loop may be rank-local.
        """
        row: dict[str, Any] = {}
        for name, collector in due:
            # Unmet needs make a collector produce nothing; they do not make it skip the
            # reduction. `unmet` reads observable keys, which come from
            # `module.observables()` and are rank-local, so skipping the collector here
            # would put a rank-local branch in front of `_reduce_declared`'s `all_reduce`.
            # An empty result keeps the collective sequence a function of `reduce` and the
            # cadence, identical on every rank, and this rank contributes the identity.
            if unmet := collector.unmet(record):
                log.debug("collector %s produced nothing: observables absent %s", name,
                          sorted(unmet))
                produced: dict[str, Any] = {}
            else:
                produced = collector.compute(record)
            for key, value in produced.items():
                row[f"{name}/{key}"] = value
            for key, reduced in self._reduce_declared(collector, produced).items():
                row[f"{name}/{key}"] = reduced
        return row

    def _reduce_declared(self, collector: Any, produced: dict[str, Any]) -> dict[str, float]:
        """The uniform engine-side pass, over the keys a collector declared.

        `reduce` is a class attribute, so every rank iterates the same key set in the same
        order and the collective sequence is identical on all of them. Reducing the returned
        keys instead would let a collector whose output depends on its input -- `Spectrum`
        emitting nothing for a batch with fewer than two rows, `ArrayDump` emitting nothing
        off rank 0 -- call `all_reduce` on some ranks and not others, and that hangs the job
        for good.

        A rank that did not produce a declared key contributes the reduction's identity: 0 for
        `sum` and `mean`, and 0 for `max` over quantities that are counts, rates and memory.
        The result is written only if this rank produced the key, so a column appears when the
        writing rank measured it.
        """
        declared: dict[str, str] = getattr(collector, "reduce", {}) or {}
        if not declared or self.world_size == 1:
            return {}
        out: dict[str, float] = {}
        for key in sorted(declared):  # sorted: the same order on every rank
            local = produced.get(key)
            value = float(local) if isinstance(local, int | float) else 0.0
            tensor = torch.tensor([value], device=self.fabric.device)
            reduced = float(self.fabric.all_reduce(tensor, reduce_op=declared[key]).item())
            if key in produced:
                out[key] = reduced
        return out
