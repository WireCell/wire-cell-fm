"""SIGTERM to a checkpoint, without writing one from inside a signal handler.

An evicted job otherwise loses everything since the last periodic checkpoint. This closes
that to one step.

- The handler sets a flag and returns. It runs on whatever stack the signal interrupted,
  possibly mid-`torch.save` or inside a NCCL collective, so a checkpoint written from there
  can be truncated. The loop polls the flag where the model state is consistent, writes
  there, and exits.
- The decision to stop is collective. Whether a rank saw SIGTERM is rank-local, and
  branching on it hangs the job: one rank leaves the loop while the others wait in the next
  reduce. `should_stop` all-reduces the flag with MAX, so the whole group stops at the same
  step. One scalar reduce per step, alongside a gradient all-reduce already happening.

A second SIGTERM is not caught: the previous handler is restored and the signal kills the
process. The supervisor is escalating by then, and SIGKILL cannot be trapped.
"""

from __future__ import annotations

import logging
import signal
from collections.abc import Callable
from types import FrameType
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["PreemptionGuard"]


class PreemptionGuard:
    """Traps SIGTERM and SIGINT, exposing a flag the loop polls at a safe point.

    Usage is a context manager so the previous handlers are always restored -- which matters
    in the test suite, where an uninstalled handler leaks into every later test in the
    process, and under `fabric.launch()`, which re-invokes the script.
    """

    def __init__(self, signals: tuple[int, ...] = (signal.SIGTERM, signal.SIGINT)):
        self.signals = signals
        self.signalled = False
        self.signal_number: int | None = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> PreemptionGuard:
        for sig in self.signals:
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except ValueError:
                # Not the main thread: a DataLoader worker or a spawned rank helper. The
                # rank that owns the loop installs its own; this one simply has no handler.
                log.debug("could not install a handler for signal %s off the main thread", sig)
        return self

    def __exit__(self, *exc: object) -> None:
        for sig, previous in self._previous.items():
            try:
                signal.signal(sig, previous)
            except ValueError:
                pass
        self._previous.clear()

    def _handle(self, signum: int, frame: FrameType | None) -> None:
        if self.signalled:
            # Already stopping and the supervisor is escalating. Restore and re-raise so the
            # default disposition applies: a process that swallows repeated SIGTERMs is the
            # one that gets SIGKILLed with no checkpoint at all.
            previous = self._previous.get(signum, signal.SIG_DFL)
            signal.signal(signum, previous)
            signal.raise_signal(signum)
            return
        self.signalled = True
        self.signal_number = signum

    def should_stop(self, all_reduce: Callable[..., Any] | None = None) -> bool:
        """Whether the group should stop now. Collective when `all_reduce` is given.

        `all_reduce` is `StepContext`'s -- Fabric's, or a no-op on one device. Passing
        `None` makes the answer rank-local, which is correct only at `world_size == 1`.
        """
        if all_reduce is None:
            return self.signalled
        import torch

        flag = torch.tensor([1.0 if self.signalled else 0.0])
        reduced = all_reduce(flag, reduce_op="max")
        return bool(float(reduced.item()) > 0.0)
