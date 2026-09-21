"""Figures over what a run already wrote: the metrics streams and the probe JSONs.

Nothing here computes a number. Every value plotted is read from `metrics/{step,epoch}.jsonl`
or from `probes/*.json`, which are the durable results; a figure is a view that rebuilds in
seconds, the same relation `wcfm eval merge` has to the probe files.

One run or several: every entry point takes a list of run directories and draws one series per
run on shared axes, so a comparison is the same code path as a single run's diagnostics.

matplotlib is in the `analysis` extra, not in the framework's own dependencies, so it is
imported inside `wcfm.plotting.style` rather than at module scope. Importing this package
costs nothing in the config-only environment.
"""

from __future__ import annotations

from .diagnostics import plot_diagnostics
from .probes import plot_probes

__all__ = ["plot_diagnostics", "plot_probes"]
