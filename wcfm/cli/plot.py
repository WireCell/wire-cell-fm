"""`wcfm plot`: figures over a run's metrics streams and probe JSONs.

Several run directories on one command line overlay, one colour per run, which is how two
objectives are compared. Nothing is recomputed: a figure is a view over files the run and the
probe suite already wrote, so it is safe to regenerate at any time and safe to delete.

`wcfm.plotting` is imported inside the handler, not at module scope. It reaches matplotlib,
which is in the `analysis` extra, and `wcfm plot --help` has to work where there is none.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["main"]

USAGE = """usage: wcfm plot <run_dir> [run_dir...] [options]

Draws into <run_dir>/plots for one run, or --out-dir for several.

options:
  --what=diagnostics,probes   which suites to draw   (default: both)
  --figures=loss,spectrum     restrict the diagnostic figures
                              (loss, collapse, schedules, throughput, gradnorm,
                              spectrum)
  --out-dir=P                 where the figures go   (default: <run_dir>/plots; required
                              when more than one run is given)
  --sources=student,teacher   probe branches to draw (default: student; `all` for every one)
  --smooth=N                  rolling-mean window on the step traces  (default: 200)
  --format=png|pdf|svg        (default: png)

`run_dir` is the run's own directory -- the one holding metrics/, probes/ and checkpoints/.
A directory named `probes` is also accepted, for results collected outside a run.
"""


def _flags(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    flags, rest = {}, []
    for a in argv:
        if a.startswith("--"):
            key, _, value = a[2:].partition("=")
            flags[key] = value if value != "" else "true"
        else:
            rest.append(a)
    return flags, rest


def main(argv: list[str]) -> int:
    flags, rest = _flags(argv)
    if not rest or "help" in flags or "h" in flags:
        print(USAGE)
        return 0 if rest or "help" in flags or "h" in flags else 2

    run_dirs = [Path(r) for r in rest]
    missing = [str(d) for d in run_dirs if not d.is_dir()]
    if missing:
        print(f"wcfm plot: not a directory: {', '.join(missing)}", file=sys.stderr)
        return 2

    if "out-dir" in flags:
        out_dir = Path(flags["out-dir"])
    elif len(run_dirs) == 1:
        out_dir = run_dirs[0] / "plots"
    else:
        print(
            "wcfm plot: --out-dir is required with more than one run, because a comparison "
            "figure does not belong to either run's directory",
            file=sys.stderr,
        )
        return 2

    what = {w for w in flags.get("what", "diagnostics,probes").split(",") if w}
    unknown = what - {"diagnostics", "probes"}
    if unknown:
        print(f"wcfm plot: unknown --what {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    fmt = flags.get("format", "png")
    smooth = int(flags.get("smooth", 200))
    src = flags.get("sources", "student")
    sources = () if src == "all" else tuple(s for s in src.split(",") if s)

    try:
        from wcfm.plotting import plot_diagnostics, plot_probes
        from wcfm.plotting.diagnostics import DIAGNOSTIC_FIGURES
    except ImportError as exc:
        print(
            f"wcfm plot: {exc}. Plotting needs the `analysis` extra (matplotlib); the framework "
            "itself does not depend on it.",
            file=sys.stderr,
        )
        return 2

    figures = tuple(f for f in flags.get("figures", "").split(",") if f) or DIAGNOSTIC_FIGURES
    unknown_figures = set(figures) - set(DIAGNOSTIC_FIGURES)
    if unknown_figures:
        print(
            f"wcfm plot: unknown --figures {', '.join(sorted(unknown_figures))}; "
            f"there are {', '.join(DIAGNOSTIC_FIGURES)}",
            file=sys.stderr,
        )
        return 2

    written: list[Path] = []
    if "diagnostics" in what:
        written += plot_diagnostics(run_dirs, out_dir, smooth_window=smooth, fmt=fmt, only=figures)
    if "probes" in what:
        written += plot_probes(run_dirs, out_dir, sources=sources, fmt=fmt)

    if not written:
        print("wcfm plot: nothing to draw", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0
