"""Figure setup shared by every panel: the backend, the per-run colours, the roles.

`pyplot` is imported through `pyplot()` and never at module scope. The import has to follow
`matplotlib.use("Agg")`, because a job and a login shell both run headless and the interactive
backends fail on import there rather than at draw time.

Three roles carry the meaning of a probe curve and must stay visually distinct: `feat` is the
representation being measured, `raw` the same head on raw charge, and `chance` the floor. A
`feat` curve read without the other two says nothing -- a macro-F1 of 0.39 is strong on 6
balanced classes and weak on 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["KEY_DASHES", "ROLE", "ROLE_LABEL", "figure", "pyplot", "run_colors", "save"]

#: Line style per role. `feat` takes the run's own colour; `raw` and `chance` are drawn in the
#: run's colour too, so a two-run overlay stays readable, but never as a solid line.
ROLE: dict[str, dict[str, Any]] = {
    "feat": {"linestyle": "-", "linewidth": 1.8, "marker": "o", "markersize": 3.5},
    "raw": {"linestyle": "--", "linewidth": 1.2, "marker": "s", "markersize": 2.5, "alpha": 0.75},
    "chance": {"linestyle": ":", "linewidth": 1.2, "marker": "", "alpha": 0.6},
}

#: What a role is called in a legend. `raw` is the same head fitted on raw charge, which is the
#: only thing that says whether the representation did any work.
ROLE_LABEL = {"feat": "features", "raw": "raw charge", "chance": "chance"}

#: Dash patterns for several series of one run inside one panel -- the two halves of a split
#: loss, say. Colour is spent on the run, so it cannot also separate these.
KEY_DASHES = [(), (5, 2), (1, 1.5), (6, 2, 1, 2)]

_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#e377c2",
]


def pyplot():
    """`matplotlib.pyplot` with the Agg backend selected first."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def run_colors(runs: list[str]) -> dict[str, str]:
    """One colour per run, stable in the order given."""
    return {run: _PALETTE[i % len(_PALETTE)] for i, run in enumerate(runs)}


def figure(n_panels: int, *, ncols: int = 2, height: float = 3.2, width: float = 6.0):
    """A grid big enough for `n_panels`, returned as `(fig, axes)` with `axes` flat.

    Trailing empty cells are removed, so a spec whose panels were partly skipped does not draw
    a blank frame next to the ones that have data.
    """
    plt = pyplot()
    ncols = min(ncols, max(n_panels, 1))
    nrows = -(-max(n_panels, 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    flat = [ax for row in axes for ax in row]
    for ax in flat[n_panels:]:
        fig.delaxes(ax)
    return fig, flat[:n_panels]


def save(fig, out_dir: Path | str, name: str, fmt: str = "png", dpi: int = 130) -> Path:
    """Write `<out_dir>/<name>.<fmt>` and close the figure. Returns the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.{fmt}"
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    pyplot().close(fig)
    return path
