"""The label vocabulary the probe suite scores against.

Split out of the probes themselves because the pools are drawn against it at extraction time,
and extraction may not import a probe: `wcfm/eval/pools.py` needs `PID_CLASSES` to balance over
and `probe_pid` needs the same tuple to name its columns, so a second copy in either place is a
way for the two to disagree without anything failing.

These are truth categories rather than model vocabulary -- they name what the simulation
deposited -- so they belong on the framework's evaluation side and `tests/test_import_graph.py`
is satisfied.

Every value here is fixed by the results already recorded against it. `PID_NAMES` is the order
the columns of every `pid_*.json` are in, `PID_CLASSES` is the order pools are concatenated in,
and `PID_HEADLINE` is which classes enter the macro average. Changing any of them makes a new
number incomparable with an old one, and nothing raises.
"""

from __future__ import annotations

#: Per-pixel particle categories, in the order `probe_pid` reports them.
PID_NAMES: tuple[str, ...] = (
    "Background",
    "Track",
    "Shower",
    "Michel",
    "DeltaRay",
    "Blip",
    "Other",
)

#: The class ids, parallel to :data:`PID_NAMES`. Class 0 is "Background / no truth".
PID_CLASSES: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

#: The classes that enter the macro average. Background is excluded: it is the *absence* of
#: truth, so scoring it rewards a head for recognising unlabelled pixels, and it dominates the
#: population. It is still pooled and still reported per class -- a probe that never sees it
#: cannot report the prevalence-weighted score -- just not averaged in.
PID_HEADLINE: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

#: `probe_knn_pid` scores only the truthed classes, so its class axis is `PID_NAMES` less
#: Background, and its class index `i` means `PID_CLASSES[i + 1]`.
PIXEL_CLASS_NAMES: tuple[str, ...] = PID_NAMES[1:]

#: Instance-size bins for `probe_instance`'s breakdown, as recorded.
SIZE_BIN_NAMES: tuple[str, ...] = ("1", "2-3", "4-9", "10-99", "100-999", "1000+")


def pid_name(cls: int) -> str:
    """The reported name of a class id."""
    return PID_NAMES[PID_CLASSES.index(cls)]
