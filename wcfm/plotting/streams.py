"""Reading `metrics/*.jsonl` back as plottable series.

A step record carries only what was collected at that step. The core scalars land every step,
`gradnorm/*` every 100 and `spectrum/*` every 500, so a stream of 36000 records holds a
`gradnorm/dec2/grad_norm` on 360 of them. Every series is therefore built as paired
`(x, y)` filtered on the key being present and finite -- indexing a record by a key it does not
carry is the failure mode this module exists to remove, and a series read as a bare list would
also misalign the x axis by the cadence ratio.

A restart can leave a real gap in a stream, where the records between the last sync and the
restored checkpoint were never written back. A gap is data, not a defect, so nothing here
interpolates across one.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["families", "keys_present", "read_records", "series", "smooth"]


def read_records(run_dir: Path | str, stream: str = "step") -> list[dict]:
    """Records from one of a run's streams, in order; empty when the stream is absent."""
    from wcfm.metrics.writer import read_stream

    return read_stream(Path(run_dir) / "metrics" / f"{stream}.jsonl")


def series(records: list[dict], key: str, x: str = "step") -> tuple[list[float], list[float]]:
    """`(xs, ys)` over the records that carry both `key` and `x` as finite scalars.

    Booleans are excluded: `preempted` is an int to `isinstance` and a flag to a reader.
    """
    xs: list[float] = []
    ys: list[float] = []
    for rec in records:
        xv, yv = rec.get(x), rec.get(key)
        if not _scalar(xv) or not _scalar(yv):
            continue
        xs.append(float(xv))
        ys.append(float(yv))
    return xs, ys


def _scalar(v) -> bool:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    return v == v and abs(v) != float("inf")  # NaN and +-inf would break an axis range


def keys_present(records: list[dict]) -> set[str]:
    """Every key any record carries as a finite scalar."""
    found: set[str] = set()
    for rec in records:
        for k, v in rec.items():
            if _scalar(v):
                found.add(k)
    return found


def families(present: set[str], prefix: str, depth: int = 1) -> list[str]:
    """Keys under `prefix`, grouped to `depth` path segments past it.

    `families(present, "gradnorm/")` gives the parameter groups the run recorded rather than a
    hardcoded list, so a run with a head this module has never heard of still plots.
    """
    out: set[str] = set()
    for key in present:
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].split("/")
        if len(parts) >= depth:
            out.add("/".join(parts[:depth]))
    return sorted(out)


def smooth(ys: list[float], window: int) -> list[float]:
    """Centred rolling mean, shrinking at the ends so the result is as long as the input.

    Step traces are noisy at batch scale and the trend is what a training curve is read for;
    the raw trace stays on the same axes underneath.
    """
    if window <= 1 or len(ys) < 2:
        return list(ys)
    half = window // 2
    out: list[float] = []
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out
