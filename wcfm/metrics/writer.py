"""Append-only JSONL: `metrics/step.jsonl` and `metrics/epoch.jsonl`, plus
`metrics/schema.json`.

One flat record per line, `O_APPEND`, flushed per record, so a kill costs at most the last
line. A stream's file appears on its first record, so a stream nothing writes to has no file
at all (`_open`).

Absent, never null. A key that was not measured is not in the record, and `schema.json`
carries the index of names actually written, so a reader can tell "never measured in this
run" from "measured, and it was zero". A sentinel value in the column cannot carry that
difference.

Resume truncates. A run that resumes from epoch N re-writes the steps from N onward, so on
open the writer drops a trailing partial line and every record at or beyond the resume step.
Without that a resumed run's stream holds two records for those steps and a plot silently
averages the pre-eviction and post-eviction values.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["MetricsWriter", "STREAMS", "read_stream"]

STREAMS = ("step", "epoch")
SCHEMA_VERSION = 1


class MetricsWriter:
    """Rank-0 only. Other ranks get an instance whose writes are no-ops, so no call site in
    the engine branches on rank."""

    def __init__(
        self,
        metrics_dir: Path | str,
        *,
        enabled: bool = True,
        max_record_bytes: int = 4096,
        resume_step: int | None = None,
    ):
        self.dir = Path(metrics_dir)
        self.enabled = bool(enabled)
        self.max_record_bytes = int(max_record_bytes)
        self.names: dict[str, set[str]] = {s: set() for s in STREAMS}
        self._handles: dict[str, Any] = {}
        self._over_budget = 0
        if self.enabled:
            self.dir.mkdir(parents=True, exist_ok=True)
            # Truncation is eager and opening is lazy. Truncating rewrites a file and needs
            # no handle, and doing it here rather than on a stream's first write means a
            # resumed run that dies before recording anything still leaves no records from
            # beyond the resume point. Otherwise a reader polling the directory in that
            # window sees the pre-eviction tail as if it were current.
            if resume_step is not None:
                for stream in STREAMS:
                    self._truncate(self.dir / f"{stream}.jsonl", resume_step)

    def write(self, stream: str, record: dict) -> None:
        if not self.enabled:
            return
        if stream not in STREAMS:
            raise ValueError(f"unknown metrics stream {stream!r}; expected one of {STREAMS}")
        # Two passes, and both are needed: the first drops keys the caller passed as `None`,
        # the second drops what `_jsonable` turns into `None` -- a NaN, an inf, or an array
        # too large to inline.
        clean = {k: _jsonable(v) for k, v in record.items() if v is not None}
        clean = {k: v for k, v in clean.items() if v is not None}
        self.names[stream].update(clean)
        line = json.dumps(clean, separators=(",", ":"))
        if len(line) > self.max_record_bytes:
            # An over-budget record is written whole: truncating it would drop whichever keys
            # happened to sort last. The count lands in `schema.json`, where it says the
            # budget is set too low.
            self._over_budget += 1
        handle = self._open(stream)
        handle.write(line + "\n")
        handle.flush()

    def _open(self, stream: str) -> Any:
        """Open a stream's file on its first record.

        Lazily, so a stream nothing writes to has no file. An empty file cannot be told apart
        from a run that has not reached its first record yet: a run killed inside its first
        epoch leaves no `epoch.jsonl` at all, which a reader polling the directory can read as
        "nothing yet", where an empty file says "measured nothing".
        """
        if (handle := self._handles.get(stream)) is not None:
            return handle
        handle = open(self.dir / f"{stream}.jsonl", "a")
        self._handles[stream] = handle
        return handle

    def write_schema(self, extra: dict | None = None) -> None:
        if not self.enabled:
            return
        blob = {
            "schema_version": SCHEMA_VERSION,
            # Every stream is listed, written or not, so a reader can tell an absent column
            # from an absent stream: `{}` for one that produced no records at all.
            "streams": {s: sorted(self.names[s]) for s in STREAMS},
            "records_over_budget": self._over_budget,
            "max_record_bytes": self.max_record_bytes,
            **(extra or {}),
        }
        tmp = self.dir / "schema.json.tmp"
        tmp.write_text(json.dumps(blob, indent=2) + "\n")
        tmp.replace(self.dir / "schema.json")

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.enabled:
            self.write_schema()
        self.close()

    @staticmethod
    def _truncate(path: Path, resume_step: int) -> None:
        """Drop a trailing partial line and every record at or beyond `resume_step`."""
        if not path.exists():
            return
        kept: list[str] = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # the partial last line of a killed run
            if int(rec.get("step", -1)) >= resume_step:
                continue
            kept.append(line)
        path.write_text("".join(line + "\n" for line in kept))


def read_stream(path: Path | str) -> list[dict]:
    """Records from a stream, in order, skipping a trailing partial line.

    A killed run leaves half a line. The writer truncates on resume, but a reader can be
    pointed at a stream nobody has reopened, so it tolerates one here too.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _jsonable(value: Any) -> Any:
    """Plain JSON, or `None` for something that should be absent rather than null.

    NaN and inf are dropped: `json.dumps` writes them as bare `NaN`/`Infinity`, which is valid
    Python and invalid JSON, so one diverged step would make the whole stream unreadable by
    every other tool. A NaN loss is reported by the engine's skip counter.
    """
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return None if value.size > 64 else [_jsonable(v) for v in value.ravel().tolist()]
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)
