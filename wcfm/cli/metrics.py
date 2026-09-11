"""`wcfm metrics`: read the stream back out.

`summary` answers what a run actually recorded and how much of it, from `schema.json` and the
streams themselves, without loading any of them into a notebook.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

__all__ = ["main"]

USAGE = """usage: wcfm metrics <subcommand> [options] <run_dir>

  wcfm metrics summary runs/my_run   what this run recorded, and how much

`run_dir` is the run's own directory -- the one holding metrics/, checkpoints/ and debug/.
"""


def _records(run_dir: Path, stream: str = "step") -> list[dict]:
    from wcfm.metrics.writer import read_stream

    return read_stream(run_dir / "metrics" / f"{stream}.jsonl")


def _summary(run_dir: Path, argv: list[str]) -> int:
    from wcfm.metrics.writer import STREAMS

    metrics_dir = run_dir / "metrics"
    schema_path = metrics_dir / "schema.json"
    if schema_path.exists():
        schema = json.loads(schema_path.read_text())
        print(f"schema v{schema.get('schema_version')} -- {schema_path}")
        for stream, names in (schema.get("streams") or {}).items():
            print(f"  {stream:8s} {len(names):4d} names")
        if over := schema.get("records_over_budget"):
            print(f"  {over} records over max_record_bytes={schema.get('max_record_bytes')}")
    else:
        print(f"no schema.json in {metrics_dir} (a run that never flushed one)")

    for stream in STREAMS:
        path = metrics_dir / f"{stream}.jsonl"
        if not path.exists():
            continue
        records = _records(run_dir, stream)
        size = path.stat().st_size
        first = records[0].get("step") if records else None
        last = records[-1].get("step") if records else None
        print(f"  {stream:8s} {len(records):6d} records  {size / 1024:8.1f} KiB  "
              f"steps {first}..{last}")

    arrays = metrics_dir / "arrays"
    if arrays.is_dir():
        files = sorted(arrays.glob("*.npy"))
        total = sum(f.stat().st_size for f in files)
        print(f"  arrays   {len(files):6d} .npy      {total / 1024**2:8.1f} MiB")
    return 0


SUBCOMMANDS = {"summary": _summary}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    sub, rest = argv[0], argv[1:]
    if sub not in SUBCOMMANDS:
        print(
            f"wcfm metrics: unknown subcommand {sub!r}; try {', '.join(SUBCOMMANDS)}",
            file=sys.stderr,
        )
        return 2

    positional = [a for a in rest if not a.startswith("-")]
    if len(positional) != 1:
        print(f"wcfm metrics {sub}: expected exactly one run directory", file=sys.stderr)
        return 2

    run_dir = Path(positional[0]).expanduser()
    if not run_dir.is_dir():
        print(f"wcfm metrics: no such run directory {run_dir}", file=sys.stderr)
        return 2
    return SUBCOMMANDS[sub](run_dir, rest)
