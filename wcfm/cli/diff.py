"""`wcfm diff <a> <b>`: what two runs actually differ by.

The diff is the definition of an ablation. Done by eye across 60-key JSON files, two runs that
differ only in a derived quantity nobody printed look identical, and a campaign's record says
what somebody meant to vary rather than what varied.

Three views:

    wcfm diff run_a run_b              the resolved configs
    wcfm diff run_a run_b --code       the source trees the two runs executed
    wcfm diff run_a run_b --all        including run.name and the other by-construction keys

The config view is the one that matters and it is pure Python over the `config.yaml` every run
writes. `--code` shells out to `diff -ru`, because a source tree diff is what `diff` is for and
reimplementing it would buy nothing.

Absence is a difference. A key present in one config and missing from the other is reported as
`<absent>` rather than skipped: "this run had no `model.terms.occupancy` at all" is the most
interesting thing a diff can say, and treating a missing key as equal to any value hides exactly
the ablations worth seeing.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from wcfm.cli.jobpack import ARCHIVE_NAME

__all__ = ["main"]

USAGE = """usage: wcfm diff <run_a> <run_b> [options]

  --code        also diff the source trees the two runs executed (job/repo.tgz)
  --all         include keys excluded by default (run.name, run.output_root, hydra.*)
  --context=N   lines of context for --code (default: 3)

Each argument is a run directory -- the one holding config.yaml, checkpoints/ and metrics/.
"""

_ABSENT = "<absent>"


def _fmt(v) -> str:
    return _ABSENT if v is _MISSING else ("''" if v == "" else str(v))


_MISSING = object()


def _diff_configs(a: Path, b: Path, include_noise: bool) -> int:
    from wcfm.eval.config_diff import differing_keys, load_run_config

    ca, cb = load_run_config(a), load_run_config(b)
    keys = differing_keys({a.name: ca, b.name: cb}, include_noise=include_noise)

    print(f"a: {a}")
    print(f"b: {b}\n")
    if not keys:
        scope = (
            ""
            if include_noise
            else " (excluding run.name and the other by-construction keys; --all shows them)"
        )
        print(f"the two resolved configs are identical{scope}")
        return 0

    width = max(len(k) for k in keys)
    va = max(len(_fmt(ca.get(k, _MISSING))) for k in keys)
    print(f"{'key'.ljust(width)}  {'a'.ljust(va)}  b")
    print(f"{'-' * width}  {'-' * va}  -")
    for k in keys:
        left = _fmt(ca.get(k, _MISSING)).ljust(va)
        print(f"{k.ljust(width)}  {left}  {_fmt(cb.get(k, _MISSING))}")
    print(f"\n{len(keys)} differing key(s)")
    return 0


def _diff_code(a: Path, b: Path, context: int) -> int:
    """`diff -ru` over the two runs' code archives.

    What a run executed is not the same fact as the git sha it recorded: a run is routinely
    submitted from a dirty tree, and several have been. So this compares the trees.

    Each Condor run keeps `job/repo.tgz` -- the archive that was transferred to the worker,
    written by `wcfm/cli/jobpack.py` at submit time. Until 2026-09-10 this looked for a `code/`
    directory that nothing in this repository ever wrote, inherited from the old one: `--code`
    could not succeed on any wcfm run.
    """
    archives = [r / "job" / ARCHIVE_NAME for r in (a, b)]
    missing = [str(p) for p in archives if not p.is_file()]
    if missing:
        print(
            f"wcfm diff --code: no code archive at {', '.join(missing)}. A run keeps one only "
            "if it was submitted to Condor; for a local run the fallback is `git diff` between "
            "the two `run_metadata.json` shas -- but note a sha is a lower bound, since a run "
            "can be submitted from a dirty tree.",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        trees = []
        for label, archive in zip(("a", "b"), archives, strict=True):
            dest = Path(tmp) / label
            with tarfile.open(archive) as tar:
                # `filter="data"` refuses absolute paths and traversal; these archives are our
                # own, but extraction defaults change between Python versions and a diff is not
                # worth a surprise.
                tar.extractall(dest, filter="data")
            trees.append(dest)
        print(f"\n=== code: {archives[0]} vs {archives[1]} ===")
        # `diff` exits 1 when the trees differ, which is not an error here.
        completed = subprocess.run(
            ["diff", "-ru", f"-U{context}", str(trees[0]), str(trees[1])], check=False
        )
    return 0 if completed.returncode in (0, 1) else completed.returncode


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    flags = {a[2:].split("=")[0]: (a.split("=", 1)[1] if "=" in a else "true")
             for a in argv if a.startswith("--")}
    rest = [a for a in argv if not a.startswith("--")]
    if len(rest) != 2:
        print(f"wcfm diff: needs exactly two run directories, got {len(rest)}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        return 2

    a, b = Path(rest[0]), Path(rest[1])
    rc = _diff_configs(a, b, include_noise=bool(flags.get("all")))
    if flags.get("code"):
        rc = _diff_code(a, b, int(flags.get("context", 3))) or rc
    return rc
