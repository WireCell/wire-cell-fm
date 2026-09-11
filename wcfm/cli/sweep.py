"""`wcfm sweep`: Hydra enumerates, Condor launches.

One resolved config per point, stamped with `sweep_id`, its overrides and a content hash;
`run_name` defaults to `<sweep>_<hash8>`, and `sweeps/<id>/manifest.json` maps name to
overrides. Seeds are an axis like any other.

Hydra's sweeper does the enumeration -- `OverridesParser` and `BasicSweeper.split_arguments`
below are Hydra's own, so the full sweep syntax works: `model=dino,hybrid`,
`run.seed=range(1,4)`, `optim.lr=choice(1e-3,1e-4)`, globs. Hydra's launcher does not do the
launching. The shipped launchers are `basic` (sequential, in this process), `joblib`, `submitit`
(SLURM) and `ray`, none of which submits to HTCondor, so `--multirun` alone would run every
point of a sweep one after another on the login node. This is the seam: Hydra enumerates the
points, `wcfm submit` queues each one, and the manifest records what was enumerated.

The manifest earns its place against diffing the runs' resolved configs, which
`wcfm eval compare --by-config` does, on two things a config diff cannot recover:

- Which runs were one campaign. A diff over an arbitrary set of runs says what differs; it
  cannot say that these twelve were meant to be read together and that one of them is missing.
- Which runs are seed replicas. Without a manifest, `--group-by-seed` has to guess replica
  families by stripping a `_seed\d+` suffix off run names. With `seed_axis` declared here, two
  points are replicas exactly when every axis except the seed agrees, which is what
  `replica_group` records.

The two views stay complementary. The manifest says what was intended;
`--by-config` reads what the runs actually resolved to. When they disagree that is a finding --
a point whose config does not match its manifest entry did not run what the campaign thinks it
ran -- and it is only visible because both exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

__all__ = ["MANIFEST_FILE", "enumerate_points", "load_manifest", "main", "point_hash"]

MANIFEST_FILE = "manifest.json"

#: Override keys that name a seed. A sweep over any of these is a replica axis rather than a
#: scientific one, and `--group-by-seed` collapses it. `run.seed` is this repo's; the others are
#: accepted because a caller may sweep a seed that reaches the model.
SEED_KEYS: tuple[str, ...] = ("run.seed", "seed", "data.seed")

USAGE = """usage: wcfm sweep [options] <hydra overrides ...>

  wcfm sweep model=dino,hybrid run.seed=range(1,4) --id lr_scan
  wcfm sweep optim.lr=choice(1e-3,1e-4) model=hybrid --smoke --dry-run

Enumerates with Hydra's own sweeper, so the full sweep syntax works: `a,b`, `range(1,4)`,
`choice(a,b)`, `glob(*)`. A plain `key=value` is a fixed override, not an axis.

options:
  --id NAME          sweep id (default: sweep_<timestamp>). Names the manifest directory
  --sweeps-root DIR  where manifests go (default: $WCFM_SWEEPS or ./sweeps)
  --max N            refuse to enumerate more than N points (default 64)
  --smoke            passed to every point

The GPU count is not a flag here either: it comes from `launch.devices`, so a point's
allocation is whatever its own overrides resolve to. `launch=single_gpu,multi_2gpu` is
therefore a legitimate sweep axis rather than something to pass alongside one.
  --dry-run          enumerate and validate every point, submit nothing. NOTE: this still
                     inherits `wcfm submit --dry-run`, which writes each point's `.sub` into
                     its run directory -- so a dry run of N points creates N directories under
                     the output root. Nothing is queued and no manifest is written
  --repo DIR         checkout to package for the jobs (default: this package's repository)
  --config-dir DIR   a conf/ tree other than the repo's

`run.name` is set per point to `<sweep_id>_<hash8>` and must NOT be given by hand: every point
of a sweep would otherwise write into the same run directory.
"""


def point_hash(overrides: list[str]) -> str:
    """A content hash of one point's overrides, stable across runs and orderings.

    Sorted before hashing, so two invocations that list the same axes in a different order
    produce the same run name for the same point rather than a second directory holding an
    identical run.
    """
    payload = "\n".join(sorted(overrides))
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def enumerate_points(overrides: list[str]) -> tuple[list[list[str]], list[str]]:
    """`(points, swept keys)`, using Hydra's own sweeper.

    `BasicSweeper.split_arguments` is what `--multirun` calls, so a sweep enumerated here and
    one enumerated by Hydra contain the same points in the same order.
    """
    from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
    from hydra.core.override_parser.overrides_parser import OverridesParser

    parsed = OverridesParser.create().parse_overrides(overrides)
    axes = [o.get_key_element() for o in parsed if o.is_sweep_override()]
    batches = BasicSweeper.split_arguments(parsed, max_batch_size=None)
    points = [list(point) for batch in batches for point in batch]
    return points, axes


def _axis_values(point: list[str], axes: list[str]) -> dict[str, str]:
    out = {}
    for override in point:
        key, _, value = override.partition("=")
        if key in axes:
            out[key] = value
    return out


def _seed_axis(axes: list[str]) -> str | None:
    for key in SEED_KEYS:
        if key in axes:
            return key
    return None


def build_manifest(
    sweep_id: str, overrides: list[str], *, max_points: int = 64
) -> dict:
    """Enumerate and stamp, without submitting anything.

    Separated from `main` so `--dry-run` writes the same manifest a real submission does, and
    so the enumeration is testable without Condor.
    """
    points, axes = enumerate_points(overrides)
    if len(points) > max_points:
        raise SystemExit(
            f"wcfm sweep: this enumerates {len(points)} points, over the --max of {max_points}. "
            "Each point is a GPU job; raise --max deliberately if that is really the intent."
        )
    seed_axis = _seed_axis(axes)
    science_axes = [a for a in axes if a != seed_axis]

    entries = {}
    for point in points:
        values = _axis_values(point, axes)
        digest = point_hash(point)
        name = f"{sweep_id}_{digest}"
        entries[name] = {
            "overrides": point,
            "axis_values": values,
            "hash": digest,
            # Two points are seed replicas when every axis EXCEPT the seed agrees. Declared,
            # not inferred from the run name -- which is the whole reason the manifest exists.
            "replica_group": point_hash(
                [f"{k}={values[k]}" for k in sorted(science_axes) if k in values]
            )
            if seed_axis
            else None,
        }
    return {
        "sweep_id": sweep_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(overrides),
        "axes": axes,
        "science_axes": science_axes,
        # `None` means no seed was swept, which is a finding rather than a default: a single
        # seed per point gives `--group-by-seed` no spread to report.
        "seed_axis": seed_axis,
        "n_points": len(points),
        "points": entries,
    }


def load_manifest(path: Path | str) -> dict:
    """Read a manifest, given its directory or the file itself."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_FILE
    if not p.exists():
        raise SystemExit(
            f"{p} does not exist. `wcfm sweep` writes one per campaign under "
            "$WCFM_SWEEPS (or ./sweeps); pass the sweep's directory or its manifest.json."
        )
    return json.loads(p.read_text())


def write_manifest(root: Path, manifest: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / MANIFEST_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path


def _pop_opt(args: list[str], name: str, default: str | None = None) -> str | None:
    for i, arg in enumerate(args):
        if arg == name:
            value = args[i + 1] if i + 1 < len(args) else default
            del args[i : i + 2]
            return value
        if arg.startswith(name + "="):
            del args[i]
            return arg.split("=", 1)[1]
    return default


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    args = list(argv)
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")
    smoke = "--smoke" in args
    if smoke:
        args.remove("--smoke")

    sweep_id = _pop_opt(args, "--id", "") or f"sweep_{time.strftime('%Y%m%d_%H%M%S')}"
    sweeps_root = Path(
        _pop_opt(args, "--sweeps-root", "") or os.environ.get("WCFM_SWEEPS", "sweeps")
    )
    max_points = int(_pop_opt(args, "--max", "64") or 64)
    # `--gpus` was retired on 2026-09-10. Caught by name rather than left to Hydra, which
    # rejects it as an override with a lexer error that names nothing useful.
    if any(a == "--gpus" or a.startswith("--gpus=") for a in args):
        print(
            "wcfm sweep: --gpus was removed. The GPU count now comes from `launch.devices` "
            "in the config, so it is stated once instead of twice:\n"
            "    launch=single_gpu | launch=multi_2gpu | launch=multi_6gpu\n"
            "    launch.devices=N        (for a count with no preset)\n"
            "One count sets both Condor's request_gpus and Fabric's world, so the two cannot "
            "disagree.",
            file=sys.stderr,
        )
        return 2
    repo = _pop_opt(args, "--repo", "")
    config_dir = _pop_opt(args, "--config-dir", "")

    overrides = [a for a in args if not a.startswith("--")]
    stray = [a for a in args if a.startswith("--")]
    if stray:
        print(f"wcfm sweep: unknown option(s) {stray}", file=sys.stderr)
        return 2
    if not overrides:
        print("wcfm sweep: needs at least one override to sweep over", file=sys.stderr)
        return 2
    if any(o.startswith("run.name=") for o in overrides):
        print(
            "wcfm sweep: `run.name` is set per point to <sweep_id>_<hash8> and must not be "
            "given by hand -- every point would otherwise write into one run directory and "
            "overwrite the last. Use --id to name the campaign.",
            file=sys.stderr,
        )
        return 2

    manifest = build_manifest(sweep_id, overrides, max_points=max_points)
    root = sweeps_root / sweep_id
    axes = manifest["axes"]

    print(f"sweep:    {sweep_id}")
    print(f"points:   {manifest['n_points']}")
    print(f"axes:     {axes or '(none -- nothing is swept, this is a single run)'}")
    if manifest["seed_axis"]:
        groups = {p["replica_group"] for p in manifest["points"].values()}
        print(
            f"seeds:    {manifest['seed_axis']} is a replica axis; "
            f"{len(groups)} distinct configuration(s), "
            f"{manifest['n_points'] // max(1, len(groups))} replica(s) each"
        )
    else:
        # The plan is explicit that one seed per point is not a result. Say so; do not refuse.
        print(
            "seeds:    NOT swept. One seed per point is not a result -- a difference between "
            "two points cannot be read until it clears the seed-to-seed spread. Add "
            "`run.seed=range(1,4)` unless this sweep is exploratory."
        )
    print(f"manifest: {root / MANIFEST_FILE}")

    if not dry_run:
        write_manifest(root, manifest)

    from wcfm.cli import submit as submit_cli

    failures = []
    for name, entry in manifest["points"].items():
        point_args = [*entry["overrides"], f"run.name={name}"]
        if repo:
            point_args += ["--repo", repo]
        if config_dir:
            point_args += ["--config-dir", config_dir]
        if smoke:
            point_args.append("--smoke")
        if dry_run:
            point_args.append("--dry-run")

        print(f"\n{'=' * 78}\n== {name}   {entry['axis_values']}\n{'=' * 78}")
        rc = submit_cli.main(point_args)
        if rc != 0:
            failures.append(name)

    if dry_run:
        print(f"\ndry run: {manifest['n_points']} point(s) enumerated, nothing submitted")
        print(
            "         each point was composed and validated through the real submit path, so "
            f"{manifest['n_points']} run director(ies) with a .sub in them now exist under the "
            "output root -- that is `wcfm submit --dry-run`'s behaviour, inherited. No manifest "
            "was written."
        )
        print(f"         the manifest that WOULD be written to {root / MANIFEST_FILE}:\n")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if failures:
        # Every point is attempted even if one fails, for the reason the probe runner has: a
        # campaign should not lose eleven queued jobs because the twelfth was misconfigured.
        # The manifest still lists them, so a re-run submits only what is missing.
        print(f"\n{len(failures)} of {manifest['n_points']} point(s) FAILED to submit: {failures}")
        return 1
    print(f"\nsubmitted {manifest['n_points']} point(s); manifest at {root / MANIFEST_FILE}")
    return 0
