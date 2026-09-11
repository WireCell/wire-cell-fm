"""What two runs actually differ by, read off their resolved configs.

The diff *is* the definition of an ablation. Today it is done by eye across 60-key JSON files,
which is why two runs that differ only in a derived quantity nobody printed look identical.

Two consumers, and they are the same question asked twice:

* `wcfm diff <a> <b>` -- what is different between these two runs;
* `wcfm eval compare --by-config` -- one column per key that varies across the runs in a table.

  This is not a substitute for the sweep manifest. `wcfm sweep` writes
  `sweeps/<id>/manifest.json` and `compare --sweep <id>` reads it, and the two views answer
  different questions. The manifest records what a campaign intended to vary and which of its
  points are seed replicas, neither of which is recoverable from a set of resolved configs: a
  diff over arbitrary runs cannot say which were meant to be read together, nor that one is
  missing. `--by-config` records what actually differed, including a derived value nobody swept
  on purpose. A disagreement between them is a finding.

Flattening to dotted keys rather than walking the tree makes "one column per differing key"
fall out directly, and makes the diff stable under a config that gained a nesting level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["differing_keys", "flatten", "load_run_config", "run_config_path"]

#: Keys that differ between any two runs by construction and say nothing about what was varied.
#: Excluded from the *default* view only -- `--all` shows them, because "the seed is the only
#: difference" is exactly what a seed-replica table needs to establish.
NOISE_PREFIXES: tuple[str, ...] = (
    "run.name",
    "run.output_root",
    "hydra.",
)


def flatten(node: Any, prefix: str = "") -> dict[str, Any]:
    """A nested config as `{"a.b.c": value}`.

    Lists are flattened by index rather than compared whole, so "the third term's weight
    changed" reads as one key instead of two long lists a reader has to align by eye.
    """
    out: dict[str, Any] = {}
    if isinstance(node, dict):
        for k, v in node.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = node
    return out


def run_config_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "config.yaml"


def load_run_config(run_dir: Path | str) -> dict:
    """The run's own resolved config, flattened. Raises with the path if it is not there."""
    from omegaconf import OmegaConf

    path = run_config_path(run_dir)
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist, so there is nothing to diff. Every run `wcfm train` writes "
            "records its fully resolved config there; a directory without one was not produced "
            "by this framework, or the job died before `write_run_dir`."
        )
    return flatten(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


def differing_keys(configs: dict[str, dict], *, include_noise: bool = False) -> list[str]:
    """The dotted keys whose value is not the same across every config given.

    A key missing from one config counts as differing: "this run had no `model.terms.occupancy`
    at all" is the most interesting difference there is, and treating absence as equal to any
    value would hide exactly the ablations worth seeing.
    """
    if len(configs) < 2:
        return []
    all_keys: set[str] = set()
    for cfg in configs.values():
        all_keys |= set(cfg)

    _MISSING = object()
    out = []
    for key in sorted(all_keys):
        if not include_noise and key.startswith(NOISE_PREFIXES):
            continue
        values = [cfg.get(key, _MISSING) for cfg in configs.values()]
        first = values[0]
        if any(v is not first and v != first for v in values[1:]):
            out.append(key)
    return out
