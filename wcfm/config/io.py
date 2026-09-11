"""What a run writes about itself, and the derivations more than one package needs.

Two artifacts, with different jobs:

`config.yaml`
    The fully resolved config: what the run trains on, nested, interpolations expanded.
    `wcfm eval` rebuilds the model and the loader from it and `wcfm diff` compares two of
    them. The derived values land here too -- `epoch_len`, `total_iters`, `warmup_iters`,
    the per-rank batch size -- because a number that exists only in a log line cannot be
    diffed between runs.

`run_metadata.json`
    What the config cannot say: git sha, dirty flag and branch, the launch command, world
    size, hostname, the `wcfm env-check` block, split ids, and compute accounting.

This module stays torch-free, so the config-only environment can check it, and so the two
derivations below have exactly one implementation between the engine and the recorded config.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

SCHEMA_VERSION = 2


def resolved_dict(cfg: DictConfig) -> dict:
    """The config as plain data, interpolations resolved, ready to be written or pickled."""
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]


def derive(cfg: DictConfig, epoch_len: int, world_size: int) -> dict:
    """The numbers a run implies but does not state.

    `epoch_len` is `len(train_loader)`: a fact about the dataset and the world size, not
    about the config, so none of these can be recovered from a config file alone. They are
    recorded with the run for that reason.
    """
    epochs = int(cfg.optim.epochs)
    total_iters = epochs * epoch_len
    return {
        "epoch_len": epoch_len,
        "total_iters": total_iters,
        # The learning rate's warmup. Every other scheduled quantity carries its own on its
        # `optim.schedules` entry.
        "warmup_iters": warmup_iters_from_epochs(
            cfg.optim.warmup_epochs, epoch_len, total_iters
        ),
        "world_size": world_size,
        "batch_size_per_rank": per_rank_batch_size(int(cfg.data.global_batch_size), world_size),
    }


def warmup_iters_from_epochs(
    warmup_epochs: float, epoch_len: int, total_iters: int
) -> int:
    """Warmup epochs to iterations, capped at a fifth of the run."""
    return min(int(float(warmup_epochs) * epoch_len), int(0.2 * total_iters))


def per_rank_batch_size(global_batch_size: int, world_size: int) -> int:
    """The one place a global batch is divided by the world size.
    An inexact split is refused. `wcfm.data.build` re-exports this 
    rather than reimplementing it, so the loader and the recorded 
    config cannot disagree about the number.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} is not divisible by "
            f"world_size={world_size}; pick a global batch that divides evenly rather than "
            f"letting the split floor"
        )
    return global_batch_size // world_size


def repo_root() -> Path:
    """The repository this package was imported from.

    `wcfm/` sits at the repo root, so this file is `<repo>/wcfm/config/io.py` 
    and the root is `parents[2]`.
    """
    return Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except Exception:  # noqa: BLE001 - no git, no repo, or a timeout are all "unknown"
        return None
    return out.stdout.strip()


def _env_git() -> dict | None:
    """The `git` block for a worker, which has no repository to ask.

    A Condor job runs from an archive of the tree (`wcfm/cli/jobpack.py`) unpacked into its
    scratch directory. The archive excludes `.git`, so every `git` call there fails -- and
    before this the runs whose provenance matters were exactly the ones recording
    `{sha: null, dirty: null, branch: null}`.

    The submitting side *is* the repository, so `wcfm submit`, `wcfm test --gpu` and
    `wcfm eval submit` read the identity there and put it in the job's `environment`. These are
    the same facts from the same source, recorded earlier.
    """
    sha = os.environ.get("WCFM_GIT_SHA")
    if not sha:
        return None
    dirty = os.environ.get("WCFM_GIT_DIRTY")
    branch = os.environ.get("WCFM_GIT_BRANCH")
    return {
        "sha": sha,
        "dirty": bool(int(dirty)) if dirty in ("0", "1") else None,
        "branch": branch or None,
    }


def provenance(
    cfg: DictConfig,
    *,
    argv: list[str],
    world_size: int,
    env: dict | None = None,
    derived: dict | None = None,
    module: dict | None = None,
    repo: Path | None = None,
) -> dict:
    repo = repo or repo_root()
    status = _git("status", "--porcelain", cwd=repo)
    sha = _git("rev-parse", "HEAD", cwd=repo)
    git = {
        "sha": sha,
        "dirty": bool(status) if status is not None else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo),
    }
    if sha is None:
        git = _env_git() or git
    return {
        "schema_version": SCHEMA_VERSION,
        "git": git,
        "launch": {
            "command": " ".join(argv),
            "world_size": world_size,
            "num_workers": int(cfg.run.num_workers),
            "hostname": os.uname().nodename,
        },
        "env": env or {},
        "data": {
            "train_split_id": str(cfg.data.splits.id),
            "eval_set_id": str(cfg.data.splits.id),
            "backend": str(cfg.data.backend),
        },
        "derived": derived or {},
        # Filled in by the engine once the module is built and an epoch has run: parameter
        # count per module, and measured throughput. An architecture comparison needs both.
        "compute": {"parameters": None, "throughput_samples_per_s": None},
        # Whatever the module says about itself, verbatim, from its optional `provenance()`.
        # An opaque dict because this package must not name a model type.
        "module": module or {},
    }


def write_run_dir(
    cfg: DictConfig,
    run_dir: Path,
    *,
    argv: list[str],
    world_size: int,
    env: dict | None = None,
    derived: dict | None = None,
    module: dict | None = None,
) -> Path:
    """Create `<run_dir>/{checkpoints,debug,probes,features,metrics}` and write the two
    artifacts described at the top of this module. Returns the run directory."""
    run_dir = Path(run_dir)
    for sub in ("checkpoints", "debug", "probes", "features", "metrics"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    _write_json(
        run_dir / "run_metadata.json",
        provenance(
            cfg,
            argv=argv,
            world_size=world_size,
            env=env,
            derived=derived,
            module=module,
        ),
    )
    return run_dir


def _write_json(path: Path, obj: dict) -> None:
    """Write via a temp file and rename, so a reader polling the run directory never sees
    half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")
    tmp.replace(path)
