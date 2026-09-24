"""Packaging the checkout for a Condor job.

A tarball rather than a path, because a job is submitted minutes or hours before it starts.
Condor transfers `transfer_input_files` from the submit machine when the job starts -- only
`condor_submit -spool` copies at submit time -- so pointing `transfer_input_files` straight at
the live checkout lets an edit made during the queue window change what runs. Every submit path
writes one archive of the tree next to the run's own outputs and transfers that. The archive is
written once and never touched again, so the run is pinned from the instant it is queued while
the checkout stays editable. It is about a megabyte.

What the job gets is a tree rather than an installed package, and `wire_cell_fm.egg-info/` must
land beside `wcfm/` on the PYTHONPATH entry: the model's config schemas arrive through
`[project.entry-points."wcfm.config_schemas"]`, which `importlib.metadata` reads from package
metadata found by scanning `sys.path` directories. A tarball without it composes no `model=`
preset and the job dies in Hydra.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

__all__ = [
    "ARCHIVE_NAME",
    "CONTENTS",
    "OPTIONAL",
    "REQUIRED",
    "check_repo",
    "git_environment",
    "pack_repo",
    "pop_opt",
    "request_disk",
    "stage_executable",
]

ARCHIVE_NAME = "repo.tgz"

#: What a worker cannot run without. A tree missing any of these is refused before queueing.
REQUIRED = (
    "wcfm",
    "conf",
    "gridutils",
    "pyproject.toml",
    "README.md",
    "wire_cell_fm.egg-info",
)

#: Packed when present and skipped when not, so a checkout without it still submits. `wcfm test
#: --gpu` runs the suite on a worker, so that command asks for it through `check_repo(also=...)`.
OPTIONAL = ("tests",)

#: Everything a submit packs, from a tree that has all of it.
CONTENTS = (
    "wcfm",
    "conf",
    "tests",
    "gridutils",
    "pyproject.toml",
    "README.md",
    "wire_cell_fm.egg-info",
)

_EXCLUDE = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", ".git"}


def _keep(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    name = Path(info.name).name
    if name in _EXCLUDE or name.endswith(".pyc"):
        return None
    return info


def check_repo(repo: Path, also: tuple[str, ...] = ()) -> None:
    """Refuse a tree that would not run, here rather than on a worker.

    Separate from `pack_repo` so that `--dry-run` can make this check without writing a
    megabyte: the missing-egg-info case is the one worth catching early, and it is the one a
    dry run would otherwise not see. Raises `FileNotFoundError`.

    `also` names entries this caller needs on top of `REQUIRED`. `wcfm test --gpu` passes
    `("tests",)`, because a job that runs pytest on a worker needs the suite it runs.
    """
    repo = Path(repo)
    missing = [c for c in (*REQUIRED, *also) if not (repo / c).exists()]
    if "wire_cell_fm.egg-info" in missing:
        raise FileNotFoundError(
            f"no wire_cell_fm.egg-info in {repo}. The model's config schemas arrive through an "
            "entry point read from package metadata; without it no model= preset resolves on "
            "the worker. Regenerate it with:\n  uv pip install -e . --no-deps"
        )
    if missing:
        raise FileNotFoundError(f"{repo} is missing {', '.join(missing)}")


def pack_repo(repo: Path, dest_dir: Path) -> Path:
    """Write `<dest_dir>/repo.tgz` from `repo` and return its path."""
    repo, dest_dir = Path(repo).resolve(), Path(dest_dir)
    check_repo(repo)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / ARCHIVE_NAME
    tmp = archive.with_suffix(".tgz.tmp")
    # Written to a temp name and renamed: `condor_submit` reads this file, and a submission
    # racing a half-written archive is a failure that would surface on the worker.
    with tarfile.open(tmp, "w:gz") as tar:
        for item in CONTENTS:
            if (repo / item).exists():
                tar.add(repo / item, arcname=item, filter=_keep)
    tmp.replace(archive)
    return archive


def stage_executable(script: Path, dest_dir: Path) -> Path:
    """Copy a job script beside the archive and return the copy.

    Condor also transfers the `executable` at job start, so an executable named in the live
    checkout is exposed to the same queue-window edit as the tree -- and worse, bash reads a
    script incrementally, so an edit to a *running* job's script truncates it silently. The
    copy is what the `.sub` names.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    staged = dest_dir / Path(script).name
    shutil.copy2(script, staged)
    staged.chmod(0o755)
    return staged


def git_environment(repo: Path) -> str:
    """The repo's identity as Condor `environment` assignments.

    The worker has no `.git` -- the archive excludes it -- so `provenance()` would write a null
    `git` block on exactly the runs whose provenance matters. The submitting side *is* the
    repository, so the identity is read here and carried in the environment;
    `wcfm/config/io.py::_env_git` reads it back.

    Only `bool(status)` is ever used, so the dirty *file list* is not carried: it would have to
    survive Condor's environment quoting to say nothing more than `1`.
    """

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, timeout=10, check=True
            )
        except Exception:  # noqa: BLE001 - no git, no repo, or a timeout are all "unknown"
            return None
        return out.stdout.strip()

    sha = _git("rev-parse", "HEAD")
    if not sha:
        return ""
    status = _git("status", "--porcelain")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD") or ""
    return (
        f"WCFM_GIT_SHA={sha} WCFM_GIT_BRANCH={branch} "
        f"WCFM_GIT_DIRTY={1 if status else 0}"
    )


def pop_opt(args: list[str], name: str, default: str | None = None) -> str | None:
    """Remove `name VALUE` or `name=VALUE` from `args` and return VALUE, else `default`.

    The submit commands take a few options of their own ahead of arguments they pass through
    untouched -- Hydra overrides, a module's argv -- so they cannot hand the whole list to
    argparse. `args` is edited in place.
    """
    for i, arg in enumerate(args):
        if arg == name:
            value = args[i + 1] if i + 1 < len(args) else default
            del args[i : i + 2]
            return value
        if arg.startswith(name + "="):
            del args[i]
            return arg.split("=", 1)[1]
    return default


def request_disk(default_kib: str) -> str:
    """`request_disk` in KiB, overridable with `WCFM_REQUEST_DISK`.

    The default Condor computes is the size of the executable plus the input files -- about a
    megabyte, while the job writes checkpoints and feature stores into the same scratch
    directory. The L40S nodes advertise terabytes, so this is generous on purpose.
    """
    return os.environ.get("WCFM_REQUEST_DISK", default_kib)
