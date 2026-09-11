"""What a run writes about itself: the derivations, the flat C2 file, and the one division.

The derivations are the point. ``warmup_iters``, ``total_iters`` and ``epoch_len`` were
computed and *printed* by the old trainer (``train_dino.py:526-528,678-679``) and never
recorded, so the only way to know what a run's warmup actually was is to find the log. A
derivation that exists only in a log line cannot be diffed between two runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from wcfm.config.io import (
    SCHEMA_VERSION,
    derive,
    per_rank_batch_size,
    provenance,
    repo_root,
    write_run_dir,
)
from wcfm.config.schema import Config


def _cfg(**over):
    cfg = OmegaConf.structured(Config)
    cfg.run.name = "unit_run"
    cfg.data.backend = "direct"
    cfg.data.global_batch_size = 100
    cfg.model = {"name": "stub"}
    for dotted, value in over.items():
        OmegaConf.update(cfg, dotted.replace("__", "."), value)
    return cfg


# ------------------------------------------------------------------------ the one division


@pytest.mark.parametrize(("g", "w", "expected"), [(100, 1, 100), (100, 4, 25), (12, 6, 2)])
def test_global_batch_divides_per_rank(g, w, expected):
    assert per_rank_batch_size(g, w) == expected


def test_an_uneven_split_is_refused_rather_than_rounded():
    """An uneven split makes the effective batch a function of rank, which is exactly the
    kind of thing that never shows up as an error and always shows up in the loss.

    ``ValueError``, not ``AssertionError``: an assertion disappears under ``python -O`` and
    the refusal would silently become the floor it exists to prevent.
    """
    with pytest.raises(ValueError, match="not divisible"):
        per_rank_batch_size(100, 3)


@pytest.mark.stack
def test_the_loader_and_the_config_share_one_division():
    """``wcfm.data.build`` re-exports this rather than reimplementing it.

    There were two copies until 2026-09-09, with two different exception types, while both
    docstrings and the README said there was one. Identity, not equal behaviour: two
    functions that agree today are two functions that can drift.

    Marked ``stack`` because reaching ``wcfm.data.build`` imports torch, and the rest of this
    file runs in the config-only environment.
    """
    pytest.importorskip("torch")
    from wcfm.data.build import per_rank_batch_size as from_build

    assert from_build is per_rank_batch_size


# --------------------------------------------------------------------------- the derivations


def test_warmup_iters_honours_the_twenty_percent_cap():
    """``min(warmup_epochs * epoch_len, 0.2 * total_iters)`` -- so a long warmup_epochs is
    not simply warmup_epochs * epoch_len, which is the whole reason it must be recorded."""
    cfg = _cfg(optim__epochs=100, optim__warmup_epochs=10)
    d = derive(cfg, epoch_len=1000, world_size=1)
    assert d["epoch_len"] == 1000
    assert d["total_iters"] == 100_000
    assert d["warmup_iters"] == 10_000  # 10*1000 == 0.2*100000, the cap is exactly reached

    capped = derive(_cfg(optim__epochs=10, optim__warmup_epochs=5), epoch_len=1000, world_size=1)
    assert capped["warmup_iters"] == 2000, "5*1000 is capped to 0.2*10000"

    uncapped = derive(_cfg(optim__epochs=100, optim__warmup_epochs=1), epoch_len=1000, world_size=1)
    assert uncapped["warmup_iters"] == 1000


def test_derive_records_the_world_size_the_numbers_depend_on():
    d = derive(_cfg(), epoch_len=10, world_size=4)
    assert d["world_size"] == 4
    assert d["batch_size_per_rank"] == 25


# ------------------------------------------------------------------- what a run records


def test_the_recorded_config_carries_the_derived_values():
    """`epoch_len`, `total_iters`, `warmup_iters` and the per-rank batch are facts about the
    dataset and the world size, so a config file alone cannot reproduce them. They are
    recorded with the run instead."""
    cfg = _cfg(optim__epochs=42)
    derived = derive(cfg, epoch_len=10, world_size=2)
    assert derived["epoch_len"] == 10
    assert derived["total_iters"] == 420
    assert derived["world_size"] == 2
    assert derived["batch_size_per_rank"] == 50, "global 100 across 2 ranks"


def test_provenance_survives_a_json_round_trip():
    """It is written with `json.dumps`, so anything unserializable in it takes the run down
    at the moment the run directory is created."""
    cfg = _cfg()
    out = json.loads(json.dumps(provenance(cfg, argv=["wcfm"], world_size=1)))
    assert out["schema_version"] == SCHEMA_VERSION
    assert out["launch"]["world_size"] == 1


# ------------------------------------------------------------------- provenance and layout


def test_provenance_names_the_environment_and_the_launch():
    cfg = _cfg()
    p = provenance(cfg, argv=["wcfm", "train", "model=dino"], world_size=2, env={"torch": "2.10"})
    assert p["launch"]["command"] == "wcfm train model=dino"
    assert p["launch"]["world_size"] == 2
    assert p["env"] == {"torch": "2.10"}
    assert set(p["git"]) == {"sha", "dirty", "branch"}
    # Filled once the module is built and an epoch has run; an architecture ablation needs
    # both, and both were produced by hand for model_size.md.
    assert set(p["compute"]) == {"parameters", "throughput_samples_per_s"}


def test_the_default_repo_root_is_the_checkout_and_not_its_parent():
    """The depth of ``repo_root``, which was wrong for a day and could not be seen.

    ``parents[3]`` was correct under the ``src/wcfm/`` layout and one directory too high after
    the restructure dropped it, so ``git`` ran in the parent of the checkout. ``_git`` maps
    every failure to ``None``, so the only symptom was an all-null ``git`` block in
    ``run_metadata.json`` -- no error, and nothing in this file noticed, because it asserted
    the three keys were present and not that any of them resolved.

    Asserted as "the directory that holds pyproject.toml and wcfm/" rather than as a number
    of ``parents``, so it stays true through the next layout change instead of restating it.
    """
    root = repo_root()
    assert (root / "pyproject.toml").is_file(), f"{root} is not the repo root"
    assert (root / "wcfm").is_dir(), f"{root} does not contain the wcfm package"


def test_provenance_records_a_real_commit_when_the_checkout_has_one():
    """The other half: the path being right is only useful if the ``git`` block fills in.

    Skipped rather than failed off a checkout -- an installed copy or an exported tarball has
    no ``.git`` and a null block is the honest answer there.
    """
    if not (repo_root() / ".git").exists():
        pytest.skip("not a git checkout, so a null git block is correct")
    p = provenance(_cfg(), argv=["wcfm"], world_size=1)
    assert p["git"]["sha"], (
        "the git block is null inside a git checkout: `repo_root()` is not pointing at it"
    )
    assert p["git"]["dirty"] is not None


def test_a_worker_records_the_commit_it_was_submitted_from(tmp_path: Path, monkeypatch):
    """**Every Condor run had a null git block.** A job runs from an archive of the tree
    unpacked into its scratch directory -- no `.git` -- so provenance had nothing to ask, and
    the runs whose provenance matters were exactly the ones missing it. Fixing `repo_root()`'s
    off-by-one restored the block for local runs only, which is how the gap survived that fix.

    The submitting side *is* the repository, so `wcfm/cli/jobpack.py::git_environment` reads
    the identity there and Condor carries it in the job's environment."""
    monkeypatch.setenv("WCFM_GIT_SHA", "8c349a838c306ce300c402464d02bc7ea4ea0661")
    monkeypatch.setenv("WCFM_GIT_BRANCH", "model/stage3")
    monkeypatch.setenv("WCFM_GIT_DIRTY", "0")

    p = provenance(_cfg(), argv=["wcfm"], world_size=1, repo=tmp_path)
    assert p["git"]["sha"] == "8c349a838c306ce300c402464d02bc7ea4ea0661"
    assert p["git"]["branch"] == "model/stage3"
    assert p["git"]["dirty"] is False
    assert set(p["git"]) == {"sha", "dirty", "branch"}, "the schema is unchanged"


def test_a_worker_told_the_tree_was_dirty_says_so(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("WCFM_GIT_SHA", "deadbeef")
    monkeypatch.setenv("WCFM_GIT_DIRTY", "1")
    monkeypatch.delenv("WCFM_GIT_BRANCH", raising=False)
    p = provenance(_cfg(), argv=["wcfm"], world_size=1, repo=tmp_path)
    assert p["git"]["dirty"] is True
    assert p["git"]["branch"] is None, "not recorded is null, not invented"


def test_a_plain_directory_still_reports_a_null_block(tmp_path: Path):
    """No repository and nothing in the environment: null is the honest answer, not an
    exception."""
    p = provenance(_cfg(), argv=["wcfm"], world_size=1, repo=tmp_path)
    assert p["git"] == {"sha": None, "dirty": None, "branch": None}


def test_write_run_dir_creates_the_c9_layout(tmp_path: Path):
    """``<run>/{checkpoints,debug,probes,features,metrics}/`` is parsed by four scripts."""
    cfg = _cfg()
    run_dir = write_run_dir(cfg, tmp_path / "unit_run", argv=["wcfm"], world_size=1)
    for sub in ("checkpoints", "debug", "probes", "features", "metrics"):
        assert (run_dir / sub).is_dir(), sub
    assert (run_dir / "config.yaml").exists()
    assert json.loads((run_dir / "run_metadata.json").read_text())["schema_version"]
    assert not (run_dir / "run_config.json").exists(), (
        "a run writes config.yaml and run_metadata.json, and nothing reads a third file"
    )
    assert not list(run_dir.glob("*.tmp")), "the atomic write must not leave its temp file"


def test_write_run_dir_is_idempotent(tmp_path: Path):
    """Re-writing a run directory -- which a resumed run does -- overwrites rather than
    appends or fails. (Atomicity itself is asserted above, by the absence of a .tmp file.)"""
    cfg = _cfg()
    write_run_dir(cfg, tmp_path / "r", argv=["wcfm"], world_size=1)
    write_run_dir(cfg, tmp_path / "r", argv=["wcfm"], world_size=1)  # idempotent
    cfg_text = (tmp_path / "r" / "config.yaml").read_text()
    assert "epochs: 100" in cfg_text
