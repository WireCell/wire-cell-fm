"""``wcfm submit``: composition, validation, and the submit file.

The whole point of this command being Python rather than the old ``submit.sh`` is that a run is
**validated before it is queued** -- so most of these tests are about what it refuses. Each
refusal below otherwise costs a GPU slot and is discovered from a log rather than a terminal.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")

from wcfm.cli.__main__ import COMMANDS  # noqa: E402
from wcfm.cli.submit import main  # noqa: E402

pytestmark = pytest.mark.stack

REPO = Path(__file__).resolve().parents[1]

TOY_MODEL = """# @package _global_
model:
  _target_: tests.toy.ToyModule
  dim: 4
"""


@pytest.fixture
def conf_dir(tmp_path):
    target = tmp_path / "conf"
    shutil.copytree(REPO / "conf", target)
    (target / "model").mkdir(exist_ok=True)
    (target / "model" / "toy.yaml").write_text(TOY_MODEL)
    return target


@pytest.fixture
def env(tmp_path, monkeypatch):
    libs = tmp_path / "libs"
    # All three: `wcfm train` composes its config on the worker, so hydra and omegaconf are
    # runtime dependencies of the job, not test dependencies. See the missing-libs test below.
    for pkg in ("lightning_fabric", "hydra", "omegaconf"):
        (libs / pkg).mkdir(parents=True)
    monkeypatch.setenv("WCFM_LIBS", str(libs))
    monkeypatch.setenv("WCFM_PYENV", str(tmp_path / "uvenv"))
    monkeypatch.setenv("WCFM_OUTPUT_BASE", str(tmp_path / "out"))
    monkeypatch.setenv("WCFM_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def _args(conf_dir, *extra, name="run_a"):
    return [
        "--dry-run",
        "--config-dir",
        str(conf_dir),
        "--repo",
        str(REPO),
        "model=toy",
        f"run.name={name}",
        *extra,
    ]


def _sub(env, name="run_a"):
    return (env / "out" / name / f"{name}.sub").read_text()


def test_submit_is_registered():
    assert COMMANDS["submit"] == "wcfm.cli.submit"


def test_help_says_run_name_is_required(capsys):
    assert main(["--help"]) == 0
    assert "run.name` is required" in capsys.readouterr().out


def test_a_dry_run_composes_validates_and_writes_the_sub(env, conf_dir, capsys):
    assert main(_args(conf_dir)) == 0
    out = capsys.readouterr().out
    assert "run:     run_a" in out
    assert "parameters" in out, "the module was actually built, not just named"

    text = _sub(env)
    assert "request_gpus            = 1" in text
    assert "should_transfer_files   = YES" in text
    assert "'model=toy'" in text and "'run.name=run_a'" in text
    # The executable is the staged copy in the run's own job/, never the checkout's.
    assert f"executable              = {env / 'out' / 'run_a' / 'job' / 'trainjob.sh'}" in text
    assert str(REPO / "gridutils" / "train" / "trainjob.sh") not in text


def test_the_job_transfers_an_archive_and_returns_no_files(env, conf_dir):
    """Turning transfer on has two halves and the second is the dangerous one.

    Condor reads `transfer_input_files` when the job STARTS, not at submit, so the archive is
    what pins the run -- transferring the checkout itself would leave the whole queue window
    open to an edit. And without `transfer_output_files = ""` Condor copies everything created
    at the top of scratch back to `initialdir`, which is the directory `trainjob.sh` has
    already rsynced the real run into: the duplicate would land on top of the correct result.
    """
    assert main(_args(conf_dir)) == 0
    text = _sub(env)
    archive = env / "out" / "run_a" / "job" / "repo.tgz"
    assert f"transfer_input_files    = {archive}" in text
    assert 'transfer_output_files   = ""' in text
    assert "when_to_transfer_output = ON_EXIT" in text
    # The job is told the archive's NAME: it unpacks it from its own scratch directory, and a
    # path into the checkout would not exist there.
    assert 'arguments               = "repo.tgz ' in text


def test_request_disk_is_set(env, conf_dir):
    """Never set before transfer was turned on, and the default Condor computes is the size of
    the inputs -- a megabyte -- while the job writes every checkpoint into the same scratch."""
    assert main(_args(conf_dir)) == 0
    assert "request_disk            = 50000000" in _sub(env)


def test_the_git_identity_is_carried_in_the_environment(env, conf_dir):
    """The worker has no `.git` (the archive excludes it), so without this every Condor run
    records `git: {sha: null, ...}` -- on exactly the runs whose provenance matters."""
    assert main(_args(conf_dir)) == 0
    assert "WCFM_GIT_SHA=" in _sub(env)


def test_a_dry_run_writes_no_archive(env, conf_dir):
    """A dry run of a sweep would otherwise leave one archive per point. What it *does* check
    is that the tree would package -- the missing-egg-info failure, which is the one that
    otherwise surfaces on a worker."""
    assert main(_args(conf_dir)) == 0
    assert not (env / "out" / "run_a" / "job" / "repo.tgz").exists()


def test_a_tree_without_egg_info_is_refused_before_queueing(env, conf_dir, tmp_path, capsys):
    """The model's config schemas arrive through `[project.entry-points."wcfm.config_schemas"]`,
    read from *package metadata*. A tree without it composes no `model=` preset and dies in
    Hydra on the worker -- a failure that has cost two queue slots."""
    import shutil as _shutil

    bare = tmp_path / "bare_repo"
    # Whichever of these the checkout has: a tree stripped of `docs/` is still a tree this
    # command has to refuse for the right reason.
    for item in ("wcfm", "conf", "tests", "docs", "gridutils"):
        if (REPO / item).is_dir():
            _shutil.copytree(REPO / item, bare / item, symlinks=True,
                             ignore=_shutil.ignore_patterns("__pycache__"))
    for item in ("pyproject.toml", "README.md"):
        _shutil.copy2(REPO / item, bare / item)

    args = [a if a != str(REPO) else str(bare) for a in _args(conf_dir)]
    assert main(args) == 2
    assert "wire_cell_fm.egg-info" in capsys.readouterr().err


def test_overrides_are_quoted_individually(env, conf_dir):
    """Condor splits arguments on whitespace outside quotes, so an override carrying a space
    or a bracket -- a list value, say -- would otherwise arrive as two arguments."""
    assert main(_args(conf_dir, "run.save_at=[10,20]")) == 0
    assert "'run.save_at=[10,20]'" in _sub(env)


def test_an_empty_run_name_is_refused_before_the_queue(env, conf_dir, capsys):
    assert main(_args(conf_dir, name="")) == 2
    assert "run.name is empty" in capsys.readouterr().err


def test_a_batch_that_does_not_divide_by_the_gpu_count_is_refused(env, conf_dir, capsys):
    assert main(_args(conf_dir, "data.global_batch_size=100", "launch.devices=3")) == 2
    err = capsys.readouterr().err
    # `validate(cfg)` owns this now -- one check against the config, rather than one here
    # against a flag and a different one in the job against `launch.devices`.
    assert "not divisible by world_size=3" in err


def test_a_divisible_batch_at_two_gpus_is_accepted(env, conf_dir):
    assert main(_args(conf_dir, "data.global_batch_size=100", "launch.devices=2")) == 0
    assert "request_gpus            = 2" in _sub(env)


def test_request_gpus_and_the_jobs_rank_count_both_come_from_launch_devices(env, conf_dir):
    """One source of truth, checked at both ends of the .sub.

    Until 2026-09-10 `--gpus N` set Condor's allocation while `launch.devices` set Fabric's
    world, and nothing reconciled them: `--gpus 2` with the default `launch=single_gpu` was
    accepted and produced two ranks each building a one-device Fabric. Both numbers are now
    read off the resolved config, so they cannot disagree.
    """
    assert main(_args(conf_dir, "launch=multi_2gpu", "data.global_batch_size=100")) == 0
    text = _sub(env)
    assert "request_gpus            = 2" in text
    # trainjob.sh takes it as $6 and refuses an allocation that does not match.
    args_line = next(ln for ln in text.splitlines() if ln.startswith("arguments"))
    positional = args_line.split('"')[1].split(" '")[0].split()
    assert positional[-1] == "2", f"devices is not the 6th positional arg: {positional}"


def test_the_retired_gpus_flag_says_what_replaced_it(env, conf_dir, capsys):
    """`--gpus` is caught by name. Left to Hydra it is an override with a lexer error that
    names nothing useful, and it was in the README until today."""
    assert main(_args(conf_dir, "--gpus", "2")) == 2
    err = capsys.readouterr().err
    assert "--gpus was removed" in err and "launch.devices" in err


def test_a_model_that_does_not_satisfy_the_contract_is_refused(env, conf_dir, capsys):
    """Otherwise ``on_step_end`` fails at the end of the first step."""
    (conf_dir / "model" / "bad.yaml").write_text(
        "# @package _global_\nmodel:\n  _target_: torch.nn.Linear\n"
        "  in_features: 2\n  out_features: 2\n"
    )
    args = _args(conf_dir)
    args[args.index("model=toy")] = "model=bad"
    assert main(args) == 2
    assert "does not build" in capsys.readouterr().err


def test_a_bad_model_option_is_a_message_not_a_traceback(env, conf_dir, capsys):
    """`model=` gained a default on 2026-09-10, so omitting it no longer fails -- it composes
    `mae`. A *misspelled* option is the live path, and it must still be one sentence rather
    than a traceback through Hydra's internals."""
    args = _args(conf_dir)
    args[args.index("model=toy")] = "model=toyy"
    assert main(args) == 2
    err = capsys.readouterr().err
    assert "toyy" in err, "the message has to name what could not be found"
    assert "Traceback" not in err


def test_omitting_model_submits_the_default_objective(env, conf_dir, capsys):
    """The cost of the default: `wcfm submit run.name=x` queues an `mae` run rather than
    refusing. It is validated and built like any other, so this is exit 0, and the objective
    that trained is recoverable only from the recorded config."""
    args = [a for a in _args(conf_dir) if a != "model=toy"]
    assert main(args) == 0, capsys.readouterr().err


def test_missing_shared_libs_are_refused_with_the_rebuild_command(
    tmp_path, monkeypatch, conf_dir, capsys
):
    """The job installs nothing -- `getenv=False` leaves uv off the worker's PATH -- so the
    check belongs here, with the command that fixes it."""
    monkeypatch.setenv("WCFM_LIBS", str(tmp_path / "absent"))
    monkeypatch.setenv("WCFM_OUTPUT_BASE", str(tmp_path / "out"))
    assert main(_args(conf_dir)) == 2
    err = capsys.readouterr().err
    assert "--no-deps" in err, "without it the resolver drops a second torch"
    assert "lightning-fabric>=2.6,<3" in err, "never the umbrella `lightning`"


@pytest.mark.parametrize("absent", ["hydra", "omegaconf", "lightning_fabric"])
def test_a_missing_runtime_dependency_is_named_before_the_queue(
    tmp_path, monkeypatch, conf_dir, capsys, absent
):
    """**hydra and omegaconf are runtime dependencies of the job**, not test dependencies:
    `wcfm train` composes the config on the worker. They lived only in `wcfm-testlibs` until
    2026-09-09, so the first real training job -- cluster 2261, the Stage 3 smoke run -- passed
    every login-node check and then died on `import hydra` after taking a GPU slot. Nothing
    before Stage 3 could train, so nothing had ever exercised the path.

    Parametrised over all three so a future edit cannot drop one from the check while the
    other two keep the test green."""
    libs = tmp_path / "libs"
    for pkg in ("lightning_fabric", "hydra", "omegaconf"):
        if pkg != absent:
            (libs / pkg).mkdir(parents=True)
    monkeypatch.setenv("WCFM_LIBS", str(libs))
    monkeypatch.setenv("WCFM_OUTPUT_BASE", str(tmp_path / "out"))
    assert main(_args(conf_dir)) == 2
    err = capsys.readouterr().err
    assert absent in err, f"the refusal does not name {absent}"
    assert "--no-deps" in err


# ------------------------------------------------------------------ --smoke


def test_smoke_renames_the_run_and_shrinks_it(env, conf_dir, capsys):
    """The old ``submit.sh --smoke`` had to rewrite JSON; ``optim.epochs`` and
    ``data.n_subset`` are plain overrides now."""
    assert main(_args(conf_dir, "--smoke")) == 0
    assert "run:     run_a_smoke" in capsys.readouterr().out

    text = _sub(env, name="run_a_smoke")
    assert "'optim.epochs=2'" in text
    assert "'data.n_subset=2000'" in text
    assert "'run.save_every=1'" in text


def test_smoke_overrides_an_epochs_the_user_asked_for(env, conf_dir):
    """Otherwise Hydra takes the last value and a smoke run silently trains 100 epochs."""
    assert main(_args(conf_dir, "optim.epochs=100", "--smoke")) == 0
    text = _sub(env, name="run_a_smoke")
    assert "'optim.epochs=100'" not in text
    assert "'optim.epochs=2'" in text


def test_the_run_name_the_worker_composes_is_the_one_submit_validated(env, conf_dir):
    """**The bug that threw away cluster 2263's entire output.** ``--smoke`` renames the run by
    mutating ``cfg`` in this process; the worker recomposes from the forwarded overrides and so
    saw the original name. The engine writes ``<output_root>/<run.name>`` and trainjob.sh
    rsynced ``<output_root>/<$4>``, two different directories -- and the script pre-created the
    second, so rsync copied an empty tree, exited 0 and reported "Training complete!".

    The `.sub`'s 4th positional argument and its `run.name=` override must agree, always."""
    assert main(_args(conf_dir, "--smoke")) == 0
    text = _sub(env, name="run_a_smoke")
    arguments = next(ln for ln in text.splitlines() if ln.startswith("arguments"))
    positional = arguments.split('"')[1].split("'")[0].split()
    assert positional[3] == "run_a_smoke", positional
    assert "'run.name=run_a_smoke'" in text, "the worker would compose a different directory"
    assert text.count("run.name=") == 1, "a stale run.name= would make Hydra's last-wins a race"


def test_a_user_supplied_run_name_survives_to_the_worker(env, conf_dir):
    assert main(_args(conf_dir, "run.name=chosen")) == 0
    text = _sub(env, name="chosen")
    assert "'run.name=chosen'" in text and text.count("run.name=") == 1


def test_smoke_epochs_are_configurable(env, conf_dir, monkeypatch):
    monkeypatch.setenv("SMOKE_EPOCHS", "5")
    monkeypatch.setenv("SMOKE_NSUBSET", "64")
    assert main(_args(conf_dir, "--smoke")) == 0
    text = _sub(env, name="run_a_smoke")
    assert "'optim.epochs=5'" in text and "'data.n_subset=64'" in text


# ------------------------------------------------------------------ the job script


def test_the_job_script_exists_and_is_executable():
    script = REPO / "gridutils" / "train" / "trainjob.sh"
    assert script.is_file()
    import os

    assert os.access(script, os.X_OK), "Condor executes it directly"


def test_the_job_script_waits_for_the_trainer_before_syncing():
    """The one thing that changed from the old repo's trap. ``PreemptionGuard`` writes
    ``latest.pt`` on SIGTERM at the next step boundary, so a handler that rsynced on *receipt*
    would copy a checkpoint mid-write and ``--resume auto`` would fail on it."""
    text = (REPO / "gridutils" / "train" / "trainjob.sh").read_text()
    handler = text[text.index("on_term()") : text.index("trap on_term")]
    assert 'kill -TERM "$trainer_pid"' in handler
    assert 'wait "$trainer_pid"' in handler
    assert handler.index('wait "$trainer_pid"') < handler.index("sync_back"), (
        "sync_back must come after the wait, or it copies a half-written latest.pt"
    )


def test_the_job_script_is_valid_bash():
    import subprocess

    script = REPO / "gridutils" / "train" / "trainjob.sh"
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_the_job_script_installs_nothing():
    """`getenv=False` leaves uv off the worker's PATH, the cluster venv has no pip, and a
    --target install without --no-deps drops a second torch. Three dead submissions."""
    text = (REPO / "gridutils" / "train" / "trainjob.sh").read_text()
    lines = [
        ln
        for ln in text.splitlines()
        if ("pip install" in ln or "uv pip" in ln) and not ln.strip().startswith(("#", "echo"))
    ]
    assert not lines, f"the job script installs things: {lines}"

