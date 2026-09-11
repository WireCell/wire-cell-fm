"""``wcfm test --gpu``: the submit, checked without a cluster.

What is testable here is everything up to ``condor_submit`` -- and that is most of the value,
because the failures this command exists to prevent are login-node failures. Three of the four
dead submissions this project has already paid for were visible before the job started:
``uv`` absent from the worker's PATH under ``getenv=False``, a second torch dropped by a
``--target`` install without ``--no-deps``, and an install attempted per node.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wcfm.cli.__main__ import COMMANDS
from wcfm.cli.test_gpu import main

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Shared library directories that exist, so the precondition check passes and the rest of
    the command can be exercised. Their *absence* is what the next test covers."""
    for name, sub in (("WCFM_LIBS", "lightning_fabric"), ("WCFM_TESTLIBS", "pytest")):
        root = tmp_path / name.lower()
        (root / sub).mkdir(parents=True)
        monkeypatch.setenv(name, str(root))
    monkeypatch.setenv("WCFM_PYENV", str(tmp_path / "uvenv"))
    monkeypatch.setenv("WCFM_OUTPUT_BASE", str(tmp_path / "out"))
    return tmp_path


def _sub_file(tmp_path: Path) -> Path:
    subs = list((tmp_path / "out").glob("*/*.sub"))
    assert len(subs) == 1, f"expected one submit file, found {subs}"
    return subs[0]


def test_test_is_registered_as_a_subcommand():
    assert COMMANDS["test"] == "wcfm.cli.test_gpu"


def test_help_is_reachable_and_names_the_cpu_alternative(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "--gpu" in out
    assert 'pytest -m "not gpu and not distributed"' in out, (
        "someone reaching for this command for the CPU suites must be redirected"
    )


def test_a_mode_is_required(capsys):
    """``wcfm test`` with no mode is almost certainly someone wanting the CPU suites."""
    assert main(["-k", "reduce"]) == 2
    err = capsys.readouterr().err
    assert "not gpu and not distributed" in err
    assert "--dist-cpu" in err, "the queue-free option must be offered"


def test_dist_cpu_is_offered_in_the_help_ahead_of_the_condor_path(capsys):
    """It is the one people should reach for first: same signal, no queue slot."""
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert out.index("--dist-cpu") < out.index("--gpu   "), (
        "the local option should be listed before the one that waits for a slot"
    )


def test_dist_cpu_refuses_unrecognised_arguments(capsys):
    """A silently dropped flag means a run that did not test what was asked for."""
    assert main(["--dist-cpu", "--gpus", "2"]) == 2
    assert "unrecognised arguments" in capsys.readouterr().err


def test_a_dry_run_writes_a_submit_file_and_submits_nothing(fake_env, capsys):
    assert main(["--gpu", "--dry-run"]) == 0
    text = _sub_file(fake_env).read_text()

    assert "request_gpus            = 2" in text
    assert "should_transfer_files   = YES" in text
    assert "getenv                  = False" in text
    assert 'transfer_output_files   = ""' in text, (
        "without it Condor copies everything new at the top of scratch back to initialdir"
    )
    assert 'arguments               = "repo.tgz ' in text
    assert str(REPO / "gridutils" / "test" / "test_gpu_job.sh") not in text, (
        "the executable is the staged copy; the checkout must not be named"
    )
    assert "dry run" in capsys.readouterr().out


def test_the_job_script_that_gets_staged_exists_and_is_executable():
    """It is copied beside the archive at submit time and Condor transfers it like any other
    input. A submit naming a path that is not there fails on the worker, minutes later."""
    script = REPO / "gridutils" / "test" / "test_gpu_job.sh"
    assert script.is_file()
    assert os.access(script, os.X_OK), "Condor executes it directly; it must be +x"


def test_missing_shared_libraries_are_refused_before_submitting(tmp_path, monkeypatch, capsys):
    """**The job installs nothing.** ``getenv=False`` leaves ``~/.local/bin`` off the worker's
    PATH so there is no ``uv`` there, and the cluster venv has no pip -- an earlier submission
    died on exactly that in six seconds. So the check belongs here, with the command that
    fixes it."""
    monkeypatch.setenv("WCFM_LIBS", str(tmp_path / "nope"))
    monkeypatch.setenv("WCFM_TESTLIBS", str(tmp_path / "also-nope"))
    monkeypatch.setenv("WCFM_OUTPUT_BASE", str(tmp_path / "out"))

    assert main(["--gpu", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert "no lightning-fabric" in err and "no pytest" in err
    assert "--no-deps" in err, (
        "the rebuild command must carry --no-deps: without it the resolver drops a second "
        "torch into the target, which shadows the pinned 2.10.0+cu128 on PYTHONPATH"
    )
    assert "lightning-fabric>=2.6,<3" in err, "never the umbrella `lightning`"


def test_fewer_than_two_gpus_warns_that_the_distributed_suite_becomes_vacuous(
    fake_env, capsys
):
    """Said in three places -- here, the job script, and the ``two_rank_fabric`` fixture --
    because a green distributed suite that ran on one rank is the worst outcome available."""
    assert main(["--gpu", "--dry-run", "--gpus", "1"]) == 0
    err = capsys.readouterr().err
    assert "one rank always agrees with itself" in err
    assert "request_gpus            = 1" in _sub_file(fake_env).read_text()


def test_a_k_expression_is_forwarded_to_both_stages(fake_env):
    assert main(["--gpu", "--dry-run", "-k", "reduce"]) == 0
    assert "'reduce'" in _sub_file(fake_env).read_text()


def test_unrecognised_arguments_are_refused_rather_than_ignored(fake_env, capsys):
    """A silently dropped flag means a job that did not run what was asked for."""
    assert main(["--gpu", "--dry-run", "--nproc", "4"]) == 2
    assert "unrecognised arguments" in capsys.readouterr().err


def test_the_output_directory_is_created_so_condor_can_write_its_log(fake_env):
    assert main(["--gpu", "--dry-run"]) == 0
    out_dir = _sub_file(fake_env).parent
    assert out_dir.is_dir()
    assert out_dir.name.startswith("wcfm_test_gpu_")


# ------------------------------------------------------------------ the suites exist


def test_the_distributed_marker_does_not_require_cuda():
    """The change that took this suite off the 2-GPU queue. ``pyproject.toml``'s marker
    description and ``conftest``'s skip must both be about ranks, not devices -- the first
    version said "needs two or more CUDA devices" and the file carried a ``gpu`` mark, which
    together meant a correctness fix waited ninety minutes for a slot it did not need."""
    assert "needs 2+ ranks" in (REPO / "pyproject.toml").read_text()

    conftest = (REPO / "tests" / "conftest.py").read_text()
    assert 'if "distributed" in item.keywords and world < 2' in conftest

    suite = (REPO / "tests" / "test_engine_distributed.py").read_text()
    assert "pytestmark = pytest.mark.distributed" in suite, (
        "a `gpu` mark here would re-impose the CUDA skip and undo this"
    )


def test_the_distributed_suite_has_a_control():
    """Spike (c)'s gate, applied to this suite: a test that computes through the unwrapped
    module must diverge, or every other assertion in the file proves nothing -- the two
    spellings differ only in whether DDP was armed."""
    suite = (REPO / "tests" / "test_engine_distributed.py").read_text()
    assert "test_the_control_diverges_so_this_suite_can_detect_its_own_failure" in suite


def test_both_markers_actually_select_tests():
    """The point of the whole command: a marker with no tests behind it is a trigger for
    nothing, which is how the old repo's eight hand-submitted `.sub` files ended up unrun."""
    import subprocess
    import sys

    for marker in ("gpu", "distributed"):
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "-m", marker, "--collect-only", "-q"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert "no tests ran" not in out.stdout.lower(), (
            f"marker {marker!r} selects nothing:\n{out.stdout[-2000:]}"
        )
        assert " selected" in out.stdout or "tests collected" in out.stdout, out.stdout[-2000:]


def test_the_markers_are_declared_in_pyproject():
    """An undeclared marker is a warning, and under ``-W error`` a failure."""
    text = (REPO / "pyproject.toml").read_text()
    for marker in ("gpu:", "distributed:", "needs_data:", "stack:"):
        assert marker in text, f"marker {marker!r} is used but not declared"
