"""``wcfm datagen``: the CPU submit for the dataset builders, checked without a cluster.

Everything up to ``condor_submit`` is testable here, and that is where the failures this command
prevents would be visible: a module name that does not exist, a venv that is not there, a
checkout named in the ``.sub`` instead of the staged copy.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from wcfm.cli.__main__ import COMMANDS
from wcfm.cli.datagen import main

REPO = Path(__file__).resolve().parents[1]

SHARD_ARGS = [
    "create_shards",
    "--datadir",
    "/a",
    "/b",
    "--apa",
    "0",
    "--outdir",
    "/s",
    "--shard_size",
    "4000",
]


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    pyenv = tmp_path / "uvenv"
    (pyenv / "bin").mkdir(parents=True)
    (pyenv / "bin" / "python").touch()
    monkeypatch.setenv("WCFM_PYENV", str(pyenv))
    monkeypatch.setenv("WCFM_OUTPUT_BASE", str(tmp_path / "out"))
    return tmp_path


def _sub_file(tmp_path: Path) -> Path:
    subs = list((tmp_path / "out" / "datagen").glob("*/*.sub"))
    assert len(subs) == 1, f"expected one submit file, found {subs}"
    return subs[0]


def test_datagen_is_registered_as_a_subcommand():
    assert COMMANDS["datagen"] == "wcfm.cli.datagen"


def test_help_names_both_builders(capsys):
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "create_shards" in out and "pack_dataset" in out


def test_a_dry_run_writes_a_cpu_submit_file_and_passes_the_arguments_through(fake_env, capsys):
    assert main(["--dry-run", "shards_2M", *SHARD_ARGS]) == 0
    sub = _sub_file(fake_env)
    text = sub.read_text()

    assert sub.parent.name == "shards_2M"
    assert "request_gpus" not in text, "a dataset builder is a CPU job"
    assert "Requirements" not in text, "no GPU node requirement on a CPU job"
    assert "should_transfer_files   = YES" in text
    assert "getenv                  = False" in text
    assert 'transfer_output_files   = ""' in text
    assert "wcfm.data.prep.create_shards '--datadir' '/a' '/b'" in text, (
        "the module's argv must reach the worker quoted and in order"
    )
    assert 'arguments               = "repo.tgz ' in text
    assert str(REPO / "gridutils" / "datagen" / "datajob.sh") not in text, (
        "the executable is the staged copy; the checkout must not be named"
    )
    assert "dry run" in capsys.readouterr().out


def test_resources_come_from_the_environment(fake_env, monkeypatch):
    monkeypatch.setenv("WCFM_REQUEST_MEMORY", "64000")
    monkeypatch.setenv("WCFM_REQUEST_CPUS", "16")
    assert main(["--dry-run", "pack", "pack_dataset", "--datadir", "/a"]) == 0
    text = _sub_file(fake_env).read_text()
    assert "request_memory          = 64000" in text
    assert "request_cpus            = 16" in text


def test_a_module_that_is_not_a_prep_tool_is_refused(fake_env, capsys):
    assert main(["--dry-run", "x", "train"]) == 2
    err = capsys.readouterr().err
    assert "wcfm.data.prep.train" in err
    assert "create_shards" in err, "the refusal must name what is accepted"


def test_a_job_name_and_a_module_are_both_required(fake_env, capsys):
    assert main(["--dry-run", "only_a_name"]) == 2
    assert "required" in capsys.readouterr().err


def test_a_missing_venv_is_refused_before_submitting(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WCFM_PYENV", str(tmp_path / "nope"))
    monkeypatch.setenv("WCFM_OUTPUT_BASE", str(tmp_path / "out"))
    assert main(["--dry-run", "x", *SHARD_ARGS]) == 2
    err = capsys.readouterr().err
    assert "no venv" in err and "build_env.sh" in err


def test_the_job_scripts_exist_and_are_executable():
    for name in ("datajob.sh", "unpack_apa.sh"):
        script = REPO / "gridutils" / "datagen" / name
        assert script.is_file()
        assert os.access(script, os.X_OK), f"{name} is run directly; it must be +x"
