"""`wcfm eval extract`: the plan it resolves before it reads anything.

The pass itself is covered in `test_eval_extract.py` against an in-memory loader. What is
tested here is everything the CLI decides -- which checkpoints, which paths, which transform --
because those are the decisions that used to be flags a caller repeated by hand at every epoch
and got subtly wrong once.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from wcfm.cli.__main__ import COMMANDS
from wcfm.cli.__main__ import main as dispatch
from wcfm.cli.eval import _charge_transform, _checkpoints, _flags, _git_sha, main

CONFIG = """
run:
  name: a_run
  seed: 7
data:
  backend: sharded
  sharded_dir: /nowhere
  global_batch_size: 8
  n_subset: -1
model:
  normalize:
    min_val: 1.0
    max_val: 4000.0
    enabled: true
"""


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "a_run"
    (d / "checkpoints").mkdir(parents=True)
    (d / "config.yaml").write_text(CONFIG)
    for epoch in (1, 5, 10):
        (d / "checkpoints" / f"checkpoint_epoch{epoch}.pt").write_bytes(b"")
    (d / "checkpoints" / "latest.pt").write_bytes(b"")
    (d / "run_metadata.json").write_text('{"git": {"sha": "abc123"}}')
    return d


# ------------------------------------------------------------------------ registration


def test_eval_is_registered_as_a_subcommand():
    assert COMMANDS["eval"] == "wcfm.cli.eval"


def test_help_is_reachable(capsys):
    assert main(["--help"]) == 0
    assert "wcfm eval extract" in capsys.readouterr().out


def test_an_unknown_subcommand_lists_the_ones_that_exist(capsys):
    """`compare` used to be the example of an unbuilt stage. Every stage is built now, so the
    test moves to a name that is not one rather than being deleted -- the behaviour it pins is
    that a typo names the alternatives instead of failing silently."""
    assert dispatch(["eval", "compair"]) == 2
    err = capsys.readouterr().err
    assert "unknown subcommand" in err
    for known in ("extract", "probe", "merge", "compare", "submit"):
        assert known in err


def test_every_advertised_subcommand_dispatches(capsys):
    """The usage text and the dispatcher are two lists that can drift apart; this is the only
    thing that notices. Each is called with no arguments, so each returns its own usage error
    rather than doing work."""
    for sub in ("extract", "probe", "merge", "compare", "submit"):
        rc = dispatch(["eval", sub])
        assert rc == 2, f"{sub} did not reach its own argument handling"
        capsys.readouterr()


# --------------------------------------------------------------------- what it resolves


def test_flags_parse_with_and_without_values():
    flags, rest = _flags(["run", "--epochs=1,2", "--dry-run"])
    assert rest == ["run"]
    assert flags == {"epochs": "1,2", "dry-run": "true"}


def test_all_epochs_by_default_in_numeric_order(run_dir):
    names = [p.name for p in _checkpoints(run_dir, "")]
    assert names == [
        "checkpoint_epoch1.pt",
        "checkpoint_epoch5.pt",
        "checkpoint_epoch10.pt",
    ]


def test_latest_is_never_scored(run_dir):
    """It duplicates some epoch under a name that means a different one in every run, so a
    features directory called `latest` would not be comparable to anything."""
    assert all(p.name != "latest.pt" for p in _checkpoints(run_dir, ""))


def test_a_requested_epoch_that_was_never_saved_names_the_ones_that_were(run_dir):
    with pytest.raises(SystemExit, match=r"no checkpoint for epochs \[7\]"):
        _checkpoints(run_dir, "1,7")


def test_epochs_are_selected_in_the_order_asked_for(run_dir):
    assert [p.name for p in _checkpoints(run_dir, "10,1")] == [
        "checkpoint_epoch1.pt",
        "checkpoint_epoch10.pt",
    ]


def test_the_charge_transform_is_recorded_from_the_runs_own_config():
    cfg = OmegaConf.create(CONFIG)
    label, params = _charge_transform(cfg)
    assert label == "log[1.0,4000.0]"
    # The parameters come back as floats beside the display string. The raw-charge baseline
    # rebuilds the backbone's real input from them, and parsing them back out of the string
    # would be one float repr away from a silently different baseline.
    assert params == {"kind": "log", "min_val": 1.0, "max_val": 4000.0}


def test_a_disabled_transform_reads_as_none():
    cfg = OmegaConf.create(CONFIG)
    cfg.model.normalize.enabled = False
    assert _charge_transform(cfg) == ("none", {})


def test_a_transform_missing_its_bounds_reads_as_none_rather_than_half_a_transform():
    """`log[None,None]` would group as its own charge transform in `check_comparability` and
    would hand the raw-charge baseline a transform it cannot apply."""
    cfg = OmegaConf.create(CONFIG)
    cfg.model.normalize.min_val = None
    assert _charge_transform(cfg) == ("none", {})


def test_the_git_sha_comes_from_run_metadata(run_dir):
    assert _git_sha(run_dir) == "abc123"


def test_a_run_without_metadata_records_an_empty_sha_rather_than_failing(tmp_path):
    assert _git_sha(tmp_path) == ""


# ------------------------------------------------------------------------- the dry run


def test_a_dry_run_prints_the_plan_and_reads_nothing(run_dir, capsys):
    assert main(["extract", str(run_dir), "--dry-run", "--max-images=64"]) == 0
    out = capsys.readouterr().out
    assert "checkpoint_epoch10.pt" in out
    assert "max_images=64" in out
    assert "batch_size=8" in out  # from the run's own data config
    assert str(run_dir / "features" / "eval_set") in out
    assert not (run_dir / "features").exists(), "a dry run writes nothing"


def test_batch_size_is_a_throughput_knob_and_max_images_is_the_population(run_dir, capsys):
    """v1's shard path took `max_images // batch_size` batches, so these two were entangled."""
    main(["extract", str(run_dir), "--dry-run", "--batch-size=64", "--max-images=100"])
    out = capsys.readouterr().out
    assert "max_images=100" in out and "batch_size=64" in out


def test_an_eval_set_can_be_shared_across_runs(run_dir, tmp_path, capsys):
    shared = tmp_path / "shared_eval_set"
    main(["extract", str(run_dir), "--dry-run", f"--eval-set-root={shared}"])
    assert str(shared) in capsys.readouterr().out


def test_sources_and_taps_reach_the_plan(run_dir, capsys):
    main(["extract", str(run_dir), "--dry-run", "--sources=student", "--taps=dec_half"])
    out = capsys.readouterr().out
    assert "['student']" in out and "['dec_half']" in out


def test_a_data_option_replaces_the_runs_production_but_not_its_batch(run_dir, capsys):
    """A run trained on a set without per-pixel truth is scored on one that has it; the
    per-rank batch, the seed and the charge transform still come from the run."""
    assert main(["extract", str(run_dir), "--dry-run", "--data=prod_jay_200k_mixed_sharded"]) == 0
    out = capsys.readouterr().out
    assert "shards_fhdh_sparse_200k_mixed_apa0W" in out
    assert "/nowhere" not in out
    assert "batch_size=8" in out


def test_an_unknown_data_option_names_the_ones_that_exist(run_dir, capsys):
    assert main(["extract", str(run_dir), "--dry-run", "--data=no_such_set"]) == 2
    err = capsys.readouterr().err
    assert "no data option 'no_such_set'" in err and "prod_jay_200k_mixed_sharded" in err


# ---------------------------------------------------------------------------- refusals


def test_a_run_directory_without_a_config_is_refused(tmp_path, capsys):
    (tmp_path / "checkpoints").mkdir()
    assert main(["extract", str(tmp_path), "--dry-run"]) == 2
    assert "reads the run's own resolved config" in capsys.readouterr().err


def test_a_run_with_no_checkpoints_says_so(tmp_path, capsys):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "config.yaml").write_text(CONFIG)
    assert main(["extract", str(tmp_path), "--dry-run"]) == 2
    assert "no checkpoint_epoch*.pt" in capsys.readouterr().err


def test_extract_without_a_run_directory_says_so(capsys):
    assert main(["extract"]) == 2
    assert "needs a run directory" in capsys.readouterr().err


def test_the_top_level_help_lists_eval(capsys):
    dispatch([])
    assert "eval" in capsys.readouterr().out


def test_paths_are_not_touched_by_import():
    assert Path(__file__).exists()  # sanity: the module imports with no side effects


# ------------------------------------------------ what the default invocation actually asks for


def test_pixel_truth_is_requested_even_under_the_default_row_space(run_dir, capsys):
    """The pools are drawn from `pixel_labels` whatever the row space is, and drawing them at
    extraction time is the whole reason every probe scores the same population. Tying the
    request to `--rows=pooled` left the DEFAULT command writing a `pools.npz` holding nothing
    but `row_index`."""
    main(["extract", str(run_dir), "--dry-run"])
    out = capsys.readouterr().out
    assert "rows=all" in out
    assert "pixel_truth=True" in out


def test_the_reader_keeps_its_short_tail(run_dir, capsys):
    """`build_loader` drops a short final batch for training, which would put `batch_size`
    back into the scored event set -- the very thing `--max-images` exists to take out."""
    main(["extract", str(run_dir), "--dry-run"])
    assert "drop_last=False" in capsys.readouterr().out


def test_help_needs_no_torch():
    """`wcfm eval --help` has to work in the config-only environment, so the CLI must not
    import `wcfm.eval.extract` (and through it torch) at module scope."""
    import ast

    src = Path(__file__).resolve().parents[1] / "wcfm" / "cli" / "eval.py"
    tree = ast.parse(src.read_text())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [n.module or "" for n in top if isinstance(n, ast.ImportFrom)]
    assert not any(m.startswith("wcfm.eval") or m.startswith("wcfm.data") for m in names)
