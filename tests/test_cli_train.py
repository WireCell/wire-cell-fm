"""``wcfm train``: composition, validation, and the dry run.

Composition is exercised against a **copy** of the real ``conf/`` with one model option added,
rather than against a hand-built config. That is deliberate: ``conf/config.yaml`` sets
``model: ???`` and ``conf/model/`` is empty until Stage 3, so the only way to test the real
composition root today is to supply the missing group -- and doing it in a copy keeps a toy
out of the repo's own ``conf/``, where ``test_config.py``'s directory sweep would find it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning_fabric")

from wcfm.cli.__main__ import COMMANDS  # noqa: E402
from wcfm.cli.__main__ import main as dispatch  # noqa: E402
from wcfm.cli.train import _conf_dir, build_module, main, validate  # noqa: E402

pytestmark = pytest.mark.stack

REPO = Path(__file__).resolve().parents[1]

TOY_MODEL = """# @package _global_
model:
  _target_: tests.toy.ToyModule
  dim: 4
"""


@pytest.fixture
def conf_dir(tmp_path):
    """The real conf/ plus a model option, so composition is the production path.

    ``register_all()`` here, not only inside ``main()``: ``conf/config.yaml`` opens its
    defaults list with ``base_config``, which lives in the ``ConfigStore`` rather than in the
    tree, and the store is a **process-global singleton**. The tests that compose directly
    through ``initialize_config_dir`` were relying on an earlier test having called ``main()``
    in the same process -- so ``pytest -m needs_data`` on its own failed with "Could not load
    'base_config'" while the full run passed. A test that only passes in company is not a
    test; the fixture that hands over the composition root registers the schema behind it.
    """
    from wcfm.config.store import register_all

    register_all()
    target = tmp_path / "conf"
    shutil.copytree(REPO / "conf", target)
    (target / "model").mkdir(exist_ok=True)
    (target / "model" / "toy.yaml").write_text(TOY_MODEL)
    return target


def _argv(conf_dir, *overrides, dry=True):
    args = ["--config-dir", str(conf_dir), "model=toy", *overrides]
    return (["--dry-run"] if dry else []) + args


# ------------------------------------------------------------------ dispatch


def test_train_is_registered_as_a_subcommand():
    """Extending the existing entry point rather than adding a second console script."""
    assert COMMANDS["train"] == "wcfm.cli.train"


def test_help_is_reachable_without_hydra_or_torch(capsys):
    assert main(["--help"]) == 0
    assert "model=" in capsys.readouterr().out


def test_the_top_level_help_lists_train(capsys):
    assert dispatch([]) == 0
    assert "train" in capsys.readouterr().out


def test_an_unknown_subcommand_still_says_so(capsys):
    assert dispatch(["trian"]) == 2
    assert "unknown command" in capsys.readouterr().err


# ------------------------------------------------------------------ config dir


def test_the_default_config_dir_is_the_repos_conf_not_the_working_directory():
    """A Condor job's cwd is wherever ``trainjob.sh`` left it, and ``conf/`` is deliberately
    not package data -- so it is resolved from ``wcfm.__file__``."""
    resolved, rest = _conf_dir(["model=toy"])
    assert resolved == REPO / "conf"
    assert rest == ["model=toy"], "the overrides pass through untouched"


@pytest.mark.parametrize(
    "argv", [["--config-dir", "/somewhere"], ["--config-dir=/somewhere"]]
)
def test_both_config_dir_spellings_are_accepted(argv):
    resolved, rest = _conf_dir([*argv, "model=toy"])
    assert resolved == Path("/somewhere")
    assert rest == ["model=toy"]


def test_a_missing_config_dir_is_reported_rather_than_composed(capsys, tmp_path):
    assert main(["--config-dir", str(tmp_path / "nope"), "model=toy"]) == 2
    assert "no config directory" in capsys.readouterr().err


# ------------------------------------------------------------------ the dry run


def test_a_dry_run_constructs_the_module_and_exits_without_touching_a_device(
    conf_dir, capsys, tmp_path
):
    """The point of a dry run is to fail on a login node instead of after a queue wait."""
    code = main(_argv(conf_dir, "run.name=dry", f"run.output_root={tmp_path}"))
    assert code == 0

    out = capsys.readouterr().out
    assert "ToyModule constructed" in out
    assert "parameters" in out
    assert not (tmp_path / "dry").exists(), "a dry run writes no run directory"


def test_omitting_the_model_trains_the_default_objective(conf_dir, tmp_path, capsys):
    """``conf/config.yaml`` selects a default objective, so `wcfm train run.name=x` composes.

    It was ``model: ???`` until 2026-09-10, and this test pinned the refusal -- a
    ``ConfigCompositionException`` naming the key and listing the options. That decision was
    reversed; what is worth keeping is the CLI-level half of its replacement, which
    `test_model_config.py::test_the_default_objective_is_recorded_not_merely_inherited`
    checks at composition: the run starts, and an inherited objective is byte-identical to an
    explicit one, so the resolved config records what trained even when the command line
    does not.
    """
    code = main(["--dry-run", "--config-dir", str(conf_dir), f"run.output_root={tmp_path}",
                 "run.name=x"])
    assert code == 0, capsys.readouterr().err
    out = capsys.readouterr().out
    assert "dry run" in out and "parameters" in out


def test_a_model_option_that_does_not_exist_names_the_group(conf_dir, tmp_path, capsys):
    """A typo in ``model=`` must name the group it could not find, not fail later."""
    code = main(
        [
            "--dry-run",
            "--config-dir",
            str(conf_dir),
            "model=absent",
            "run.name=x",
            f"run.output_root={tmp_path}",
        ]
    )
    assert code == 2
    assert "model" in capsys.readouterr().err


def test_omitting_model_uses_the_default_objective(tmp_path, capsys):
    """`model=` was mandatory until 2026-09-10; `conf/config.yaml` now defaults it to `mae`.

    The cost of the default is that a forgotten `model=` is a silently wrong objective rather
    than an error, so the run's recorded config is the only place that says what trained --
    which is why this asserts the objective actually reached the module, not just exit 0.
    """
    assert main(["--dry-run", "run.name=x", f"run.output_root={tmp_path}"]) == 0
    out = capsys.readouterr().out
    assert "SslModule constructed" in out


def test_a_conf_tree_that_makes_model_mandatory_still_explains_itself(tmp_path, capsys):
    """`explain_missing_model`'s first branch: reachable through `--config-dir`, since another
    tree may keep `model: ???`."""
    conf = tmp_path / "conf"
    (conf / "model").mkdir(parents=True)
    (conf / "model" / "toy.yaml").write_text(TOY_MODEL)
    (conf / "config.yaml").write_text(
        "defaults:\n  - base_config\n  - model: ???\n  - _self_\n"
        "hydra:\n  job:\n    chdir: false\n  run:\n    dir: .\n  output_subdir: null\n"
    )
    code = main(["--dry-run", f"--config-dir={conf}", "run.name=x"])
    assert code == 2
    err = capsys.readouterr().err
    assert "must specify 'model'" in err and "'toy'" in err


# ------------------------------------------------------------------ validation


def _cfg(**over):
    from omegaconf import OmegaConf

    base = OmegaConf.create(
        {
            "run": {"name": "r", "output_root": "/tmp"},
            "data": {"global_batch_size": 100},
            "optim": {"accumulate_grad_batches": 1},
            "launch": {"devices": 2},
        }
    )
    for dotted, value in over.items():
        OmegaConf.update(base, dotted.replace("__", "."), value)
    return base


def test_an_empty_run_name_is_refused_before_anything_is_written():
    """Every C9 path is ``<output_root>/<name>/...``, so an unnamed run writes into
    ``<output_root>/`` and the next one overwrites it."""
    with pytest.raises(ValueError, match="run.name is empty"):
        validate(_cfg(run__name="  "))


def test_an_indivisible_batch_fails_on_the_login_node_not_inside_the_loader():
    """``build_loader`` would raise on every rank at once, several minutes into shard
    reading. 100 does not divide by 3.

    ``ValueError``: ``per_rank_batch_size`` refuses rather than asserts, so the refusal
    survives ``python -O``.
    """
    with pytest.raises(ValueError, match="not divisible"):
        validate(_cfg(launch__devices=3))


def test_zero_accumulation_is_refused_because_no_optimizer_step_would_ever_run():
    with pytest.raises(ValueError, match="accumulate_grad_batches"):
        validate(_cfg(optim__accumulate_grad_batches=0))


def test_a_valid_config_passes_quietly():
    assert validate(_cfg()) is None


# ------------------------------------------------------------------ the contract


def test_a_module_missing_a_contract_method_is_refused_before_the_run(tmp_path):
    """Otherwise ``on_step_end`` fails at the end of the first step, which on the sharded
    backend is several minutes of shard reading later."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"model": {"_target_": "torch.nn.Linear", "in_features": 2,
                                      "out_features": 2}})
    with pytest.raises(TypeError, match="does not satisfy TrainingModule"):
        build_module(cfg)


def test_the_toy_module_satisfies_the_contract():
    from omegaconf import OmegaConf

    module = build_module(OmegaConf.create({"model": {"_target_": "tests.toy.ToyModule"}}))
    assert type(module).__name__ == "ToyModule"


# ------------------------------------------------------------------ a real run


def test_a_real_short_run_writes_a_run_directory_and_a_metrics_stream(conf_dir, tmp_path):
    """End to end through the CLI: compose the real tree, build the module, train two epochs
    on the packed backend's loader replaced by nothing -- the loader comes from the config, so
    this uses `run()` with a loader the test supplies."""
    import json

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from wcfm.engine.trainer import Trainer

    from .toy import ToyModule, toy_loader

    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=toy",
                "metrics=full",
                "run.name=cli_run",
                f"run.output_root={tmp_path}",
                "run.num_workers=0",
                "run.precision=32-true",
                "run.resume=none",
                "optim.epochs=2",
                "metrics.step_cadence=1",
                "metrics.collectors.spectrum.cadence=1",
                "metrics.collectors.gradnorm.cadence=1",
                "metrics.collectors.arrays.cadence=1",
                "launch.devices=1",
                "data.global_batch_size=2",
            ],
        )
        OmegaConf.resolve(cfg)

        from lightning_fabric import Fabric

        result = Trainer(
            cfg,
            ToyModule(),
            fabric=Fabric(accelerator="cpu", devices=1, precision="32-true"),
            loader=toy_loader(steps=4),
            run_dir=tmp_path / "cli_run",
        ).fit()

    assert result["steps"] == 8
    metrics = tmp_path / "cli_run" / "metrics"
    rows = [json.loads(x) for x in (metrics / "step.jsonl").read_text().splitlines() if x.strip()]

    # The full collector set actually produced columns, under the names config gave them --
    # `spectrum` from conf/metrics/full.yaml, `toy/feat` from the module's own observables().
    assert any("spectrum/toy/feat/participation_ratio" in r for r in rows)
    assert any("spectrum/toy/feat/n_rows" in r for r in rows)
    # The 1-D `toy/centre` buffer is not a feature matrix, so Spectrum skips it rather than
    # emitting something meaningless for it.
    assert not any(k.startswith("spectrum/toy/centre") for r in rows for k in r)
    assert any(k.startswith("gradnorm/") for r in rows for k in r)
    assert any("throughput/samples_per_s" in r for r in rows)
    # ArrayDump wrote a file and the stream recorded a pointer to it, not the matrix.
    pointer = next(r["arrays/toy/feat/cov_file"] for r in rows if "arrays/toy/feat/cov_file" in r)
    assert (metrics / "arrays" / pointer).exists()

    schema = json.loads((metrics / "schema.json").read_text())
    assert "lr" in schema["streams"]["step"]


# ------------------------------------------------------------------ run(), for real

# Every test above either stops at `--dry-run` or builds `Trainer` directly with a toy loader,
# so `run()`'s one non-dry line -- `Trainer(cfg, module, argv=...).fit()`, which builds a loader
# from the config -- had never executed. This closes it against the real production: /gpfs01 is
# mounted on the login node and the sparse readers are pure IO, so it needs no GPU.

SHARD_DIR = Path("/gpfs01/lbne/users/fm/cffm-data/shards_fhdh_sparse_200k_mixed_apa0W")


@pytest.mark.needs_data
def test_run_trains_from_a_config_built_loader_against_real_shards(conf_dir, tmp_path):
    """The one line no other test reaches. `n_subset` keeps it to a couple of shards.

    The module is a toy, but the loader, the config, the schedules, the checkpoint and the
    metrics stream are all the production path -- which is the point: a `Batch` off the real
    reader is what the engine's explicit `.to(device)` and the collectors actually see.
    """
    import json

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from wcfm.cli.train import run

    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=toy",
                "data=prod_jay_200k_mixed_sharded",
                "run.name=real_run",
                f"run.output_root={tmp_path}",
                "run.num_workers=0",
                "run.precision=32-true",
                "run.resume=none",
                "run.save_every=1",
                "optim.epochs=1",
                "metrics.step_cadence=1",
                "launch.devices=1",
                "data.global_batch_size=4",
                "data.n_subset=400",
                "data.buffer_size=200",
            ],
        )
        OmegaConf.resolve(cfg)
        result = run(cfg, argv=["wcfm", "train"])

    assert "dry_run" not in result, "the training summary, not the dry-run summary"
    assert result["steps"] > 0, "the config-built loader yielded batches"
    assert result["epochs_run"] == 1
    assert not result["preempted"]

    run_dir = tmp_path / "real_run"
    assert (run_dir / "checkpoints" / "checkpoint_epoch1.pt").exists()

    rows = [
        json.loads(x)
        for x in (run_dir / "metrics" / "step.jsonl").read_text().splitlines()
        if x.strip()
    ]
    assert rows and all("lr" in r for r in rows)

    # epoch_len came from `len(loader)` on the real reader, so the schedule spans the run.
    derived = json.loads((run_dir / "run_metadata.json").read_text())["derived"]
    assert derived["epoch_len"] == result["steps"]
    assert derived["total_iters"] == result["steps"]


@pytest.mark.needs_data
def test_a_toy_model_still_gets_the_real_batch_shape(conf_dir, tmp_path):
    """The toy takes `x` off a `ToyBatch`; a real `Batch` carries `voxels` and `meta`. This
    asserts what the engine hands over, so the difference is visible here rather than as an
    AttributeError inside a module in Stage 3."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from wcfm.data.build import build_loader

    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=toy",
                "data=prod_jay_200k_mixed_sharded",
                f"run.output_root={tmp_path}",
                "run.name=shape",
                "data.global_batch_size=4",
                "data.n_subset=400",
                "data.buffer_size=200",
            ],
        )
        OmegaConf.resolve(cfg)

    batch = next(iter(build_loader(cfg.data, world_size=1, num_workers=0)))
    assert hasattr(batch, "voxels") and hasattr(batch, "meta")
    assert hasattr(batch, "to"), "the engine moves it explicitly; spike (b)"
