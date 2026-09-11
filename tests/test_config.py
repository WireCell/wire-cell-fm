"""Composition, typing and the two silent traps.

Neither trap announces itself. A missing ``# @package _global_`` turns trailing overrides
into a literal key with dots in its name that never applies; ``hydra.job.chdir`` defaulting
to true moves the process out from under the Condor job's rsync layout. Both are one line in
a YAML file and both are pinned here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf, ValidationError

from wcfm.config.schema import FRAMEWORK_GROUPS, Config
from wcfm.config.store import ENTRY_POINT_GROUP, register_framework, register_plugins

CONF = Path(__file__).resolve().parents[1] / "conf"
PACKAGE_GLOBAL = "# @package _global_"


# The framework's own tests exercise the model axis with a stub registered into the same
# group: a model group entry needs nothing from the framework but a name. The real presets
# are tested in tests/test_model_config.py.
STUB_MODEL = {"name": "_stub", "terms": {"stub": {"weight": 1.0}}}


@pytest.fixture
def hydra_conf():
    # Never clear the repo: hydra's own `hydra/config` node lives in it too, and clearing it
    # makes every compose fail with a confusing MissingConfigException. Registration is
    # idempotent, so re-registering is the right way to be sure.
    GlobalHydra.instance().clear()
    register_framework()
    ConfigStore.instance().store(group="model", name="_stub", node=STUB_MODEL)
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        yield
    GlobalHydra.instance().clear()


# ------------------------------------------------------------------------------ the traps


@pytest.mark.parametrize("group", ["model", "experiment"])
def test_model_and_experiment_files_declare_package_global(group: str):
    """Absolute-path defaults entries land correctly without the header, so the file looks
    like it works; only the trailing overrides silently vanish.

    Top-level files only. A sub-group file (``conf/model/backbone/attn_mae.yaml``) lands at its
    group path by design and must NOT carry the header; ``tests/test_model_config.py`` pins
    that half.
    """
    offenders = [
        p.relative_to(CONF)
        for p in (CONF / group).glob("*.yaml")
        if not p.read_text().lstrip().startswith(PACKAGE_GLOBAL)
    ]
    assert not offenders, f"missing '{PACKAGE_GLOBAL}' as the first line: {offenders}"


def test_hydra_does_not_change_the_working_directory(hydra_conf):
    """The Condor job manages its own cwd and rsync layout."""
    cfg = compose(config_name="config", overrides=["model=_stub"], return_hydra_config=True)
    assert cfg.hydra.job.chdir is False


def test_config_yaml_pins_chdir_in_the_file_itself():
    text = (CONF / "config.yaml").read_text()
    assert "chdir: false" in text, "the value must be in the file, not only in a default"


# ----------------------------------------------------------------------------- composition


def test_config_yaml_names_a_default_objective_in_the_file_itself():  # noqa: D401
    """`model: mae` as of 2026-09-10, in the file rather than only in a schema default.

    It was `model: ???` -- mandatory -- and the reason was concrete: inheriting an objective
    is exactly how 21 archived configs came to be recorded as `hybrid` while their filenames
    said `mae`. The default reintroduces that failure mode, so the mitigation is that the
    resolved config records the objective either way; `test_model_config.py` pins that an
    inherited `mae` is indistinguishable from an explicit one.

    Text-level on purpose: this suite registers no model plugin, so it cannot compose a real
    preset -- and a default objective that lived only in the schema would be exactly the
    invisible inheritance the note above is about.
    """
    text = (CONF / "config.yaml").read_text()
    assert "- model: mae" in text
    assert "???" not in text, "a mandatory model would contradict test_model_config.py"


def test_framework_axes_compose(hydra_conf):
    cfg = compose(config_name="config", overrides=["model=_stub"])
    assert cfg.run.seed == 42
    assert cfg.optim.epochs == 100
    assert cfg.data.backend == "sharded", "the default production is the sharded 200k mixed"
    assert cfg.launch.find_unused_parameters is True
    assert cfg.metrics.step_cadence == 100


def test_group_selection_and_dotted_overrides(hydra_conf):
    cfg = compose(
        config_name="config",
        overrides=["model=_stub", "data=prod_jay_200k_mixed_sharded", "launch=multi_2gpu",
                   "optim.lr=3e-4"],
    )
    assert cfg.data.backend == "sharded"
    assert cfg.launch.devices == 2
    assert cfg.optim.lr == pytest.approx(3e-4)


def test_the_schema_is_load_bearing_not_decorative(hydra_conf):
    """A typo in a value type is caught at compose time -- which is submit time -- rather
    than an hour into a job."""
    # Hydra wraps the underlying omegaconf ValidationError; both are checked so a future
    # hydra that stops wrapping does not turn this test green for the wrong reason.
    with pytest.raises((ValidationError, ConfigCompositionException), match="optim.epochs"):
        compose(config_name="config", overrides=["model=_stub", "optim.epochs=ninety"])


def test_schedules_interpolate_against_the_optimizer(hydra_conf):
    cfg = compose(config_name="config", overrides=["model=_stub", "optim.lr=5e-4"])
    assert cfg.optim.schedules.lr.base_value == pytest.approx(5e-4)
    assert cfg.optim.schedules.lr.final_value == cfg.optim.min_lr
    assert cfg.optim.schedules.lr._target_.startswith("wcfm.engine.optim.")


# ------------------------------------------------------------------------------ the store


def test_every_framework_axis_is_registered_as_its_own_group():
    """Per-group registration, never a union: OmegaConf cannot represent a Union of
    dataclasses, so the earlier draft's `objective: A | B | C` would never have registered."""
    GlobalHydra.instance().clear()
    register_framework()
    repo = ConfigStore.instance().repo
    assert "base_config.yaml" in repo
    for group, _node in FRAMEWORK_GROUPS:
        assert group in repo and "base.yaml" in repo[group], group


def test_model_schemas_arrive_by_entry_point_so_the_framework_never_names_them():
    """The import-graph rule is satisfied by construction rather than by care: model group
    schemas are contributed through the `wcfm.config_schemas` entry point, so no framework
    module mentions `wcfm.model`. Stage 3 registers `model`; `tests/test_model_config.py`
    asserts it is found."""
    assert ENTRY_POINT_GROUP == "wcfm.config_schemas"
    assert isinstance(register_plugins(), list)


def test_model_is_untyped_on_purpose():
    node = OmegaConf.structured(Config)
    assert OmegaConf.is_missing(node, "model")
    # Any shape at all: the framework never learns what a model is.
    node.model = {"terms": {"dino": {"weight": 1.0}}}
    assert node.model.terms.dino.weight == 1.0


def test_the_metrics_presets_and_the_sharded_production_compose(hydra_conf):
    """Every metrics preset uses a same-group relative default (`- minimal`), which is its own
    small trap; and `backend: sharded` with an empty `sharded_dir` composes clean and fails at
    first read, so the pairing is checked here rather than on the cluster.

    The cadence assertion is on `default`, not on `full`. Until 2026-09-10 there was no
    middle preset and this pinned `termgrad` off in `full`; `default` is now the everyday
    choice and is where "does not silently cost model time" has to hold. `full` is free to
    enable whatever it likes -- that is what distinguishes it -- so nothing here pins its
    cadences and retuning them does not break this test.
    """
    cfgs = {
        name: compose(
            config_name="config",
            overrides=["model=_stub", f"metrics={name}", "data=prod_jay_200k_mixed_sharded"],
        )
        for name in ("minimal", "default", "full")
    }
    assert set(cfgs["minimal"].metrics.collectors) == {"throughput"}
    for name in ("default", "full"):
        assert set(cfgs[name].metrics.collectors) == {
            "throughput",
            "gradnorm",
            "spectrum",
            "arrays",
            "termgrad",
        }, name

    # `arrays` writes O(D^2) bytes per firing and `termgrad` makes the MODEL do extra work
    # (~1.5x a step). Both must be configured in `default` -- so their meaning is documented
    # and enabling them is one override -- and both must be off.
    everyday = cfgs["default"].metrics.collectors
    assert int(everyday.arrays.cadence) == 0, "`default` must not write O(D^2) arrays"
    assert int(everyday.termgrad.cadence) == 0, "`default` must not make the model do extra work"

    assert cfgs["full"].data.backend == "sharded"
    assert cfgs["full"].data.sharded_dir, "a sharded production names its shard directory"
    assert not cfgs["full"].data.datadir, (
        "a sharded production leaves datadir empty: it has no direct-read tree, and a path "
        "carried over from another production would be a lie in the recorded config"
    )


def test_a_schedule_is_a_selectable_group(hydra_conf):
    """Swapping cosine for a linear decay is a config entry naming a class, like every other
    component."""
    from hydra.core.config_store import ConfigStore as _CS

    assert "cosine.yaml" in _CS.instance().repo["optim"]["schedule"]
