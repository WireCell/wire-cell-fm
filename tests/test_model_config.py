"""The model axis composes, is typed, and takes the plan's overrides -- in the config-only
environment, with no torch. The schema arrives through the ``wcfm.config_schemas`` entry
point, so this is also the test that the plugin is discoverable at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf, ValidationError

from wcfm.config.store import register_all, register_plugins

CONF = Path(__file__).resolve().parents[1] / "conf"
PRESETS = sorted(p.stem for p in (CONF / "model").glob("*.yaml"))
EXPERIMENTS = sorted(p.stem for p in (CONF / "experiment").glob("*.yaml"))


@pytest.fixture
def hydra_all():
    GlobalHydra.instance().clear()
    register_all()
    with initialize_config_dir(config_dir=str(CONF), version_base="1.3"):
        yield
    GlobalHydra.instance().clear()


def test_the_model_schema_plugin_is_discoverable():
    """If this fails, `wire_cell_fm.egg-info/` next to `wcfm/` is missing or stale: run
    `uv pip install -e . --no-deps` in the checkout. An rsync that excludes `*.egg-info`
    produces exactly this."""
    assert "model" in register_plugins()


def test_the_shipped_presets():
    assert PRESETS == ["dino", "hybrid", "kd", "mae"], (
        "`mae` (charge + occupancy, no teacher) arrived with the occupancy term in Stage 4. "
        "`dino_recon` (dino + charge) was dropped on 2026-09-10: it matched no archived run "
        "and nothing had trained it. Adding a charge term to any preset is still one "
        "override -- see test_a_charge_term_can_be_added_to_a_teacher_preset_and_removed_again. "
        "`kd` (distill alone, no augment, no teacher) arrived with the distill term."
    )


def test_the_default_objective_is_recorded_not_merely_inherited(hydra_all):
    """`conf/config.yaml` selects `model: mae`, so `wcfm train run.name=x` composes.

    It was `model: ???` until 2026-09-10, and the reason was concrete: inheriting an objective
    is how 21 archived configs came to be recorded as `hybrid` while their filenames said
    `mae`. What makes the default survivable is that inheritance leaves no trace in the
    result -- an inherited `mae` is byte-identical to an explicit one, so the resolved tree in
    `run_config.json` says what trained even when the command line does not.
    """
    inherited = compose(config_name="config", overrides=["run.name=t"])
    explicit = compose(config_name="config", overrides=["model=mae", "run.name=t"])
    assert OmegaConf.to_container(inherited.model) == OmegaConf.to_container(explicit.model)
    assert set(inherited.model.terms) == {"charge", "occupancy"}
    assert inherited.model.teacher._target_.endswith("NoTeacher"), "mae has no teacher"


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_composes_with_the_ssl_module_target(hydra_all, preset):
    cfg = compose(config_name="config", overrides=[f"model={preset}", "run.name=t"])
    assert cfg.model._target_ == "wcfm.model.modules.SslModule"
    assert cfg.model.backbone._target_ == "wcfm.model.backbones.MinkUNetAttention"
    assert cfg.model.terms, "a preset with no term trains nothing"


@pytest.mark.parametrize("preset", PRESETS)
def test_a_teacher_is_present_exactly_when_a_term_distils(hydra_all, preset):
    """`SslModule.validate` refuses both halves of the mismatch -- a term wanting a teacher
    that is not there, and a teacher no term reads, which would let `--source=teacher`
    extraction return features from initialisation weights. The presets must agree with it."""
    cfg = compose(config_name="config", overrides=[f"model={preset}", "run.name=t"])
    distils = "dino" in cfg.model.terms
    expected = "EmaTeacher" if distils else "NoTeacher"
    assert cfg.model.teacher._target_ == f"wcfm.model.modules.{expected}"


def test_mae_is_charge_plus_occupancy_and_injects_both_roles(hydra_all):
    """The one preset where two injection roles are live at once: `masked` for the charge
    term at full resolution, `candidate` for the occupancy question at half."""
    cfg = compose(config_name="config", overrides=["model=mae", "run.name=t"])
    assert set(cfg.model.terms) == {"charge", "occupancy"}
    assert list(cfg.model.backbone.inject_roles) == ["masked", "candidate"]
    assert cfg.model.augment.masker.build_candidates is True
    assert cfg.model.augment.masker._target_ == "wcfm.model.augment.RegionMasker"
    # The term refuses to run on an uncapped candidate set; the preset must not need an
    # override to be usable.
    assert cfg.model.augment.masker.neg_per_pos


def test_kd_is_the_distill_term_alone_on_the_whole_image(hydra_all):
    """The student and the teacher encode the same pixels, so there is no cropper and no
    masker. `checkpoint` stays MISSING: the preset cannot name a teacher, and a bare
    `model=kd` is refused at compose rather than trained against nothing."""
    cfg = compose(config_name="config", overrides=["model=kd", "run.name=t"])
    assert list(cfg.model.terms) == ["distill"]
    assert cfg.model.augment.cropper is None
    assert cfg.model.augment.masker is None
    assert cfg.model.teacher._target_ == "wcfm.model.modules.NoTeacher"
    assert OmegaConf.is_missing(cfg.model.terms.distill, "checkpoint")


def test_dino_and_hybrid_differ_only_in_score_injected(hydra_all):
    dino = compose(config_name="config", overrides=["model=dino", "run.name=t"])
    hybrid = compose(config_name="config", overrides=["model=hybrid", "run.name=t"])
    assert dino.model.terms.dino.score_injected is False
    assert hybrid.model.terms.dino.score_injected is True
    d, h = OmegaConf.to_container(dino.model), OmegaConf.to_container(hybrid.model)
    d["terms"]["dino"].pop("score_injected")
    h["terms"]["dino"].pop("score_injected")
    assert d == h, "ADR 0005: hybrid is dino with injection, nothing else"


def test_a_charge_term_can_be_added_to_a_teacher_preset_and_removed_again(hydra_all):
    """What the deleted `dino_recon` preset was: hybrid plus a charge term. It is two
    overrides, which is why the preset did not need to exist."""
    add = ["model=hybrid", "+model/term@model.terms.charge=charge", "run.name=t"]
    cfg = compose(config_name="config", overrides=add)
    assert list(cfg.model.terms) == ["dino", "charge"]
    # 0.1 comes from ChargeTermConfig itself, so the override reproduces `dino_recon` exactly
    # -- the preset's own `charge: {weight: 0.1}` was restating the schema default.
    assert cfg.model.terms.charge.weight == pytest.approx(0.1)
    cfg = compose(config_name="config", overrides=[*add, "~model.terms.charge"])
    assert list(cfg.model.terms) == ["dino"]


def test_a_term_can_be_added_by_group_selection(hydra_all):
    cfg = compose(
        config_name="config",
        overrides=["model=dino", "+model/term@model.terms.charge=charge", "run.name=t"],
    )
    assert list(cfg.model.terms) == ["dino", "charge"]


def test_production_constants_flow_from_data_into_the_model(hydra_all):
    """The cropper's canvas and the normaliser's percentiles describe a production."""
    cfg = compose(config_name="config", overrides=["model=dino", "run.name=t"])
    assert cfg.model.augment.cropper.image_w == cfg.data.image_w == 1050
    assert cfg.model.augment.cropper.image_h == cfg.data.image_h == 1500
    assert cfg.model.normalize.min_val == cfg.data.feat_min_val
    assert cfg.model.normalize.enabled is True
    cfg = compose(config_name="config", overrides=["model=dino", "data.image_w=1100", "run.name=t"])
    assert cfg.model.augment.cropper.image_w == 1100


def test_the_model_schema_is_load_bearing(hydra_all):
    with pytest.raises((ValidationError, ConfigCompositionException), match="teacher_temp"):
        compose(
            config_name="config",
            overrides=["model=dino", "model.terms.dino.teacher_temp=hot", "run.name=t"],
        )
    with pytest.raises((ValidationError, ConfigCompositionException), match="win_ch"):
        compose(
            config_name="config",
            overrides=["model=dino", "model.augment.masker.win_ch=wide", "run.name=t"],
        )


def test_the_old_backbone_registry_is_reachable_as_flags(hydra_all):
    """The old repo's 13+2 backbone classes are flags on one class, not presets.

    The five ablation *files* (`attn_default`, `attn_noenc`, `attn_noflash`,
    `attn_noflashenc`, `base`) were deleted on 2026-09-10: no preset selected them, nothing
    trained them, and each forced `inject_roles: []` so none could run hybrid or mae. The
    capability they demonstrated is unchanged and is checked here directly -- an ablation is
    an override, not a file.
    """
    cfg = compose(config_name="config", overrides=["model=dino", "run.name=t"])
    shipped = OmegaConf.to_container(cfg.model.backbone)
    for key, value in (
        ("attention", True),
        ("spatial_encoding", True),
        ("flash_attention", True),
        ("inject_roles", ["masked"]),
    ):
        assert shipped[key] == value, f"attn_mae is the shipped backbone: {key}"

    for override, key, value in (
        ("model.backbone.attention=false", "attention", False),
        ("model.backbone.spatial_encoding=false", "spatial_encoding", False),
        ("model.backbone.flash_attention=false", "flash_attention", False),
        ("model.backbone.inject_roles=[]", "inject_roles", []),
    ):
        cfg = compose(
            config_name="config", overrides=["model=dino", override, "run.name=t"]
        )
        assert OmegaConf.to_container(cfg.model.backbone)[key] == value, override


def test_augment_options(hydra_all):
    cfg = compose(
        config_name="config", overrides=["model=dino", "model/augment=mask_only", "run.name=t"]
    )
    assert cfg.model.augment.cropper is None and cfg.model.augment.masker._target_.endswith(
        "BlockMasker"
    )
    cfg = compose(
        config_name="config",
        overrides=[
            "model=dino",
            "model/augment=mask_only",
            "model/masker@model.augment.masker=region",
            "run.name=t",
        ],
    )
    assert cfg.model.augment.masker._target_.endswith("RegionMasker")
    assert cfg.model.augment.masker.image_w == 1050, "the region grid is defined on the canvas"


def test_the_head_can_be_removed_with_null(hydra_all):
    cfg = compose(
        config_name="config",
        overrides=["model=dino", "model.terms.dino.proj_head=null", "run.name=t"],
    )
    assert cfg.model.terms.dino.proj_head is None


def test_preset_files_declare_package_global_and_subgroups_do_not():
    """Top-level presets carry overrides on other axes and must be `_global_`; a sub-group file
    lands under its group path, and a `_global_` header there would put the backbone at the
    config root."""
    for p in (CONF / "model").glob("*.yaml"):
        assert p.read_text().lstrip().startswith("# @package _global_"), p
    for p in (CONF / "model").glob("*/*.yaml"):
        assert "@package" not in p.read_text(), f"{p}: a sub-group file must not re-package itself"


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_composes(hydra_all, experiment):
    """Every file in `conf/experiment/` must actually load.

    `hybrid_baseline_mixed_b100_pefix` did not, from the day it was committed until
    2026-09-10: its defaults said `- /model: hybrid` where `model` is already a group in
    `conf/config.yaml`, so Hydra refused with "Multiple values for model" and the fix was the
    word `override`. Nothing caught it because nothing composed it -- the presets were tested,
    the experiments were not, and this one is the comparison target for the whole old-vs-new
    framework question, so its first real use would have been the run that mattered.

    A composition failure is the cheapest bug in this repo to find and among the most annoying
    to hit, because it lands after the queue.
    """
    cfg = compose(config_name="config", overrides=[f"+experiment={experiment}"])
    assert cfg.model is not None and cfg.data is not None
    assert str(cfg.model._target_).endswith("SslModule")
