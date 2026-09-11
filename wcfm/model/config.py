"""The typed schema for the `model` axis, contributed through the `wcfm.config_schemas` entry
point so that no framework module ever names `wcfm.model`.

Torch-free by construction. `wcfm.config.store.register_plugins` calls `register` in the
config-only environment too, which is where `wcfm env-check` runs, so this module imports
dataclasses and the ConfigStore and nothing else.

A default is written in exactly one place, and that place is here. The files under
`conf/model/` carry only the values that differ from these defaults, which keeps a yaml from
becoming a second copy of a number. Each class's constructor also has defaults, so a test can
build one without a config, and `tests/test_model_config.py` asserts the two agree: a constructor
default that drifts from the schema default is invisible until a run reads one and a test reads
the other.

Each option is registered as the schema of its own group entry, never as a union, because
OmegaConf cannot represent a Union of dataclasses. A yaml selects its schema with a same-group
default (`defaults: [base_dino]`), the way `conf/metrics/full.yaml` selects `minimal`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

# ------------------------------------------------------------------------------- backbone


@dataclass
class MinkUNetConfig:
    """`wcfm.model.backbones.MinkUNetAttention`: one class, and the flags that vary it."""

    _target_: str = "wcfm.model.backbones.MinkUNetAttention"
    in_ch: int = 1
    # (stem, encoder stage 1, encoder stage 2, decoder), the four channel widths that were
    # literals at minkunet_attention.py:53-80.
    widths: list[int] = field(default_factory=lambda: [32, 32, 64, 64])
    out_dim: int = 64
    attention: bool = True
    spatial_encoding: bool = True
    flash_attention: bool = True
    encoding_dim: int = 32
    encoding_range: float = 125.0
    attn_ch: int = 128
    heads: int = 4
    # Which roles this backbone holds a mask token for, at every skip it injects into. Empty
    # means the backbone cannot inject and can only run the plain dino objective. The list is
    # what makes injection a declared capability rather than an `inspect.signature` sniff.
    inject_roles: list[str] = field(default_factory=lambda: ["masked"])


# --------------------------------------------------------------------------------- augment


@dataclass
class PixelMaskerConfig:
    _target_: str = "wcfm.model.augment.PixelMasker"
    ratio: float = 0.5


@dataclass
class BlockMaskerConfig:
    _target_: str = "wcfm.model.augment.BlockMasker"
    ratio: float = 0.5
    win_ch: int = 5
    win_tick: int = 5


@dataclass
class RegionMaskerConfig:
    """The grid is defined on the canvas, so `image_w` and `image_h` are interpolated from
    `data` in the yaml rather than defaulted here."""

    _target_: str = "wcfm.model.augment.RegionMasker"
    image_w: int = MISSING
    image_h: int = MISSING
    cell_w: int = 70
    cell_h: int = 100
    flavor: str = "wipe"
    wipe_max: float = 0.75
    r1: float = 0.5
    r2: float = 0.75
    build_candidates: bool = False
    cand_stride: int = 2
    neg_per_pos: float | None = None
    max_neg: int | None = None


@dataclass
class CropperConfig:
    _target_: str = "wcfm.model.augment.Cropper"
    image_w: int = MISSING
    image_h: int = MISSING
    n_global: int = 2
    n_local: int = 4
    global_scale: list[float] = field(default_factory=lambda: [0.4, 1.0])
    local_scale: list[float] = field(default_factory=lambda: [0.05, 0.2])
    aspect_ratio: list[float] = field(default_factory=lambda: [0.75, 1.333])
    blur_sigma_px: float = 10.0
    heatmap_power: float = 1.0
    min_active_pixels: int = 10
    max_attempts: int = 50


@dataclass
class AugmentConfig:
    """Crops, then a mask per student view. Either half may be absent."""

    _target_: str = "wcfm.model.augment.Augment"
    cropper: Any = None
    masker: Any = None


@dataclass
class LogTransformConfig:
    """The charge normalisation is a stage of `training_step`, so it is model config, but its
    constants describe a production, so they interpolate from `data`."""

    _target_: str = "wcfm.model.augment.FeatureLogTransform"
    min_val: float = MISSING
    max_val: float = MISSING
    enabled: bool = True


# --------------------------------------------------------------------------------- teacher


@dataclass
class EmaTeacherConfig:
    _target_: str = "wcfm.model.modules.EmaTeacher"
    momentum_start: float = 0.996
    momentum_end: float = 1.0


@dataclass
class NoTeacherConfig:
    _target_: str = "wcfm.model.modules.NoTeacher"


# ----------------------------------------------------------------------------------- terms


@dataclass
class ProjHeadConfig:
    hidden_dim: int = 256
    output_dim: int = 128
    n_layers: int = 2


@dataclass
class PenaltyConfig:
    enabled: bool = False
    weight: float = 1.0


@dataclass
class VarPenaltyConfig(PenaltyConfig):
    gamma: float = 1.0


@dataclass
class DinoTermConfig:
    """`hybrid` is this term with `score_injected: true`, not a third objective."""

    _target_: str = "wcfm.model.terms.DinoTerm"
    weight: float = 1.0
    score_injected: bool = False
    # `null` removes the head; the loss then L2-normalises the backbone features itself.
    proj_head: ProjHeadConfig | None = field(default_factory=ProjHeadConfig)
    center_momentum: float = 0.9
    use_centering: bool = True
    teacher_temp: float = 0.07
    student_temp: float = 0.1
    cov_penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    var_penalty: VarPenaltyConfig = field(default_factory=VarPenaltyConfig)


@dataclass
class ChargeTermConfig:
    _target_: str = "wcfm.model.terms.ChargeTerm"
    weight: float = 0.1


@dataclass
class OccupancyTermConfig:
    """`alpha` and `gamma` are the focal-loss knobs. They are exposed rather than fixed at
    0.25/2.0 because the positive rate depends on how the candidate set was built, and this
    term runs on an enumerated set."""

    _target_: str = "wcfm.model.terms.OccupancyTerm"
    weight: float = 1.0
    alpha: float = 0.25
    gamma: float = 2.0


# ---------------------------------------------------------------------------------- module


@dataclass
class SslModuleConfig:
    """The composition root of the model axis. `terms` is keyed by the name each term is
    recorded under; `+/model/term@model.terms.pid=pid` adds one, `~model.terms.charge`
    removes one."""

    _target_: str = "wcfm.model.modules.SslModule"
    # Nested containers arrive as plain dicts, so `terms` can become an nn.ModuleDict without
    # anyone unwrapping a DictConfig.
    _convert_: str = "all"
    backbone: Any = MISSING
    terms: dict[str, Any] = field(default_factory=dict)
    augment: Any = MISSING
    teacher: Any = None
    normalize: Any = None
    # Backbone taps to request on every forward and publish as observables.
    observe_taps: list[str] = field(default_factory=list)


GROUPS: tuple[tuple[str, str, type], ...] = (
    ("model/module", "ssl", SslModuleConfig),
    ("model/backbone", "base_minkunet", MinkUNetConfig),
    ("model/masker", "base_pixel", PixelMaskerConfig),
    ("model/masker", "base_block", BlockMaskerConfig),
    ("model/masker", "base_region", RegionMaskerConfig),
    ("model/cropper", "base_cropper", CropperConfig),
    ("model/augment", "base_augment", AugmentConfig),
    ("model/normalize", "base_log", LogTransformConfig),
    ("model/teacher", "base_ema", EmaTeacherConfig),
    ("model/teacher", "base_none", NoTeacherConfig),
    ("model/term", "base_dino", DinoTermConfig),
    ("model/term", "base_charge", ChargeTermConfig),
    ("model/term", "base_occupancy", OccupancyTermConfig),
)


def register() -> None:
    """The `wcfm.config_schemas` entry point. Idempotent: `store` overwrites."""
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    for group, name, node in GROUPS:
        cs.store(group=group, name=name, node=node)
