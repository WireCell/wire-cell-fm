"""The typed backing schema for the framework's config groups.

One dataclass per group in conf/, giving every key a type and a default. Hydra merges the
YAML onto these, so a key that is not declared here is a composition error
and a value of the wrong type is caught before a device is touched.

Note: 
- These dataclasses are the only place a framework default is written; 
  the YAML in conf/ restates a value only to change it.
- Batch size is global, so DataConfig.global_batch_size is what the job trains on across
  every rank; the per-rank division happens in one place (io.per_rank_batch_size).
- Each model option registers the schema of its own group entry in store.py. 
  That is also what keeps this module free of any wcfm.model import.

"""
from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING


@dataclass
class RunConfig:
    """Identity, placement and cadence of one run. Nothing here is science."""

    name: str = ""
    seed: int = 42
    # Everything a run writes goes under <output_root>/<name>/, in the fixed subdirectories
    # {checkpoints,debug,probes,features,metrics}.
    output_root: str = "./runs"
    num_workers: int = 5
    # 32-true | bf16-mixed | 16-mixed. 
    precision: str = "32-true"
    # Sparse scatter accumulates with atomics, 
    # so the same seed does not give bit-identical parameters.
    deterministic: bool = False
    # "auto" resumes from latest.pt if one is there, 
    # "none" always starts fresh, 
    # anything else is a path to a checkpoint.
    resume: str = "auto"
    save_every: int = 10
    save_every_minutes: int = 0
    # Explicit epochs to checkpoint at, on top of the `save_every` interval. Points of a
    # sweep with different `epochs` can then be probed at the same epochs.
    save_at: list[int] = field(default_factory=list)


@dataclass
class SplitConfig:
    """Which events a run trains on. 'in-sample' means all of them: there is no held-out
    split yet, and the id says so rather than implying one."""

    id: str = "in-sample"
    train_frac: float = 1.0


@dataclass
class DataConfig:
    """One production plus how to read it. backend selects the reader."""

    backend: str = MISSING  # direct | sharded | packed
    datadir: str = ""
    sharded_dir: str = ""
    packed_path: str = ""
    cache_dir: str = "./data"
    buffer_size: int = 3000
    n_subset: int = -1

    # Per-pixel truth is opt-in on all three backends because HDF5 decompresses those datasets
    # on every read; event-level truth is always returned. `extra` implies `pixel`.
    return_pixel_truth: bool = False
    return_extra_truth: bool = False

    # Across all ranks. Must divide by launch.devices; io.per_rank_batch_size does the split.
    global_batch_size: int = MISSING

    # What the production is, as opposed to how the run treats it.
    image_h: int = 1500
    image_w: int = 1050
    apa: int = 0
    view: str = "W"
    use_log_transform: bool = True
    feat_min_val: float = 3.75
    feat_max_val: float = 83861.2

    splits: SplitConfig = field(default_factory=SplitConfig)


@dataclass
class ScheduleConfig:
    """One scheduled scalar, chosen by '_target_' like any other component. 'CosineScheduler'
    is one such class; a linear decay is a different '_target_' and no code change.
    
    "epochs" and "steps_per_epoch" are not keys here. The engine injects them at
    instantiation, since "steps_per_epoch" is len(train_loader) and no config knows it.
    """

    _target_: str = "wcfm.engine.optim.CosineScheduler"
    base_value: float = MISSING
    final_value: float = MISSING
    warmup_epochs: float = 0.0
    warmup_value: float = 0.0
    freeze_epochs: float = 0.0


@dataclass
class OptimConfig:
    """What the framework schedules: learning rate and weight decay, both properties of the
    optimizer. Teacher momentum, teacher temperature and mask ratio belong to the model and
    are scheduled by it through 'on_step_end'."""

    name: str = "adamw"
    epochs: int = 100
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.04
    weight_decay_end: float = 0.4
    warmup_epochs: int = 1
    # Unused by the shipped presets, but declared anyway: each one changes optimizer
    # construction, checkpoint contents and the resume path together
    clip_grad_norm: float = 0.0
    accumulate_grad_batches: int = 1
    freeze_backbone: bool = False
    # Keyed by quantity name, which the engine does not enumerate.
    # Each value is a `ScheduleConfig`, registered as the `optim/schedule` group.
    schedules: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    """What a run records. Always on: there is no flag that changes what gets written, only
    which collectors run and how often. See `conf/metrics/` for the presets."""

    # Collectors keyed by name, each an object with a "_target_".
    collectors: dict[str, Any] = field(default_factory=dict)
    step_cadence: int = 100
    max_record_bytes: int = 4096


@dataclass
class LaunchConfig:
    """How many ranks the job runs on, and how they are wrapped. Single node only;
    'num_nodes' is here so the contract does not foreclose more."""

    # The one place the rank count is stated: it sets Fabric's world, Condor's request_gpus,
    # and whether the job starts under torchrun (>1) or plain python (1)
    devices: int = 1
    num_nodes: int = 1
    strategy: str = "ddp"
    # Required. A head that runs on only some views receives no gradient on the others, and
    # without this DDP leaves that bucket unfinished and performs no reduction at all.
    find_unused_parameters: bool = True
    static_graph: bool = False


@dataclass
class Config:
    """The composition root."""

    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    model: Any = MISSING

    # Set by `wcfm train --dry-run`: compose, validate and construct on CPU, then exit.
    dry_run: bool = False


FRAMEWORK_GROUPS: tuple[tuple[str, type], ...] = (
    ("run", RunConfig),
    ("data", DataConfig),
    ("optim", OptimConfig),
    ("metrics", MetricsConfig),
    ("launch", LaunchConfig),
)
