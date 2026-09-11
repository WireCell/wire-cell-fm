"""`wcfm train`: compose a config, build the module, hand both to the engine.

Hydra composes over `conf/` and the framework never names a model type, so this file is short:
it resolves, validates, instantiates `cfg.model` and constructs a `Trainer`. Everything it knows
about the thing it is training is that `_target_` produced it and that it satisfies
`TrainingModule`.

Three things worth stating about the shape:

- `model` has a default -- `conf/config.yaml` selects `mae` -- so `wcfm train` with no `model=`
  trains that objective rather than refusing. The cost is that a forgotten `model=` is a
  silently wrong objective rather than an error. A `conf/` tree reached through `--config-dir`
  may still set `model: ???`, which is why `explain_missing_model` keeps that branch.
- `hydra.job.chdir` stays false and this file does not touch it. The Condor job manages its own
  working directory and rsync layout, and `tests/test_config.py` pins it.
- The protocol is checked before the run rather than discovered during it. A module missing
  `on_step_end` would otherwise fail at the end of the first step, which on the sharded backend
  is several minutes of shard reading later.

`--dry-run` is not a flag on the loop. It is `Config.dry_run`, so a dry run composes, validates
and constructs on CPU and exits before `Trainer.setup` touches a device or a shard.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["explain_missing_model", "main", "run"]

USAGE = """usage: wcfm train [--dry-run] [hydra overrides ...]

  wcfm train run.name=my_run                             # model defaults to mae
  wcfm train model=hybrid run.name=my_run                # a different objective
  wcfm train model=dino data=prod_jay_100k run.name=x    # a different production
  wcfm train --dry-run model=hybrid run.name=x  # compose, validate, construct, exit
  wcfm train --config-dir /path/to/conf ...     # a conf/ tree other than the repo's

`model=` defaults to `mae` (conf/config.yaml). Pass it explicitly to train something else.
"""


def _conf_dir(argv: list[str]) -> tuple[Path, list[str]]:
    """`--config-dir` out of the argv, defaulting to the repo's `conf/`.

    Resolved from `wcfm.__file__` rather than from the working directory: a Condor job's cwd is
    whatever `trainjob.sh` left it at, and `conf/` is not package data, since a copy inside the
    wheel would be a second place a default could be written.
    """
    rest: list[str] = []
    override: str | None = None
    iterator = iter(argv)
    for arg in iterator:
        if arg == "--config-dir":
            override = next(iterator, None)
        elif arg.startswith("--config-dir="):
            override = arg.split("=", 1)[1]
        else:
            rest.append(arg)
    if override:
        return Path(override).expanduser().resolve(), rest

    import wcfm

    return (Path(wcfm.__file__).resolve().parent.parent / "conf"), rest


def explain_missing_model(exc: Exception, config_dir: Path) -> None:
    """Turn the two confusing composition failures into the sentence that fixes them.

    Shared with `wcfm submit`, since it is the same first-encounter message. The second case is
    the one that bites a checkout reached through PYTHONPATH: every file under `conf/model/`
    selects its typed schema from the ConfigStore (`defaults: [base_dino]`), and those nodes
    arrive through the `wcfm.config_schemas` entry point, which needs the `wirecell_fm.egg-info/`
    an editable install writes next to `wcfm/`. Hydra's own words for that are "Could not find
    'model/term/base_dino'", which reads like a typo.
    """
    from wcfm.config.store import LOADED

    text = str(exc)
    if "must specify 'model'" in text:
        options = sorted(p.stem for p in (config_dir / "model").glob("*.yaml"))
        print(f"\nmodel options in {config_dir / 'model'}: {options}", file=sys.stderr)
    elif "base_" in text and "model" not in LOADED:
        print(
            "\nThe model config schema plugin did not load (wcfm.config_schemas -> "
            "wcfm.model.config:register), so no `model/*/base_*` node exists to compose onto. "
            "Run `uv pip install -e . --no-deps` in the checkout so `wirecell_fm.egg-info/` "
            "exists next to `wcfm/`; an rsync that excludes `*.egg-info` produces this.",
            file=sys.stderr,
        )


def validate(cfg) -> None:
    """The checks worth making before a device is touched.

    Deliberately few: the typed schema and `test_config.py` already carry most of it. These
    are the ones whose failure mode is a wasted queue slot rather than an immediate error.
    """
    from wcfm.config.io import per_rank_batch_size

    if not str(cfg.run.name).strip():
        raise ValueError(
            "run.name is empty. Every C9 path is <output_root>/<name>/..., so an unnamed run "
            "writes into <output_root>/ and a second one overwrites it."
        )
    # Fails here rather than inside `build_loader` on rank 0 while the others wait.
    per_rank_batch_size(int(cfg.data.global_batch_size), int(cfg.launch.devices))
    if int(cfg.optim.accumulate_grad_batches) < 1:
        raise ValueError(
            f"optim.accumulate_grad_batches={cfg.optim.accumulate_grad_batches}; "
            "1 means no accumulation, 0 means no optimizer step ever runs"
        )


def build_module(cfg):
    """Instantiate `cfg.model` and check it against the contract before the run starts."""
    from hydra.utils import instantiate

    from wcfm.engine.protocol import TrainingModule

    module = instantiate(cfg.model)
    missing = [
        name
        for name in (
            "training_step",
            "param_groups",
            "observables",
            "on_step_end",
            "state_dict",
            "load_state_dict",
        )
        if not callable(getattr(module, name, None))
    ]
    if missing:
        raise TypeError(
            f"{type(module).__name__} does not satisfy TrainingModule; missing {missing}. "
            "See wcfm.engine.protocol."
        )
    # `runtime_checkable` only checks method presence, which is what the list above does more
    # informatively -- this is the assertion that the two agree.
    assert isinstance(module, TrainingModule)
    return module


def run(cfg, argv: list[str] | None = None) -> dict:
    """Validate, build, train. Separated from `main` so a test can drive it with a config
    it composed itself, without going through Hydra's decorator."""
    from wcfm.engine.trainer import Trainer

    validate(cfg)
    module = build_module(cfg)

    if bool(cfg.get("dry_run", False)):
        n_params = sum(p.numel() for p in module.parameters())
        print(f"dry run: {type(module).__name__} constructed, {n_params:,} parameters")
        print(f"dry run: run dir would be {Path(cfg.run.output_root) / cfg.run.name}")
        return {"dry_run": True, "parameters": n_params}

    return Trainer(cfg, module, argv=argv or []).fit()


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    import hydra
    from hydra.errors import HydraException
    from omegaconf import OmegaConf

    from wcfm.config.store import register_all

    register_all()
    config_dir, rest = _conf_dir(argv)
    if not config_dir.is_dir():
        print(f"wcfm train: no config directory at {config_dir}", file=sys.stderr)
        return 2

    dry = "--dry-run" in rest
    overrides = [a for a in rest if a != "--dry-run"]
    if dry:
        overrides.append("dry_run=true")

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        try:
            cfg = hydra.compose(config_name="config", overrides=overrides)
        except HydraException as exc:
            # A user error -- a missing or misspelled `model=`, a bad override -- is worth a
            # message, not a traceback through Hydra's internals. Hydra's own text names the
            # key and lists the options, so it is printed as-is rather than reworded.
            print(f"wcfm train: {exc}", file=sys.stderr)
            explain_missing_model(exc, config_dir)
            return 2
        OmegaConf.resolve(cfg)
        result = run(cfg, argv=["wcfm", "train", *rest])
    print(f"done: {result}")
    return 0
