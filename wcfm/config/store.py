"""Register to Hydra which dataclass in schema.py backs which group in conf/.

Registration is per group entry, one option at a time.
`register_framework()` does the five framework groups directly.

The model group arrives instead through the `wcfm.config_schemas` entry point: each entry
names a zero-argument callable that stores its own nodes, and `register_plugins()` calls
whatever it finds. This file therefore never names `wcfm.model`, which is what keeps the
framework free of a model import. `pyproject.toml` declares the one entry there is:

    model = "wcfm.model.config:register"

Entry points are read from package metadata, not from the source tree, so a checkout reached
by `PYTHONPATH` needs the `wirecell_fm.egg-info/` an editable install writes beside `wcfm/`.
Without it the model group simply does not exist and `model=` resolves to nothing.

A plugin that fails to load is warned about and skipped, so a broken model package leaves some
things (like `wcfm env-check`) still usable.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points

from hydra.core.config_store import ConfigStore

from .schema import FRAMEWORK_GROUPS, Config, ScheduleConfig

ENTRY_POINT_GROUP = "wcfm.config_schemas"

# The plugin names the last 'register_plugins()' loaded. The CLI reads it to turn "Could not
# find 'base_dino'" into "the model schema plugin is not installed", and 'wcfm env-check'
# prints it, because the failure it diagnoses is an rsync that dropped `*.egg-info`.
LOADED: list[str] = []


def register_framework(cs: ConfigStore | None = None) -> ConfigStore:
    """Register the top-level schema and the five framework axes."""
    cs = cs or ConfigStore.instance()
    cs.store(name="base_config", node=Config)
    for group, node in FRAMEWORK_GROUPS:
        # The group's own default entry, so `run=base` composes even with no YAML present.
        cs.store(group=group, name="base", node=node)
    # A scheduled quantity is a component like any other, so swapping cosine for a linear
    # decay is a group selection rather than a hand-written dict.
    cs.store(group="optim/schedule", name="cosine", node=ScheduleConfig)
    return cs


def register_plugins() -> list[str]:
    """Call every `wcfm.config_schemas` entry point. Returns the names that loaded."""
    loaded: list[str] = []
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            ep.load()()
        except Exception as exc:  # noqa: BLE001 - one broken plugin must not break the CLI
            warnings.warn(
                f"config schema plugin {ep.name!r} ({ep.value}) failed to register: "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        loaded.append(ep.name)
    LOADED[:] = loaded
    return loaded


def register_all() -> ConfigStore:
    register_plugins()
    return register_framework()
