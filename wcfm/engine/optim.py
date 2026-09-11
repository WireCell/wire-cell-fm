"""Optimizer construction, and the schedules the framework owns.

Learning rate and weight decay are scheduled here, both being properties of the optimizer.
Teacher momentum, teacher temperature and mask ratio belong to the model, which schedules
them in `on_step_end(ctx)` and publishes the applied value through `StepOutput.scalars`.

Three properties of the schedules:

- They are analytic: `CosineScheduler[it]` computes its value. Two edges of the shape are
  deliberate and pinned element-for-element by `tests/test_optim.py`. The warmup ramp
  reaches `base_value` at its last index, so the cosine restarts from a duplicated
  `base_value`, and the cosine divides by `n_cos`, so it stops just short of `final_value`.
  Do not smooth either out: the archived runs this framework is compared against took those
  values.
- Warmup is per quantity. Each entry carries its own `warmup_epochs`, and
  `conf/optim/adamw_cosine.yaml` sets weight decay's to 0. Writing "no warmup" as
  `warmup_value == base_value` differs, because a flat ramp still shortens the cosine to
  `total_iters - warmup_iters`.
- Epochs in, iterations out, in one place: the 0.2 cap on warmup lives in
  `config.io.warmup_iters_from_epochs`, read both here and by the `config.yaml` a run
  records.

The optimizer is built from the module's declared `param_groups`. A group may carry
`lr_scale` and `wd_scale`, which `apply_schedules` multiplies into the scheduled value, so
a frozen backbone or a discriminative rate is something the module declares.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from wcfm.config.io import warmup_iters_from_epochs

__all__ = [
    "CosineScheduler",
    "apply_schedules",
    "build_optimizer",
    "build_schedules",
    "compose_param_groups",
]


class CosineScheduler:
    """Cosine annealing with an optional linear warmup and freeze period.

    `epochs` and `steps_per_epoch` are injected by the engine at construction --
    `steps_per_epoch` is `len(train_loader)`, a fact about the dataset and the world size
    that no config knows. Every other argument is a config key, so a schedule entry stays a
    plain `_target_` node like any other component.
    """

    def __init__(
        self,
        base_value: float,
        final_value: float,
        *,
        epochs: int,
        steps_per_epoch: int,
        warmup_epochs: float = 0.0,
        warmup_value: float = 0.0,
        freeze_epochs: float = 0.0,
    ):
        self.base_value = float(base_value)
        self.final_value = float(final_value)
        self.warmup_value = float(warmup_value)

        self.total_iters = int(epochs) * int(steps_per_epoch)
        self.warmup_iters = warmup_iters_from_epochs(
            warmup_epochs, int(steps_per_epoch), self.total_iters
        )
        self.freeze_iters = int(float(freeze_epochs) * int(steps_per_epoch))

        self.cosine_iters = self.total_iters - self.warmup_iters - self.freeze_iters
        if self.cosine_iters <= 0:
            raise ValueError(
                f"warmup ({self.warmup_iters}) plus freeze ({self.freeze_iters}) leaves no "
                f"cosine phase in total_iters={self.total_iters}"
            )

    def __getitem__(self, it: int) -> float:
        if it >= self.total_iters:
            return self.final_value
        if it < 0:
            raise IndexError(f"negative iteration {it}")
        if it < self.freeze_iters:
            return 0.0

        j = it - self.freeze_iters
        if j < self.warmup_iters:
            # np.linspace(warmup_value, base_value, warmup_iters)[j]. With one warmup
            # iteration linspace yields the start value alone, never base_value.
            if self.warmup_iters == 1:
                return self.warmup_value
            span = self.base_value - self.warmup_value
            return self.warmup_value + span * j / (self.warmup_iters - 1)

        i = j - self.warmup_iters
        half_span = 0.5 * (self.base_value - self.final_value)
        return self.final_value + half_span * (1.0 + math.cos(math.pi * i / self.cosine_iters))

    def __len__(self) -> int:
        return self.total_iters

    def __repr__(self) -> str:
        return (
            f"CosineScheduler(base={self.base_value:g}, final={self.final_value:g}, "
            f"total_iters={self.total_iters}, warmup_iters={self.warmup_iters}, "
            f"freeze_iters={self.freeze_iters})"
        )


def build_schedules(optim_cfg: Any, steps_per_epoch: int) -> dict[str, Any]:
    """Instantiate one schedule per entry in `optim.schedules`.

    The keys are quantity names the engine does not enumerate: whatever the config lists
    gets scheduled. `apply_schedules` recognises `lr` and `weight_decay`, the two the
    optimizer holds; anything else is recorded and inert. A typo therefore surfaces as a
    column of numbers nobody asked for.
    """
    from hydra.utils import instantiate

    schedules = getattr(optim_cfg, "schedules", None) or {}
    out: dict[str, Any] = {}
    for name, entry in schedules.items():
        out[str(name)] = instantiate(
            entry, epochs=int(optim_cfg.epochs), steps_per_epoch=int(steps_per_epoch)
        )
    return out


def apply_schedules(
    optimizer: torch.optim.Optimizer, schedules: dict[str, Any], step: int
) -> dict[str, float]:
    """Write the scheduled learning rate and weight decay onto every param group.

    Returns the applied values as `{"lr": ..., "weight_decay": ..., "lr/<group>": ...}` so
    the engine can record them. Every scheduled value is a recorded column, per logged step:
    the learning rate actually applied is the first number anyone reads when a run diverges,
    and the engine already knows it, so the engine writes it.

    Per-group scales are reported only where a group actually carries one, so the common case
    of a single unscaled group stays two columns rather than four.
    """
    applied: dict[str, float] = {}
    for key, attr in (("lr", "lr"), ("weight_decay", "weight_decay")):
        sched = schedules.get(key)
        if sched is None:
            continue
        value = float(sched[step])
        applied[key] = value
        scale_key = "lr_scale" if key == "lr" else "wd_scale"
        for group in optimizer.param_groups:
            scale = float(group.get(scale_key, 1.0))
            group[attr] = value * scale
            name = group.get("name")
            if name is not None and scale != 1.0:
                applied[f"{key}/{name}"] = group[attr]
    return applied


def compose_param_groups(module: nn.Module, freeze_backbone: bool = False) -> list[dict]:
    """The module's declared groups, validated, with `requires_grad` honoured.

    A group is a dict carrying `params` and optionally `name`, `lr_scale`, `wd_scale` and
    `requires_grad`. `requires_grad: False` is applied to the tensors and the group is
    dropped, since an optimizer group whose parameters never receive a gradient still
    carries AdamW state for them.

    Two mistakes raise: a parameter appearing in two groups, which makes its effective
    learning rate the sum of two updates, and no trainable parameter at all, which trains
    nothing and reports a falling loss of zero.
    """
    if not hasattr(module, "param_groups"):
        raise TypeError(
            f"{type(module).__name__} does not implement param_groups(); see "
            "wcfm.engine.protocol.TrainingModule"
        )

    seen: dict[int, str] = {}
    groups: list[dict] = []
    for index, raw in enumerate(module.param_groups()):
        group = dict(raw)
        name = str(group.get("name", f"group{index}"))
        params = [p for p in group.pop("params", [])]

        trainable = bool(group.pop("requires_grad", True)) and not (
            freeze_backbone and name == "backbone"
        )
        for param in params:
            param.requires_grad_(trainable)

        for param in params:
            key = id(param)
            if key in seen:
                raise ValueError(
                    f"parameter appears in both param groups {seen[key]!r} and {name!r}; "
                    "its effective learning rate would be the sum of two updates"
                )
            seen[key] = name

        if not trainable or not params:
            continue
        group["name"] = name
        group["params"] = params
        groups.append(group)

    if not groups:
        raise ValueError(
            "no trainable parameter groups; every group was empty or declared "
            "requires_grad=False"
        )
    return groups


def build_optimizer(module: nn.Module, optim_cfg: Any) -> torch.optim.Optimizer:
    """AdamW over the module's declared groups and nothing else.

    `lr` and `weight_decay` are placeholders, overwritten by `apply_schedules` on every
    step including the first. They are passed to keep the optimizer's own defaults out.
    """
    name = str(getattr(optim_cfg, "name", "adamw")).lower()
    groups = compose_param_groups(module, bool(getattr(optim_cfg, "freeze_backbone", False)))
    if name == "adamw":
        return torch.optim.AdamW(
            groups, lr=float(optim_cfg.lr), weight_decay=float(optim_cfg.weight_decay)
        )
    if name == "sgd":
        return torch.optim.SGD(
            groups,
            lr=float(optim_cfg.lr),
            momentum=0.9,
            weight_decay=float(optim_cfg.weight_decay),
        )
    raise ValueError(f"unknown optimizer {name!r}; expected 'adamw' or 'sgd'")
