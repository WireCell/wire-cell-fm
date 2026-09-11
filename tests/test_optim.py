"""Schedules and optimizer construction.

The centrepiece is **parity with the old numpy schedule**, element for element. The new class
computes values analytically, which is what makes it independent of ``total_iters`` being
known as an array length -- but the archived runs took the numpy array's values, including its
two rough edges, so "cleaner" would have been wrong. Both edges are asserted directly, not
merely covered by the parity sweep, so that a future simplification that smooths them fails
here with a reason rather than in a diff of two loss curves.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from wcfm.config.io import warmup_iters_from_epochs  # noqa: E402
from wcfm.engine.optim import (  # noqa: E402
    CosineScheduler,
    apply_schedules,
    build_optimizer,
    compose_param_groups,
)

from .toy import ToyModule  # noqa: E402

pytestmark = pytest.mark.stack


def legacy_schedule(
    base_value: float,
    final_value: float,
    total_iters: int,
    warmup_iters: int = 0,
    start_warmup_value: float = 0.0,
    freeze_iters: int = 0,
) -> np.ndarray:
    """``dino/scheduler.py`` from the old repo, verbatim. The reference, not a reimplementation."""
    freeze_schedule = np.zeros(freeze_iters)
    warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(total_iters - warmup_iters - freeze_iters)
    cosine_schedule = (
        final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    )
    schedule = np.concatenate((freeze_schedule, warmup_schedule, cosine_schedule))
    assert len(schedule) == total_iters
    return schedule


@pytest.mark.parametrize(
    "epochs,steps_per_epoch,warmup_epochs,base,final,warmup_value,freeze_epochs",
    [
        (10, 100, 1, 1e-4, 1e-6, 0.0, 0.0),      # the lr shape of a real run
        (100, 50, 1, 1e-4, 1e-6, 0.0, 0.0),
        (10, 100, 0, 0.04, 0.4, 0.0, 0.0),        # the wd shape: no warmup, increasing
        (10, 100, 5, 1e-3, 1e-5, 0.0, 0.0),       # warmup hits the 0.2 cap
        (20, 10, 2, 0.9, 0.999, 0.0, 0.0),        # momentum-like, base < final
        (10, 20, 1, 1e-3, 1e-5, 1e-6, 0.0),       # non-zero start_warmup_value
        (10, 20, 1, 1e-3, 1e-5, 0.0, 1.0),        # freeze period
        (4, 1, 0, 1.0, 0.0, 0.0, 0.0),            # one step per epoch
    ],
)
def test_matches_the_legacy_numpy_schedule_exactly(
    epochs, steps_per_epoch, warmup_epochs, base, final, warmup_value, freeze_epochs
):
    total = epochs * steps_per_epoch
    warmup = warmup_iters_from_epochs(warmup_epochs, steps_per_epoch, total)
    freeze = int(freeze_epochs * steps_per_epoch)
    reference = legacy_schedule(
        base_value=base,
        final_value=final,
        total_iters=total,
        warmup_iters=warmup,
        start_warmup_value=warmup_value,
        freeze_iters=freeze,
    )
    sched = CosineScheduler(
        base_value=base,
        final_value=final,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=warmup_epochs,
        warmup_value=warmup_value,
        freeze_epochs=freeze_epochs,
    )
    got = np.array([sched[i] for i in range(total)])
    assert np.allclose(got, reference, rtol=0, atol=1e-15), (
        f"max |delta| = {np.abs(got - reference).max():.3e}"
    )


def test_past_the_end_returns_the_final_value():
    """``__getitem__`` beyond ``total_iters`` -- the old class's behaviour, and what a run that
    overshoots its schedule by a step relies on."""
    sched = CosineScheduler(base_value=1.0, final_value=0.1, epochs=2, steps_per_epoch=5)
    assert sched[10] == pytest.approx(0.1)
    assert sched[10_000] == pytest.approx(0.1)


def test_the_cosine_does_not_quite_reach_the_final_value():
    """Rough edge one, asserted rather than inherited: the cosine divides by ``n_cos``, not
    ``n_cos - 1``, so the last scheduled value sits short of ``final_value``. The archived runs
    took this value; smoothing it would silently change every run's endpoint."""
    sched = CosineScheduler(base_value=1.0, final_value=0.0, epochs=1, steps_per_epoch=100)
    last = sched[99]
    assert last > 0.0
    assert last == pytest.approx(0.5 * (1 + math.cos(math.pi * 99 / 100)))
    assert sched[100] == 0.0, "only past the end does it snap to final_value"


def test_warmup_reaches_base_and_the_cosine_restarts_from_it():
    """Rough edge two: ``np.linspace`` includes its endpoint, so ``base_value`` appears at the
    last warmup index and again at the first cosine index."""
    sched = CosineScheduler(
        base_value=1.0, final_value=0.0, epochs=10, steps_per_epoch=10, warmup_epochs=1
    )
    assert sched.warmup_iters == 10
    assert sched[9] == pytest.approx(1.0)
    assert sched[10] == pytest.approx(1.0)


def test_single_warmup_iteration_yields_the_start_value_only():
    """``np.linspace(a, b, 1)`` is ``[a]``: with one warmup iteration the schedule never
    reaches ``base_value`` during warmup at all."""
    sched = CosineScheduler(
        base_value=1.0,
        final_value=0.0,
        epochs=10,
        steps_per_epoch=1,
        warmup_epochs=1,
        warmup_value=0.25,
    )
    assert sched.warmup_iters == 1
    assert sched[0] == pytest.approx(0.25)


def test_freeze_period_is_zero_not_the_base_value():
    sched = CosineScheduler(
        base_value=1.0, final_value=0.0, epochs=10, steps_per_epoch=10, freeze_epochs=2
    )
    assert sched[0] == 0.0 and sched[19] == 0.0
    assert sched[20] == pytest.approx(1.0)


def test_no_cosine_phase_is_refused():
    """warmup plus freeze consuming the whole run gave the old class a zero-length arange and
    a divide-by-zero into a nan array; the length assert then fired with no explanation."""
    with pytest.raises(ValueError, match="no cosine phase"):
        CosineScheduler(
            base_value=1.0,
            final_value=0.0,
            epochs=2,
            steps_per_epoch=10,
            freeze_epochs=2,
        )


def test_per_quantity_warmup_is_not_the_same_as_a_flat_ramp():
    """The reason ``warmup_epochs`` is per schedule entry rather than inherited from
    ``OptimConfig``. Expressing "no warmup" as ``warmup_value == base_value`` leaves the ramp
    flat but still shortens the cosine to ``total_iters - warmup_iters``, so the two disagree
    from the first cosine step onward -- silently, in the direction of a faster decay."""
    kw = dict(base_value=0.04, final_value=0.4, epochs=10, steps_per_epoch=100)
    no_warmup = CosineScheduler(**kw, warmup_epochs=0)
    flat_ramp = CosineScheduler(**kw, warmup_epochs=1, warmup_value=0.04)

    assert no_warmup.cosine_iters == 1000
    assert flat_ramp.cosine_iters == 900
    assert no_warmup[0] == pytest.approx(flat_ramp[0])  # both start at base
    assert no_warmup[500] != pytest.approx(flat_ramp[500]), "the flat ramp decays faster"


def test_the_default_conf_reproduces_the_archived_lr_and_wd_shapes():
    """``conf/optim/adamw_cosine.yaml`` warms up the learning rate and not weight decay, which
    is what ``train_dino.py:681-691`` did -- the lr got ``warmup_iters``, the wd got 0."""
    lr = CosineScheduler(
        base_value=1e-4, final_value=1e-6, epochs=100, steps_per_epoch=100, warmup_epochs=1
    )
    wd = CosineScheduler(
        base_value=0.04, final_value=0.4, epochs=100, steps_per_epoch=100, warmup_epochs=0
    )
    assert lr.warmup_iters == 100 and lr.cosine_iters == 9900
    assert wd.warmup_iters == 0 and wd.cosine_iters == 10_000
    assert lr[0] == 0.0, "the learning rate starts from warmup_value"
    assert wd[0] == pytest.approx(0.04), "weight decay starts at its base value"


# ------------------------------------------------------------------ param groups


def test_param_groups_carry_scales_and_names():
    module = ToyModule()
    groups = compose_param_groups(module)
    assert [g["name"] for g in groups] == ["backbone", "head"]
    assert groups[1]["lr_scale"] == 2.0


def test_freeze_backbone_drops_the_group_and_clears_requires_grad():
    """A frozen group is dropped, not merely zeroed: an optimizer group whose parameters never
    receive a gradient still carries AdamW state for them."""
    module = ToyModule()
    groups = compose_param_groups(module, freeze_backbone=True)
    assert [g["name"] for g in groups] == ["head"]
    assert all(not p.requires_grad for p in module.net.parameters())
    assert all(p.requires_grad for p in module.head.parameters())


def test_a_parameter_in_two_groups_is_refused():
    """Its effective learning rate would be the sum of two updates, which no metric shows."""

    class Doubled(ToyModule):
        def param_groups(self):
            shared = list(self.net.parameters())
            return [{"name": "a", "params": shared}, {"name": "b", "params": shared}]

    with pytest.raises(ValueError, match="both param groups"):
        compose_param_groups(Doubled())


def test_no_trainable_group_is_refused():
    """Trains nothing and reports a falling loss of zero."""

    class Frozen(ToyModule):
        def param_groups(self):
            return [{"name": "all", "params": list(self.parameters()), "requires_grad": False}]

    with pytest.raises(ValueError, match="no trainable parameter groups"):
        compose_param_groups(Frozen())


def test_a_module_without_param_groups_says_so():
    with pytest.raises(TypeError, match="param_groups"):
        compose_param_groups(torch.nn.Linear(2, 2))


# ------------------------------------------------------------------ application


class _Cfg:
    name = "adamw"
    lr = 1e-4
    weight_decay = 0.04
    freeze_backbone = False


def test_apply_schedules_writes_lr_and_wd_and_reports_what_it_wrote():
    """Every scheduled value is a recorded column. The engine knows it, so the engine writes
    it -- today lr appears only as text in training.log."""
    module = ToyModule()
    optimizer = build_optimizer(module, _Cfg())
    schedules = {
        "lr": CosineScheduler(base_value=1.0, final_value=0.0, epochs=10, steps_per_epoch=10),
        "weight_decay": CosineScheduler(
            base_value=0.1, final_value=0.5, epochs=10, steps_per_epoch=10
        ),
    }
    applied = apply_schedules(optimizer, schedules, step=0)

    assert applied["lr"] == pytest.approx(1.0)
    assert applied["weight_decay"] == pytest.approx(0.1)
    # lr_scale=2.0 on the head, so its group gets twice the scheduled rate and is reported
    # separately; the unscaled backbone group is not, so a single-group run stays two columns.
    assert applied["lr/head"] == pytest.approx(2.0)
    assert "lr/backbone" not in applied

    by_name = {g["name"]: g for g in optimizer.param_groups}
    assert by_name["backbone"]["lr"] == pytest.approx(1.0)
    assert by_name["head"]["lr"] == pytest.approx(2.0)
    assert by_name["head"]["weight_decay"] == pytest.approx(0.1)


def test_build_optimizer_rejects_an_unknown_name():
    class Bad(_Cfg):
        name = "lamb"

    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(ToyModule(), Bad())


def test_the_registered_schedule_group_matches_what_cosinescheduler_accepts():
    """``store.py`` registers ``optim/schedule=cosine`` as ``ScheduleConfig``, but
    ``conf/optim/adamw_cosine.yaml`` writes its entries inline -- so nothing exercised that
    registration, and a field added to one and not the other would be caught nowhere until
    somebody tried ``optim/schedule=linear`` in Stage 3.

    Every ``ScheduleConfig`` field must be a real ``CosineScheduler`` argument, and the two
    the engine injects (``epochs``, ``steps_per_epoch``) must NOT be fields -- they are facts
    about the dataset that no config knows.
    """
    import inspect
    from dataclasses import fields

    from hydra.core.config_store import ConfigStore

    from wcfm.config.schema import ScheduleConfig
    from wcfm.config.store import register_framework

    register_framework()
    repo = ConfigStore.instance().repo
    assert "cosine.yaml" in repo["optim"]["schedule"], "the group entry is registered"

    accepted = set(inspect.signature(CosineScheduler.__init__).parameters) - {"self"}
    declared = {f.name for f in fields(ScheduleConfig)} - {"_target_"}

    unknown = declared - accepted
    assert not unknown, f"ScheduleConfig fields CosineScheduler cannot take: {unknown}"

    injected = {"epochs", "steps_per_epoch"}
    assert not (declared & injected), "epochs/steps_per_epoch are injected, not config keys"
    assert injected <= accepted, "the engine must be able to inject both"
