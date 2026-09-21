"""The loop, run end to end on a CPU Fabric over ``tests/toy.py``.

"No backbone forward is CPU-testable" is a statement about the backbone. The loop -- gradient
accumulation, ``no_backward_sync`` gating, the non-finite guard, clipping, the schedule
application, checkpoint cadence, preemption and resume -- is all reachable here, and it is
where the engine's own bugs would live. The GPU suite covers what needs a device; this covers
what does not, so a regression in the loop does not wait for a Condor slot.
"""

from __future__ import annotations

import signal

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning_fabric")

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from wcfm.engine.preempt import PreemptionGuard  # noqa: E402
from wcfm.engine.protocol import StepOutput  # noqa: E402
from wcfm.engine.trainer import Trainer, _event_keys, _timing_line  # noqa: E402

from .toy import ToyBatch, ToyModule, toy_loader  # noqa: E402

pytestmark = pytest.mark.stack


def cpu_fabric():
    from lightning_fabric import Fabric

    return Fabric(accelerator="cpu", devices=1, precision="32-true")


def cfg(tmp_path, **over):
    """A complete config with the engine's axes filled in. ``metrics.collectors`` is empty on
    purpose in most tests: lr and weight decay are recorded by the engine, not a collector."""
    base = {
        "run": {
            "name": "toy",
            "seed": 42,
            "output_root": str(tmp_path),
            "num_workers": 0,
            "precision": "32-true",
            "deterministic": False,
            "resume": "none",
            "save_every": 0,
            "save_every_minutes": 0,
            "save_at": [],
        },
        "data": {"backend": "packed", "global_batch_size": 2, "splits": {"id": "in-sample"}},
        "optim": {
            "name": "adamw",
            "epochs": 2,
            "lr": 1e-3,
            "min_lr": 1e-5,
            "weight_decay": 0.04,
            "weight_decay_end": 0.4,
            "warmup_epochs": 0,
            "clip_grad_norm": 0.0,
            "accumulate_grad_batches": 1,
            "freeze_backbone": False,
            "schedules": {
                "lr": {
                    "_target_": "wcfm.engine.optim.CosineScheduler",
                    "base_value": 1e-3,
                    "final_value": 1e-5,
                    "warmup_epochs": 0,
                },
                "weight_decay": {
                    "_target_": "wcfm.engine.optim.CosineScheduler",
                    "base_value": 0.04,
                    "final_value": 0.4,
                    "warmup_epochs": 0,
                },
            },
        },
        "metrics": {"collectors": {}, "step_cadence": 1, "max_record_bytes": 4096},
        "launch": {"devices": 1, "num_nodes": 1, "strategy": "ddp",
                   "find_unused_parameters": True, "static_graph": False},
        "model": {"name": "toy"},
    }
    conf = OmegaConf.create(base)
    for dotted, value in over.items():
        OmegaConf.update(conf, dotted.replace("__", "."), value, merge=True)
    return conf


def run(tmp_path, module=None, loader=None, **over):
    module = module or ToyModule()
    trainer = Trainer(
        cfg(tmp_path, **over),
        module,
        fabric=cpu_fabric(),
        loader=loader or toy_loader(steps=4),
        run_dir=tmp_path / "run",
    )
    return trainer, trainer.fit()


# ------------------------------------------------------------------ the loop


def test_a_full_run_steps_the_optimizer_and_moves_the_parameters(tmp_path):
    module = ToyModule()
    before = module.net.weight.detach().clone()
    trainer, result = run(tmp_path, module=module)

    assert result["steps"] == 8, "2 epochs x 4 batches"
    assert result["epochs_run"] == 2
    assert not torch.equal(module.net.weight.detach(), before)
    assert module.step_ends == 8, "on_step_end runs once per optimizer step"


def test_the_module_is_handed_the_step_and_epoch_it_is_actually_on(tmp_path):
    module = ToyModule()
    trainer, _ = run(tmp_path, module=module)
    assert module.last_ctx is not None
    assert module.last_ctx.epoch == 2
    assert module.last_ctx.step == 7, "zero-based, and the 8th step is index 7"


def _spy_on_the_reduce_gate(monkeypatch) -> list[bool]:
    """Record every ``enabled`` the engine passes to its own suppression gate.

    This used to be recorded by the model, because the model took the backward. The engine
    owns both now (ADR 0006), so the assertion has to watch the engine -- which is also the
    only place it was ever really true.
    """
    from wcfm.engine.trainer import Trainer

    seen: list[bool] = []
    real = Trainer._no_backward_sync

    def spy(self, enabled):
        seen.append(bool(enabled))
        return real(self, enabled)

    monkeypatch.setattr(Trainer, "_no_backward_sync", spy)
    return seen


def test_accumulation_steps_once_per_window_and_suppresses_the_reduce_between(
    tmp_path, monkeypatch
):
    """The reduce is suppressed for every microstep but the last of a window."""
    seen = _spy_on_the_reduce_gate(monkeypatch)
    module = ToyModule()
    trainer, result = run(tmp_path, module=module, optim__accumulate_grad_batches=2)

    assert result["steps"] == 4, "4 batches per epoch at accumulate=2 is 2 steps, x2 epochs"
    assert module.step_ends == 4
    # One entry per microstep now, not per (microstep, view): 8 batches over the run,
    # alternating suppressed / not.
    assert seen == [True, False] * 4


def test_several_views_take_one_backward_between_them(tmp_path, monkeypatch):
    """The inverse of what this file asserted until 2026-09-10.

    It used to require a backward per view, "because feeding several forwards to one backward
    over-reduces gradients under DDP (ADR 0001)". The DDP fact is true; the conclusion was
    not, and it was never DINO's shape. A model now runs every view inside ONE wrapped
    forward and the engine takes ONE backward -- so the view count must NOT appear in the
    gate sequence at all.
    """
    seen = _spy_on_the_reduce_gate(monkeypatch)
    module = ToyModule(views=3)
    trainer, result = run(tmp_path, module=module)
    assert len(seen) == result["steps"], (
        f"{len(seen)} backwards for {result['steps']} steps at views=3: the view count is "
        "leaking into the backward again"
    )
    assert seen == [False] * result["steps"], "accumulate=1, so no microstep is suppressed"


def test_schedules_are_applied_and_recorded_per_step(tmp_path):
    """The learning rate actually applied is the first number anyone reads when a run
    diverges, and the engine already knows it -- so the engine writes it."""
    trainer, _ = run(tmp_path, metrics__step_cadence=1)
    rows = _stream(trainer, "step")
    assert len(rows) == 8
    assert rows[0]["lr"] == pytest.approx(1e-3)
    assert rows[0]["weight_decay"] == pytest.approx(0.04)
    assert rows[-1]["lr"] < rows[0]["lr"], "the cosine decays"
    assert rows[-1]["weight_decay"] > rows[0]["weight_decay"], "weight decay increases"
    # The head's lr_scale=2.0 is reported separately; the unscaled backbone is not.
    assert rows[0]["lr/head"] == pytest.approx(2e-3)
    assert "lr/backbone" not in rows[0]


def test_the_schedule_is_indexed_by_optimizer_step_not_by_batch(tmp_path):
    """With accumulate=2 the same rate is held across both microsteps of a window. The old
    loop indexed by batch (``train_dino.py:741``); at accumulate=1, which every archived run
    used, the two are the same sequence."""
    trainer, _ = run(tmp_path, optim__accumulate_grad_batches=2)
    rows = _stream(trainer, "step")
    assert [r["step"] for r in rows] == [0, 1, 2, 3]


# ------------------------------------------------------------------ the guards


def test_a_nonfinite_gradient_skips_the_step_and_records_the_event_keys(tmp_path):
    """A counter is not a diagnosis: the offending batch's event keys go into the stream,
    which is what makes the failure reproducible against a targeted re-read."""
    module = ToyModule(explode_at=2)
    before = module.net.weight.detach().clone()
    trainer, result = run(tmp_path, module=module)

    assert result["skipped_nonfinite"] == 1
    assert module.step_ends == 7, "on_step_end does not run for a skipped step"
    assert torch.isfinite(module.net.weight).all(), "the skip kept nan out of the parameters"
    assert not torch.equal(module.net.weight.detach(), before)

    flagged = [r for r in _stream(trainer, "step") if r.get("nonfinite_grad")]
    assert len(flagged) == 1
    assert flagged[0]["event_keys"] == [4, 5], "batch 2 of a 2-sample loader"


def test_clipping_bounds_the_gradient_norm_and_records_it(tmp_path):
    trainer, _ = run(tmp_path, optim__clip_grad_norm=1e-4)
    rows = _stream(trainer, "step")
    assert all("grad_norm" in r for r in rows[1:]), "the measured norm is a recorded column"


def test_event_keys_survive_a_batch_without_meta():
    """A batch shape that carries no meta must not make the guard itself raise."""
    assert _event_keys(ToyBatch(torch.zeros(1, 4))) is None
    assert _event_keys(object()) is None
    assert _event_keys(ToyBatch(torch.zeros(1, 4), {"event_key": [3, 4]})) == [3, 4]


# ------------------------------------------------------------------ checkpoints


def test_save_at_and_save_every_both_reach_the_checkpoint_directory(tmp_path):
    trainer, _ = run(tmp_path, optim__epochs=4, run__save_every=2)
    names = sorted(p.name for p in (tmp_path / "run" / "checkpoints").glob("*.pt"))
    assert names == ["checkpoint_epoch2.pt", "checkpoint_epoch4.pt"]

    trainer, _ = run(tmp_path / "b", optim__epochs=4, run__save_every=0, run__save_at=[3])
    names = sorted(p.name for p in (tmp_path / "b" / "run" / "checkpoints").glob("*.pt"))
    assert names == ["checkpoint_epoch3.pt", "checkpoint_epoch4.pt"], "the last epoch always saves"


def test_resume_continues_the_step_count_and_restores_the_buffer(tmp_path):
    """The buffer is the point. ``on_step_end`` increments ``centre`` once per step, so its
    value is a count of every step the run has ever taken -- exactly what the old repo lost
    when it never saved the centring buffer."""
    first = ToyModule()
    trainer, _ = run(tmp_path, module=first, optim__epochs=2, run__save_every=1)
    assert float(first.centre[0]) == 8.0

    second = ToyModule()
    trainer2 = Trainer(
        cfg(tmp_path, optim__epochs=4, run__save_every=1, run__resume="auto"),
        second,
        fabric=cpu_fabric(),
        loader=toy_loader(steps=4),
        run_dir=tmp_path / "run",
    )
    result = trainer2.fit()

    assert trainer2.start_epoch == 3, "resumed after the epoch the checkpoint recorded"
    assert result["epochs_run"] == 2, "epochs 3 and 4, not 4 from scratch"
    assert trainer2.step == 16
    assert float(second.centre[0]) == 16.0, "the buffer came back and kept counting"


def test_resume_truncates_the_metrics_stream_rather_than_doubling_it(tmp_path):
    """Without this a resumed run has two records for the same step and every plot silently
    averages the pre-eviction and post-eviction values."""
    trainer, _ = run(tmp_path, optim__epochs=2, run__save_every=1, metrics__step_cadence=1)
    assert len(_stream(trainer, "step")) == 8

    trainer2 = Trainer(
        cfg(tmp_path, optim__epochs=3, run__save_every=1, run__resume="auto",
            metrics__step_cadence=1),
        ToyModule(),
        fabric=cpu_fabric(),
        loader=toy_loader(steps=4),
        run_dir=tmp_path / "run",
    )
    trainer2.fit()
    steps = [r["step"] for r in _stream(trainer2, "step")]
    assert steps == sorted(set(steps)), f"duplicated steps after resume: {steps}"
    assert steps == list(range(12))


def test_the_run_directory_and_its_derivations_are_written(tmp_path):
    """``epoch_len``, ``total_iters`` and ``warmup_iters`` land in the recorded config: a
    derivation that only exists in a log line cannot be diffed between two runs."""
    import json

    trainer, _ = run(tmp_path)
    run_dir = tmp_path / "run"
    for sub in ("checkpoints", "debug", "probes", "features", "metrics"):
        assert (run_dir / sub).is_dir()

    from omegaconf import OmegaConf

    recorded = OmegaConf.load(run_dir / "config.yaml")
    assert recorded.run.name == "toy" and recorded.optim.epochs == 2
    derived = json.loads((run_dir / "run_metadata.json").read_text())["derived"]
    assert derived["epoch_len"] == 4 and derived["total_iters"] == 8


# ------------------------------------------------------------------ preemption


def test_sigterm_stops_the_run_and_writes_latest(tmp_path):
    module = ToyModule()
    trainer = Trainer(
        cfg(tmp_path, optim__epochs=4),
        module,
        fabric=cpu_fabric(),
        loader=toy_loader(steps=4),
        run_dir=tmp_path / "run",
    )
    trainer.setup()

    original = trainer._run_epoch

    def _epoch_then_signal(epoch, accumulate, guard):
        signal.raise_signal(signal.SIGTERM)  # arrives mid-epoch, as an eviction would
        return original(epoch, accumulate, guard)

    trainer._run_epoch = _epoch_then_signal
    result = trainer.fit()

    assert result["preempted"] is True
    assert result["epochs_run"] == 1, "stopped in the first epoch, not after all four"
    assert (tmp_path / "run" / "checkpoints" / "latest.pt").exists()
    assert trainer.step == 1, "at most one step is lost, against up to ten epochs today"


def test_the_handler_sets_a_flag_and_restores_the_previous_one():
    """It does not write a checkpoint. A handler runs on whatever stack the signal
    interrupted -- possibly mid-``torch.save`` -- and that is the truncated file that makes
    the next ``--resume auto`` fail."""
    before = signal.getsignal(signal.SIGTERM)
    with PreemptionGuard() as guard:
        assert guard.signalled is False
        signal.raise_signal(signal.SIGTERM)
        assert guard.signalled is True
        assert guard.should_stop() is True
        assert guard.signal_number == signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is before, "the handler leaked out of the context"


def test_the_timing_line_is_a_formatter_over_the_epoch_record(tmp_path):
    """Contract C10: not a second source of truth for it."""
    line = _timing_line(7, {"loss": 0.5, "steps": 4, "elapsed_s": 1.25, "data_wait_frac": None})
    assert line == "[timing] epoch=7 loss=0.5 steps=4 elapsed_s=1.25"


# ------------------------------------------------------------------ collectors


def test_a_configured_collector_is_registered_and_its_output_prefixed(tmp_path):
    trainer, _ = run(
        tmp_path,
        metrics__collectors={
            "tput": {"_target_": "wcfm.metrics.collectors.Throughput", "cadence": "step"}
        },
    )
    rows = _stream(trainer, "step")
    assert any("tput/samples_per_s" in r for r in rows)
    assert all(r.get("tput/samples_per_s", 1) > 0 for r in rows if "tput/samples_per_s" in r)


def test_an_empty_collector_dict_is_a_legitimate_configuration(tmp_path):
    """lr and weight decay are still recorded, because the engine writes them."""
    trainer, _ = run(tmp_path, metrics__collectors={})
    assert trainer.collection.collectors == {}
    assert all("lr" in r for r in _stream(trainer, "step"))


def test_an_epoch_cadence_collector_actually_fires(tmp_path):
    """``cadence="epoch"`` is the ``Collector`` class **default**, and it never fired.

    ``fires()`` answers ``end_of_epoch`` for that cadence, and the loop only ever asked with
    the step -- so an epoch-cadence collector ran zero times, produced no column, and looked
    exactly like one whose needs were unmet. It went unnoticed because every configured
    collector in ``conf/metrics/*.yaml`` names an explicit cadence, and the one shape nobody
    writes by hand is the default.
    """
    trainer, _ = run(
        tmp_path,
        optim__epochs=2,
        metrics__collectors={
            "per_epoch": {"_target_": "wcfm.metrics.collectors.Throughput", "cadence": "epoch"}
        },
    )
    epochs = _stream(trainer, "epoch")
    assert len(epochs) == 2, f"expected one epoch record each, got {len(epochs)}"
    assert all("per_epoch/samples_per_s" in r for r in epochs), (
        f"an epoch-cadence collector produced nothing: {epochs}"
    )
    # And it fired *only* there: an epoch cadence that also landed per step would be a
    # different bug with the same symptom in the epoch record.
    assert not any("per_epoch/samples_per_s" in r for r in _stream(trainer, "step")), (
        "an epoch-cadence collector also wrote into the step stream"
    )


def test_a_step_cadence_collector_does_not_also_fire_at_epoch_end(tmp_path):
    """The end-of-epoch selection is on the **cadence**, not on ``fires(..., True)``.

    ``fires`` answers ``True`` for a ``"step"`` cadence and for an integer cadence the step
    divides, whatever ``end_of_epoch`` says -- so asking it at epoch end fires those a second
    time and writes a per-step rate into the epoch record as if it described the epoch. It
    also makes the selection depend on ``self.step``, which is rank-local, in front of a
    collective. Both were live for the length of one edit on 2026-09-09.
    """
    trainer, _ = run(
        tmp_path,
        optim__epochs=2,
        metrics__step_cadence=1,
        metrics__collectors={
            "per_step": {"_target_": "wcfm.metrics.collectors.Throughput", "cadence": "step"},
            "every_two": {"_target_": "wcfm.metrics.collectors.Throughput", "cadence": 2},
        },
    )
    epochs = _stream(trainer, "epoch")
    assert epochs, "no epoch records were written"
    for row in epochs:
        leaked = [k for k in row if k.startswith(("per_step/", "every_two/"))]
        assert not leaked, f"a step-cadence collector fired at epoch end too: {leaked}"
    # ...and they did fire where they should.
    assert any("per_step/samples_per_s" in r for r in _stream(trainer, "step"))


def test_an_epoch_with_no_steps_reports_no_epoch_cadence_columns(tmp_path):
    """The end-of-epoch record is built from the last completed step, so an epoch that
    completed none must emit nothing rather than material from the epoch before it."""
    trainer = Trainer(
        cfg(
            tmp_path,
            optim__epochs=1,
            metrics__collectors={
                "per_epoch": {
                    "_target_": "wcfm.metrics.collectors.Throughput",
                    "cadence": "epoch",
                }
            },
        ),
        ToyModule(),
        fabric=cpu_fabric(),
        loader=toy_loader(steps=4),
        run_dir=tmp_path / "empty_epoch",
    )
    trainer.setup()
    assert trainer.collection.on_epoch(1) == {}, (
        "an epoch with no completed step produced columns, so they came from stale material"
    )


def _stream(trainer: Trainer, name: str) -> list[dict]:
    import json

    path = trainer.run_dir / "metrics" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ------------------------------------------------------------------ build_fabric

# Every test above injects a CPU Fabric, so these are the only coverage of the strategy the
# real runs construct -- and constructing it wrong is silent failure #1 in the README. No
# launch happens: the object is inspected, which is all that is in question here.


def test_ddp_strategy_is_constructed_with_the_unused_parameters_flag(tmp_path):
    """`strategy="ddp"` as a string would take Fabric's default
    `find_unused_parameters=False`, and ADR 0001's frozen 2-GPU reference is exactly what that
    produces: no reduction at all, silently. `_ddp_kwargs` is checked against
    lightning_fabric 2.6.5."""
    from lightning_fabric.strategies import DDPStrategy

    from wcfm.engine.trainer import build_fabric

    fabric = build_fabric(cfg(tmp_path, launch__devices=2))
    assert isinstance(fabric.strategy, DDPStrategy)
    assert fabric.strategy._ddp_kwargs == {
        "find_unused_parameters": True,
        "static_graph": False,
    }


def test_static_graph_reaches_the_strategy(tmp_path):
    from wcfm.engine.trainer import build_fabric

    fabric = build_fabric(cfg(tmp_path, launch__devices=2, launch__static_graph=True))
    assert fabric.strategy._ddp_kwargs["static_graph"] is True


def test_multi_node_gets_ddp_even_at_one_device_per_node(tmp_path):
    """Deferred, not refused: `num_nodes` is on the contract so nothing forecloses it, and a
    single device per node must still be wrapped."""
    from lightning_fabric.strategies import DDPStrategy

    from wcfm.engine.trainer import build_fabric

    fabric = build_fabric(cfg(tmp_path, launch__devices=1, launch__num_nodes=2))
    assert isinstance(fabric.strategy, DDPStrategy)


def test_one_device_does_not_get_a_ddp_wrapper(tmp_path):
    from lightning_fabric.strategies import DDPStrategy

    from wcfm.engine.trainer import build_fabric

    fabric = build_fabric(cfg(tmp_path, launch__devices=1))
    assert not isinstance(fabric.strategy, DDPStrategy)


# ------------------------------------------------------------------ resume edges


def test_resuming_a_finished_run_does_nothing_instead_of_crashing(tmp_path):
    """A sweep re-submitted after it completed hits this every time. The loop body never runs,
    so `epoch` would be unbound in the summary -- an UnboundLocalError forty hours into a
    queue, on the one path that is meant to be a no-op."""
    trainer, _ = run(tmp_path, optim__epochs=2, run__save_every=1)

    again = Trainer(
        cfg(tmp_path, optim__epochs=2, run__resume="auto"),
        ToyModule(),
        fabric=cpu_fabric(),
        loader=toy_loader(steps=4),
        run_dir=tmp_path / "run",
    )
    result = again.fit()
    assert result["epochs_run"] == 0
    assert result["steps"] == 8, "the finished run's step count is reported, not zero"
    assert again.start_epoch == 3


def test_resuming_with_fewer_epochs_than_the_checkpoint_reached_is_also_a_no_op(tmp_path):
    trainer, _ = run(tmp_path, optim__epochs=4, run__save_every=1)
    again = Trainer(
        cfg(tmp_path, optim__epochs=2, run__resume="auto"),
        ToyModule(),
        fabric=cpu_fabric(),
        loader=toy_loader(steps=4),
        run_dir=tmp_path / "run",
    )
    assert again.fit()["epochs_run"] == 0


def test_a_rank_without_saved_rng_state_reseeds_deterministically(tmp_path):
    """Only rank 0 writes a checkpoint, so only rank 0's RNG state is in it. A non-zero rank
    must not be left wherever its stream happened to be, and must not be handed rank 0's --
    that would correlate the ranks' augmentation streams after every resume."""
    trainer, _ = run(tmp_path, optim__epochs=2, run__save_every=1)

    ckpt_path = tmp_path / "run" / "checkpoints" / "checkpoint_epoch2.pt"
    blob = torch.load(ckpt_path, weights_only=False)
    assert list(blob["rng"]) == ["rank0"], "the file holds one rank's state, as documented"

    blob["rng"] = {}  # what a non-zero rank sees
    torch.save(blob, ckpt_path)

    resumed = Trainer(
        cfg(tmp_path, optim__epochs=3, run__resume="auto"),
        ToyModule(),
        fabric=cpu_fabric(),
        loader=toy_loader(steps=4),
        run_dir=tmp_path / "run",
    )
    resumed.setup()
    first = torch.randn(3)

    # Same config, same resume epoch: the reseed is a function of both, so it repeats.
    torch.manual_seed(0)
    again = Trainer(
        cfg(tmp_path, optim__epochs=3, run__resume="auto"),
        ToyModule(),
        fabric=cpu_fabric(),
        loader=toy_loader(steps=4),
        run_dir=tmp_path / "run",
    )
    again.setup()
    assert torch.equal(torch.randn(3), first), "the reseed must be reproducible"


# ------------------------------------------------------------------ precision

# `run.precision` is applied by Fabric as an autocast inside the wrapper's forward, so whether
# it reaches the model is a question about the engine, not about a device -- `torch.autocast`
# supports "cpu" with bfloat16. These belong here rather than in the GPU suite, because a
# setting that silently does nothing is exactly what a CPU test should be able to catch.


def test_the_configured_precision_reaches_the_model(tmp_path):
    """``bf16-mixed`` was inert for the whole life of the engine until `c0eb8f0`, and no test
    noticed -- the one that "passed on precision" asserted that the loop ran, not the dtype it
    ran in.

    **The output dtype cannot answer this.** Three things upcast a bf16 result on the way out:
    a module's fp32 bias winning the type promotion on an add, `_FabricModule.forward` calling
    `precision.convert_output` (`wrappers.py:138`) which casts back to the default dtype, and
    whatever the caller then does in fp32. Each hid it in turn. So the assertion is on whether
    autocast was *enabled* inside the forward.
    """
    from lightning_fabric import Fabric

    seen: set[tuple[bool, torch.dtype]] = set()

    class Recording(ToyModule):
        def training_step(self, batch, ctx):
            out = super().training_step(batch, ctx)
            seen.add(self._autocast)
            return out

    Trainer(
        cfg(tmp_path, optim__epochs=1, run__precision="bf16-mixed"),
        Recording(),
        fabric=Fabric(accelerator="cpu", devices=1, precision="bf16-mixed"),
        loader=toy_loader(steps=2),
        run_dir=tmp_path / "amp_run",
    ).fit()

    assert seen == {(True, torch.bfloat16)}, (
        f"the forward saw autocast {seen}, expected (True, bfloat16): run.precision did not "
        "reach the model"
    )


def test_mixed_precision_leaves_the_inputs_alone(tmp_path):
    """Fabric's own `MixedPrecision` casts the wrapper's floating inputs to the half type
    before the forward, on top of the autocast. `build_fabric` installs
    `AutocastOnlyPrecision` instead: the forward must see autocast enabled and its arguments
    in the dtype the caller passed. A point cloud's coordinates are floats of order one, and
    a bfloat16 cast of them moves pixels by up to a few pitches."""
    from wcfm.engine.trainer import build_fabric

    seen: set[tuple[torch.dtype, bool, torch.dtype]] = set()

    class Recording(ToyModule):
        def forward(self, x):
            out = super().forward(x)
            seen.add((x.dtype, *self._autocast))
            return out

    Trainer(
        cfg(tmp_path, optim__epochs=1, run__precision="bf16-mixed"),
        Recording(),
        fabric=build_fabric(cfg(tmp_path, run__precision="bf16-mixed")),
        loader=toy_loader(steps=2),
        run_dir=tmp_path / "amp_inputs_run",
    ).fit()

    assert seen == {(torch.float32, True, torch.bfloat16)}, (
        f"the forward saw (input dtype, autocast) = {seen}; expected float32 inputs under a "
        "bfloat16 autocast"
    )


def test_fp32_really_has_no_autocast(tmp_path):
    """The control for the test above: at ``32-true`` -- now the default -- there must be no
    autocast, so a pass at ``bf16-mixed`` cannot be an accident of always-on autocast, and the
    new default cannot be quietly running mixed anyway."""
    from lightning_fabric import Fabric

    seen: set[tuple[bool, torch.dtype]] = set()

    class Recording(ToyModule):
        def training_step(self, batch, ctx):
            out = super().training_step(batch, ctx)
            seen.add(self._autocast)
            return out

    Trainer(
        cfg(tmp_path, optim__epochs=1, run__precision="32-true"),
        Recording(),
        fabric=Fabric(accelerator="cpu", devices=1, precision="32-true"),
        loader=toy_loader(steps=2),
        run_dir=tmp_path / "fp32_run",
    ).fit()

    assert all(enabled is False for enabled, _ in seen), (
        f"autocast was enabled at 32-true: {seen}. The bf16 test above would then pass "
        "whatever the config said."
    )


def test_computing_through_self_gets_no_autocast_at_all(tmp_path):
    """The other half of `c0eb8f0`, on a CPU: bypassing ``ctx.module`` loses the precision as
    well as the gradient reduction. One bug, two silent failures -- and this is the one a
    single-device test can see, so it is asserted here rather than left to the GPU suite."""
    from lightning_fabric import Fabric

    seen: set[tuple[bool, torch.dtype]] = set()

    class Bypasses(ToyModule):
        def training_step(self, batch, ctx):
            rows = self.rows_of(batch)
            out = self.head(self.net(rows))  # NOT ctx.module, on purpose
            device_type = "cuda" if rows.is_cuda else "cpu"
            seen.add((torch.is_autocast_enabled(device_type),
                      torch.get_autocast_dtype(device_type)))
            loss = (out.float() ** 2).mean()
            return StepOutput(
                scalars={"loss": float(loss.detach())},
                n_samples=batch.batch_size,
                loss=loss,
            )

    Trainer(
        cfg(tmp_path, optim__epochs=1, run__precision="bf16-mixed"),
        Bypasses(),
        fabric=Fabric(accelerator="cpu", devices=1, precision="bf16-mixed"),
        loader=toy_loader(steps=2),
        run_dir=tmp_path / "bypass_run",
    ).fit()

    assert all(enabled is False for enabled, _ in seen), (
        f"expected no autocast when bypassing the wrapper, saw {seen}. If this ever passes "
        "with autocast enabled, precision no longer depends on ctx.module and the test above "
        "stops being evidence for anything."
    )


def test_a_launcher_world_that_disagrees_with_launch_devices_is_fatal(tmp_path, monkeypatch):
    """`launch.devices` is what Fabric builds; WORLD_SIZE is what actually exists.

    Nothing compared them until 2026-09-10. `launch.devices=1` under
    `torchrun --nproc_per_node=2` gave two processes that each believed they were alone,
    wrote the same run directory and reduced nothing -- the ADR 0001 silent-no-reduction
    failure arriving by a route the DDP wiring is not involved in. Checked before a device is
    touched, so it costs seconds rather than an epoch.
    """
    from wcfm.engine.trainer import build_fabric

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="launch.devices=1 but the launcher started"):
        build_fabric(cfg(tmp_path, launch__devices=1))


def test_a_launcher_world_that_agrees_is_accepted(tmp_path, monkeypatch):
    """The control: the guard must not fire on the configuration it exists to allow, or it
    would be indistinguishable from a hard refusal of every multi-rank run."""
    from wcfm.engine.trainer import build_fabric

    monkeypatch.setenv("WORLD_SIZE", "1")
    assert build_fabric(cfg(tmp_path, launch__devices=1)) is not None
    monkeypatch.delenv("WORLD_SIZE")
    assert build_fabric(cfg(tmp_path, launch__devices=1)) is not None
