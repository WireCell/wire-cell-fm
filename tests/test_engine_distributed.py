"""Two CUDA devices: the four paths a single-device suite cannot reach.

Run as ``torchrun --nproc_per_node=2 -m pytest -m distributed``, so **every rank is a separate
pytest process running this whole file**.

**These do not need a GPU.** DDP's gradient reduction and the arming of its reducer behave
identically over gloo on a CPU: measured 2026-09-08 with a control, the divergence from
bypassing the wrapper is ~3.9e-02 on two CPU ranks and on two L40S alike. The marker therefore
skips on *rank count*, not on device count, and the 2-GPU Condor job is confirmation rather
than the only signal -- which matters, because that slot is scarce enough that the first
attempt to verify a correctness fix sat idle for ninety minutes. Three rules follow, and
breaking any of them produces a hang rather than a failure:

* **Every assertion is on a value all ranks hold**, typically a reduced one. ``if rank == 0:
  assert ...`` lets one rank leave the test while the other waits in the next collective, and
  the job then burns its wall clock with no failure message.
* **No fixture-local temporary directory.** ``tmp_path`` is per process; ``shared_tmp`` is per
  job. See ``conftest.py``.
* **The world size is asserted, not assumed** (``two_rank_fabric``). Every test here says
  "the ranks agreed"; one rank always agrees with itself, so a job that landed on one GPU
  would report the whole suite green -- the precise false confidence this file exists to
  remove.

Two of these are regression tests for bugs found by reading rather than by running, which is
the reason to distrust the rest of the multi-GPU paths until they are covered too.
"""

from __future__ import annotations

import json
import signal

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from wcfm.engine.preempt import PreemptionGuard  # noqa: E402
from wcfm.metrics.base import Collector  # noqa: E402

from .fake_dino import fake_loader, sync_enabled  # noqa: E402
from .test_engine import cfg  # noqa: E402
from .toy import ToyModule, TwoPartToyModule, toy_loader  # noqa: E402

# `distributed` only. The `gpu` marker used to be here too, and it was what kept this file
# hostage to a CUDA device: DDP's reduction and its per-forward arming are backend-independent,
# so gloo on two CPU ranks tests the same thing (measured, with a control).
pytestmark = pytest.mark.distributed


def _trainer(fabric, run_dir, module=None, loader=None, **over):
    from wcfm.engine.trainer import Trainer

    return Trainer(
        cfg(run_dir.parent, launch__devices=2, **over),
        module or ToyModule(),
        fabric=fabric,
        loader=loader if loader is not None else toy_loader(steps=8),
        run_dir=run_dir,
    )


# ------------------------------------------------------------------ the DDP reduce


@pytest.mark.parametrize("views", [1, 3])
def test_ddp_keeps_every_ranks_parameters_identical(two_rank_fabric, shared_tmp, views):
    """The ADR 0001 assertion, run rather than argued -- and the regression test for the bug
    cluster 2250 found: the forward went through the *unwrapped* module, so DDP's reducer was
    never armed and the ranks diverged by 8.8e-3 after one epoch, silently.

    **``views=3`` is the case that matters.** DDP arms once per ``forward()``, so a fix that
    routes a single forward through the wrapper passes at ``views=1`` and breaks the moment a
    model does a backward per view -- which is the shape this whole design exists for (ADR
    0001), and which raises "Expected to have finished reduction in the prior iteration" if
    each view does not get its own arming.
    """
    fabric = two_rank_fabric
    module = ToyModule(views=views)
    result = _trainer(
        fabric, shared_tmp / f"ddp_run_v{views}", module=module, optim__epochs=1
    ).fit()

    assert result["steps"] > 0
    flat = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    gathered = fabric.all_gather(flat)

    assert gathered.shape[0] == 2, "all_gather did not return one row per rank"
    spread = float((gathered[0] - gathered[1]).abs().max())
    assert spread == 0.0, (
        f"the ranks' parameters differ by {spread:.3e} after training with views={views}: "
        "the gradient all-reduce did not happen. Either the forward bypassed ctx.module, or "
        "only the first view armed the reducer. See ADR 0001."
    )


def test_build_fabric_produces_a_strategy_that_actually_reduces(shared_tmp):
    """``build_fabric`` is what production calls, and its DDP construction was covered on CPU
    only by inspecting the object. This runs the thing it built."""
    from wcfm.engine.trainer import build_fabric

    fabric = build_fabric(cfg(shared_tmp, launch__devices=2, run__precision="32-true"))
    fabric.launch()
    assert fabric.world_size == 2, "build_fabric(devices=2) did not give two ranks"

    module = ToyModule()
    _trainer(fabric, shared_tmp / "built_run", module=module, optim__epochs=1).fit()

    flat = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    gathered = fabric.all_gather(flat)
    assert float((gathered[0] - gathered[1]).abs().max()) == 0.0


def test_fabric_does_not_add_a_second_distributed_sampler(two_rank_fabric, shared_tmp):
    """``build_loader`` already makes a ``DistributedSampler`` for the map-style backends, and
    the sharded backend is an ``IterableDataset``. Fabric's ``setup_dataloaders`` would add its
    own on top -- and if it did, each rank would see a quarter of the data rather than half,
    with no error anywhere. Pinned because a Fabric upgrade could change it silently."""
    from torch.utils.data import DataLoader, DistributedSampler

    from .toy import ToyDataset, toy_collate

    dataset = ToyDataset(16, 4)
    sampler = DistributedSampler(dataset, num_replicas=2, rank=two_rank_fabric.global_rank)
    loader = DataLoader(dataset, batch_size=2, sampler=sampler, collate_fn=toy_collate)

    prepared = two_rank_fabric.setup_dataloaders(loader, move_to_device=False)
    assert prepared.sampler is sampler, "Fabric replaced the sampler build_loader created"


# ------------------------------------------------------------------ the guards


def test_a_nonfinite_gradient_on_one_rank_stops_both(two_rank_fabric, shared_tmp):
    """``_grads_finite`` all-reduces with MIN so the verdict is the group's.

    A rank-local answer would let one rank step while the other skipped, and the two would
    hold different parameters from then on -- with nothing downstream saying so, because only
    rank 0 writes metrics. The explosion is deliberately on rank 1, the rank that does *not*
    write, which is the asymmetry that makes the bug invisible.
    """
    fabric = two_rank_fabric

    class ExplodesOnRankOne(ToyModule):
        def training_step(self, batch, ctx):
            self.explode_at = 0 if fabric.global_rank == 1 else None
            return super().training_step(batch, ctx)

    module = ExplodesOnRankOne()
    trainer = _trainer(fabric, shared_tmp / "nan_run", module=module, optim__epochs=1)
    result = trainer.fit()

    # Rank-symmetric: both ranks must report the same skip count, which is what "the group
    # agreed" means. A rank-local check gives 1 on rank 1 and 0 on rank 0.
    skipped = fabric.all_gather(
        torch.tensor([float(result["skipped_nonfinite"])], device=fabric.device)
    )
    assert float(skipped.min()) == float(skipped.max()) == 1.0, (
        f"ranks disagree on the skip: {skipped.flatten().tolist()}. The non-finite verdict "
        "is rank-local, so the ranks now hold different parameters."
    )

    flat = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    gathered = fabric.all_gather(flat)
    assert float((gathered[0] - gathered[1]).abs().max()) == 0.0, (
        "the skip did not keep the ranks in step"
    )


def test_a_sigterm_on_one_rank_stops_the_whole_group(two_rank_fabric):
    """Whether a rank saw SIGTERM is rank-local information, and branching on it is exactly
    the shape that hangs a job: one rank leaves the loop while the other waits in the next
    reduce. ``should_stop`` all-reduces with MAX, so either all stop or none do."""
    fabric = two_rank_fabric
    with PreemptionGuard() as guard:
        if fabric.global_rank == 1:
            signal.raise_signal(signal.SIGTERM)

        # Collective: every rank calls it, so this cannot hang on an asymmetric branch.
        stop = guard.should_stop(fabric.all_reduce)
        assert stop is True, (
            "a SIGTERM delivered to one rank did not stop this one: the vote is rank-local, "
            "so one rank would checkpoint and exit while the other hung in a reduce"
        )

    with PreemptionGuard() as quiet:
        assert quiet.should_stop(fabric.all_reduce) is False, "stopped with no signal at all"


# ------------------------------------------------------------------ the reduction


class _RankZeroOnly(Collector):
    """Emits its declared key on rank 0 alone -- ``ArrayDump``'s real shape after ``067cfe2``,
    since only rank 0 is given an ``arrays_dir``."""

    cadence = "step"
    reduce = {"only_on_zero": "sum", "everywhere": "sum"}

    def __init__(self, rank: int):
        self.rank = rank

    def compute(self, rec):
        out = {"everywhere": 1.0}
        if self.rank == 0:
            out["only_on_zero"] = 10.0
        return out


def test_reducing_declared_keys_does_not_hang_when_one_rank_produces_nothing(
    two_rank_fabric, shared_tmp
):
    """The regression test for ``067cfe2``, and the one that could not exist on one device.

    Reducing the keys a collector *returned* made rank 0 call ``all_reduce`` for
    ``only_on_zero`` while rank 1 did not -- and the group never recovers from that. Reducing
    the *declared* keys, a class attribute, makes the collective sequence identical on every
    rank by construction. If this test hangs, the fix has been undone.
    """
    fabric = two_rank_fabric
    trainer = _trainer(fabric, shared_tmp / "reduce_run", optim__epochs=1)
    trainer.setup()
    collector = _RankZeroOnly(fabric.global_rank)

    reduced = trainer.collection._reduce_declared(collector, collector.compute(None))

    # `everywhere` is produced by both ranks: 1 + 1.
    assert reduced["everywhere"] == pytest.approx(2.0)
    if fabric.global_rank == 0:
        # Written only where it was measured, and carrying the sum over the ranks that had it.
        assert reduced["only_on_zero"] == pytest.approx(10.0)
    else:
        assert "only_on_zero" not in reduced, (
            "a rank that did not measure the key must not report a column for it"
        )


class _NeedsAnObservable(Collector):
    """Declares a need and a reduction, so an unmet need must not skip the collective."""

    cadence = "step"
    needs = frozenset({"toy/feat"})
    reduce = {"rows": "sum"}

    def compute(self, rec):
        return {"rows": float(rec.observables["toy/feat"].shape[0])}


def test_an_unmet_need_on_one_rank_only_does_not_hang(two_rank_fabric, shared_tmp):
    """``unmet`` reads the record's observable **keys**, which are rank-local.

    ``module.observables()`` is the model's own dict, so a model that omits a key on one rank
    -- a tap that only exists on the rank holding a shard with rows in it, a diagnostic built
    behind a data-dependent branch -- makes ``unmet`` disagree across ranks. Skipping the
    collector on that rank used to skip its turn in the collective sequence too, while the
    other rank walked into ``_reduce_declared``'s ``all_reduce``: a permanent hang, not a
    failure. Now an unmet need produces ``{}`` and the reduction still runs, so the sequence
    is a pure function of the class attribute ``reduce``.

    **A hang here is the failure mode**, which shows up as this test timing out with no
    message -- the reason the job script wraps each stage in ``timeout``.
    """
    fabric = two_rank_fabric

    class OmitsOnRankOne(ToyModule):
        def observables(self):
            # Rank 1 withholds the key the collector needs.
            return {} if fabric.global_rank == 1 else super().observables()

    trainer = _trainer(
        fabric,
        shared_tmp / "unmet_run",
        module=OmitsOnRankOne(),
        optim__epochs=1,
        metrics__step_cadence=1,
    )
    trainer.setup()
    trainer.collection.collectors = {"needy": _NeedsAnObservable()}
    trainer.fit()

    # Reaching here at all is the claim. Asserted on both ranks with a collective, because a
    # rank-local assert would leave the other one waiting in the next one.
    reached = fabric.all_reduce(torch.tensor([1.0], device=fabric.device), reduce_op="sum")
    assert float(reached.item()) == 2.0, "a rank did not survive the unmet-need dispatch"

    if fabric.is_global_zero:
        rows = [
            json.loads(line)
            for line in (trainer.run_dir / "metrics" / "step.jsonl").read_text().splitlines()
            if line.strip()
        ]
        measured = [r for r in rows if "needy/rows" in r]
        assert measured, "rank 0 met the need and should have written the column"
        # The reduced sum carries rank 0's rows only: rank 1 contributed the identity.
        assert measured[-1]["needy/rows"] > 0


def test_an_epoch_cadence_collector_reduces_without_hanging(two_rank_fabric, shared_tmp):
    """The end-of-epoch firing reduces, so both ranks must walk the same collectives.

    ``_collect_epoch`` selects on the cadence rather than on ``fires(..., end_of_epoch=True)``,
    which keeps the selection independent of ``self.step`` -- a rank-local number -- and its
    "did this rank complete a step" guard is all-reduced with MIN. Either would otherwise be a
    rank-local branch in front of ``_reduce_declared``'s ``all_reduce``, which is the shape
    that hangs a job permanently rather than failing it. A hang here shows as this test
    timing out with no message, which is why the job script wraps each stage in ``timeout``.
    """
    fabric = two_rank_fabric
    trainer = _trainer(
        fabric,
        shared_tmp / "epoch_cadence",
        optim__epochs=1,
        metrics__collectors={
            "per_epoch": {"_target_": "wcfm.metrics.collectors.Throughput", "cadence": "epoch"}
        },
    )
    trainer.fit()

    # Rank 0 is the only writer, so read the reduced value where it landed.
    if fabric.is_global_zero:
        rows = [
            json.loads(line)
            for line in (trainer.run_dir / "metrics" / "epoch.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert rows, "no epoch record was written"
        assert "per_epoch/samples_per_s" in rows[-1], (
            f"the epoch-cadence collector produced nothing: {rows[-1]}"
        )

    # Asserted on every rank, not just the writer: reaching here at all is the claim, and a
    # rank-local `assert` would leave the other one waiting in the next collective.
    reached = fabric.all_reduce(torch.tensor([1.0], device=fabric.device), reduce_op="sum")
    assert float(reached.item()) == 2.0, "a rank did not reach the end of the epoch firing"


def test_throughput_sums_across_ranks_so_the_number_is_the_jobs(two_rank_fabric, shared_tmp):
    """``samples_per_s`` declares ``sum``: the rate anyone cares about is the job's, not one
    GPU's. On a single device the declaration is untested by construction."""
    from wcfm.metrics.base import StepRecord
    from wcfm.metrics.collectors import Throughput

    trainer = _trainer(two_rank_fabric, shared_tmp / "tput_run", optim__epochs=1)
    trainer.setup()

    collector = Throughput(cadence="step")
    produced = collector.compute(
        StepRecord(step=0, epoch=1, scalars={"n_samples": 4}, timing={"step": 1.0})
    )
    assert produced["samples_per_s"] == pytest.approx(4.0), "this rank's own rate"

    reduced = trainer.collection._reduce_declared(collector, produced)
    assert reduced["samples_per_s"] == pytest.approx(8.0), "the job's rate, over both ranks"


# ------------------------------------------------------------------ checkpoints


def test_only_rank_zero_writes_and_both_ranks_resume_from_it(two_rank_fabric, shared_tmp):
    """Two ranks writing the same checkpoint path concurrently is the ``ArrayDump`` bug in
    another costume. And the resume has to work from the one file that exists: every non-zero
    rank finds no RNG entry of its own and reseeds, which must not raise."""
    fabric = two_rank_fabric
    run_dir = shared_tmp / "ckpt_run"

    _trainer(fabric, run_dir, optim__epochs=1, run__save_every=1).fit()
    fabric.barrier()

    checkpoints = sorted(p.name for p in (run_dir / "checkpoints").glob("*.pt"))
    assert checkpoints == ["checkpoint_epoch1.pt"], f"unexpected checkpoints: {checkpoints}"

    resumed = _trainer(fabric, run_dir, optim__epochs=2, run__save_every=1, run__resume="auto")
    result = resumed.fit()
    assert resumed.start_epoch == 2, "both ranks read the single checkpoint rank 0 wrote"
    assert result["epochs_run"] == 1

    steps = fabric.all_gather(torch.tensor([float(resumed.step)], device=fabric.device))
    assert float(steps.min()) == float(steps.max()), "the ranks resumed to different steps"


def test_the_control_diverges_so_this_suite_can_detect_its_own_failure(
    two_rank_fabric, shared_tmp
):
    """**The control.** A model that computes through ``self`` instead of ``ctx.module`` must
    diverge -- if it does not, every other assertion in this file proves nothing.

    This is spike (c)'s pattern: that spike gated its whole result on
    ``find_unused_parameters=False`` failing, on the grounds that a probe which cannot detect
    the failure it exists to detect is uninformative. The same applies here, and more sharply,
    because the bug this suite found was invisible for exactly one reason: the two spellings
    compute identical numbers, and differ only in whether DDP was armed.

    Measured on two CPU ranks over gloo and on two L40S: the divergence is ~3.9e-02 either
    way, which is why this suite no longer needs a GPU.
    """
    fabric = two_rank_fabric

    class BypassesTheWrapper(ToyModule):
        def training_step(self, batch, ctx):
            # Deliberately wrong: the DDP wrapper is never entered, so `prepare_for_backward`
            # never runs and the reducer's hooks return early on every parameter.
            rows = self.rows_of(batch)
            total = None
            for _view in range(self.views):
                out = self.head(self.net(rows))  # NOT ctx.module: the bug, on purpose
                loss = (out.float() ** 2).mean()
                total = loss if total is None else total + loss
            from wcfm.engine.protocol import StepOutput

            assert total is not None
            return StepOutput(
                scalars={"loss": float(total.detach()) / self.views},
                n_samples=batch.batch_size,
                loss=total / self.views,
            )

    module = BypassesTheWrapper()
    _trainer(fabric, shared_tmp / "control_run", module=module, optim__epochs=1).fit()

    flat = torch.cat([p.detach().reshape(-1) for p in module.parameters()])
    gathered = fabric.all_gather(flat)
    spread = float((gathered[0] - gathered[1]).abs().max())

    assert spread > 1e-6, (
        f"the control did NOT diverge (spread={spread:.3e}). Either the ranks are being fed "
        "identical batches -- in which case nothing here tests reduction -- or something else "
        "is reducing gradients. Every other assertion in this file is uninformative until "
        "this one fails the way it is supposed to."
    )


def test_the_wrapped_handle_the_model_is_given_is_the_ddp_one(two_rank_fabric, shared_tmp):
    """What ``ctx.module`` actually is, asserted rather than assumed.

    The bug cluster 2250 found was invisible because ``self.head(self.net(x))`` and
    ``ctx.module(x)`` compute the same numbers on one device and differ only in whether DDP
    was armed. So this checks the identity of the handle directly: it must be Fabric's
    wrapper, not the module the engine was handed.
    """
    from torch.nn.parallel import DistributedDataParallel

    fabric = two_rank_fabric
    seen: dict[str, object] = {}

    class Recording(ToyModule):
        def training_step(self, batch, ctx):
            seen["module"] = ctx.module
            return super().training_step(batch, ctx)

    module = Recording()
    _trainer(fabric, shared_tmp / "handle_run", module=module, optim__epochs=1).fit()

    handed = seen["module"]
    assert handed is not module, "the model was handed itself; DDP would never arm"
    assert isinstance(getattr(handed, "_forward_module", None), DistributedDataParallel), (
        f"ctx.module wraps {type(getattr(handed, '_forward_module', None)).__name__}, "
        "not DistributedDataParallel"
    )


# --------------------------------------------- the DINO shape, on the one DDP path
#
# `fake_dino.DinoShapedModule` is the shape Stage 3's native DINO has -- an EMA teacher, a
# projection head, a backward per view, a lazily-created centring buffer, a momentum schedule
# from `ctx.extra` -- written the way a model should be written: the student path behind
# `forward`, reached per view through `ctx.module`.
#
# These tests were originally written against a second DDP path, a `wrap_submodules` hook for
# a model that wrapped its own named submodules in place. That path existed because the old
# repo's `DINODuneModel` computes through fixed `self.student(...)` call sites, and it went
# with the shim that imported that model. Every property it was holding is asserted here
# instead, on the one path: per-view arming, a frozen teacher no reducer touches, the two
# suppression gates nesting, the centring buffer, and the checkpoint round trip.


def _dino(dim=4, n_pairs=3, use_proj_head=True, **over):
    from .fake_dino import DinoShapedModule

    return DinoShapedModule(dim=dim, n_pairs=n_pairs, use_proj_head=use_proj_head, **over)


def _spy_on_the_engines_gate(monkeypatch) -> list[bool]:
    """Record DDP's ``require_backward_grad_sync`` as it stands during each engine backward.

    The model used to record this, because the model took the backward. The engine owns it
    now (ADR 0006), so the observation moves with it -- and this watches the flag from
    *inside* the suppression context, which is the state the backward actually sees.
    """
    import contextlib as _ctx

    from wcfm.engine.trainer import Trainer

    seen: list[bool] = []
    real = Trainer._no_backward_sync

    @_ctx.contextmanager
    def spy(self, enabled):
        with real(self, enabled):
            seen.append(sync_enabled(self.wrapped))
            yield

    monkeypatch.setattr(Trainer, "_no_backward_sync", spy)
    return seen


@pytest.mark.parametrize("n_pairs", [1, 3])
def test_a_dino_shaped_module_reduces_once_however_many_views(
    two_rank_fabric, shared_tmp, n_pairs, monkeypatch
):
    """A model with named submodules and several views, held to the same standard as the
    toy: after one epoch the ranks must be bit-identical.

    ``n_pairs=3`` is the case that matters. Every view goes through one ``ctx.module`` call
    and the engine takes ONE backward, so the number of reduces must not depend on the view
    count. A model that computed through its own submodules would reduce nothing at all,
    silently, which is what the control below asserts is detectable.
    """
    fabric = two_rank_fabric
    seen = _spy_on_the_engines_gate(monkeypatch)
    module = _dino(n_pairs=n_pairs)
    result = _trainer(
        fabric,
        shared_tmp / f"dino_run_p{n_pairs}",
        module=module,
        loader=fake_loader(steps=8),
        optim__epochs=1,
    ).fit()

    assert result["steps"] > 0
    # No accumulation, so every microstep reduces -- once, whatever n_pairs is.
    assert len(seen) == result["steps"], (
        f"{len(seen)} backwards for {result['steps']} steps at n_pairs={n_pairs}: the view "
        "count is leaking into the backward"
    )
    assert all(seen), f"DDP's sync flag went {seen}; every backward here must reduce"

    trainable = torch.cat(
        [p.detach().reshape(-1) for p in module.parameters() if p.requires_grad]
    )
    gathered = fabric.all_gather(trainable)
    spread = float((gathered[0] - gathered[1]).abs().max())
    assert spread == 0.0, (
        f"the ranks' student parameters differ by {spread:.3e} with n_pairs={n_pairs}. Either "
        "a view computed on `self` instead of `ctx.module`, or only the first view armed."
    )


def test_the_dino_shaped_control_diverges(two_rank_fabric, shared_tmp):
    """**The control.** The same shape computing on its own submodules must diverge.

    ``TwoPartToyModule`` calls ``self.student(...)`` and ``self.student_head(...)`` from its
    step -- exactly as ``DINODuneModel.forward_backward`` does -- and so arms nothing. It
    computes identical numbers to the correct spelling and differs only in whether DDP was
    armed, which is the entire reason the bug cluster 2250 found was invisible. Until this
    fails the way it is meant to, the test above proves nothing.
    """
    fabric = two_rank_fabric
    module = TwoPartToyModule(views=3)
    _trainer(fabric, shared_tmp / "dino_control", module=module, optim__epochs=1).fit()

    trainable = torch.cat(
        [p.detach().reshape(-1) for p in module.parameters() if p.requires_grad]
    )
    gathered = fabric.all_gather(trainable)
    spread = float((gathered[0] - gathered[1]).abs().max())
    assert spread > 1e-6, (
        f"the control did NOT diverge (spread={spread:.3e}). Something other than "
        "`ctx.module` is reducing these gradients, and the test above is uninformative."
    )


def test_ctx_module_carries_no_sync_so_a_duck_type_does_not_lie(two_rank_fabric, shared_tmp):
    """``hasattr(ctx.module, "no_sync")`` must be true, and calling it must suppress.

    ``DistributedDataParallel`` has ``no_sync``; ``_FabricModule`` does not forward it -- its
    ``__getattr__`` falls through to the bare module, not to the DDP -- so the attribute is
    absent unless the engine puts it back. A model gating suppression on that duck-type would
    then silently stop suppressing, and the cost is invisible: redundant reduces, not wrong
    numbers, so no assertion anywhere else would catch it. ``ctx.no_backward_sync`` is the
    contract; this keeps the attribute from contradicting it.
    """
    fabric = two_rank_fabric
    seen: dict[str, object] = {}

    class Recording(ToyModule):
        def training_step(self, batch, ctx):
            seen["module"] = ctx.module
            seen["has_no_sync"] = hasattr(ctx.module, "no_sync")
            with ctx.module.no_sync():
                seen["suppressed_inside"] = not _reduces(ctx.module)
            seen["restored_after"] = _reduces(ctx.module)
            return super().training_step(batch, ctx)

    _trainer(
        fabric, shared_tmp / "no_sync_run", module=Recording(), optim__epochs=1
    ).fit()

    assert seen["has_no_sync"], "ctx.module has no `no_sync`; the engine did not restore it"
    assert seen["suppressed_inside"], "`ctx.module.no_sync()` did not suppress the reduce"
    assert seen["restored_after"], "`no_sync` did not restore the previous value on exit"


def _reduces(handle) -> bool:
    """Whether DDP would reduce on the next backward through this handle."""
    ddp = getattr(handle, "_forward_module", handle)
    return bool(getattr(ddp, "require_backward_grad_sync", True))


def test_the_frozen_teacher_is_synced_at_setup_and_never_reduced(
    two_rank_fabric, shared_tmp
):
    """The teacher needs no special handling on this path, and that is a claim to check.

    Its parameters have ``requires_grad=False``, so DDP's reducer never includes them -- and
    they are still inside the wrapped module, so DDP's construction-time
    ``_sync_module_states`` (and the engine's own ``_broadcast_module_state``) broadcast them
    from rank 0. **Both halves matter.** A rank-local teacher would leave each rank distilling
    against a different target for the whole run, since an EMA only decays the difference;
    that is the divergence the old repo measured at ``train_dino.py:211-218`` and had to fix
    by hand once it had put the teacher outside every wrapper.

    So this asserts the teachers agree **after setup, before any training**, which is the
    property that comes from the sync rather than from the students happening to match.
    """
    fabric = two_rank_fabric
    # Deliberately divergent construction: seed by rank, so the teachers start different and
    # only a broadcast can bring them together. Without it this test would pass on identical
    # construction alone and would prove nothing about the sync.
    torch.manual_seed(1234 + fabric.global_rank)
    module = _dino(n_pairs=2)

    trainer = _trainer(
        fabric,
        shared_tmp / "teacher_run",
        module=module,
        loader=fake_loader(steps=8),
        optim__epochs=1,
    )
    trainer.setup()

    teacher = torch.cat([p.detach().reshape(-1) for p in module.model.teacher.parameters()])
    at_setup = fabric.all_gather(teacher)
    assert float((at_setup[0] - at_setup[1]).abs().max()) == 0.0, (
        "the ranks' teachers differ after setup. The frozen teacher is not in DDP's reducer, "
        "so nothing during training will ever bring it back into agreement."
    )
    assert not any(p.requires_grad for p in module.model.teacher.parameters())

    trainer.fit()

    after = fabric.all_gather(
        torch.cat([p.detach().reshape(-1) for p in module.model.teacher.parameters()])
    )
    spread = float((after[0] - after[1]).abs().max())
    assert spread == 0.0, (
        f"the two ranks' teachers differ by {spread:.3e} after training. The teacher is an "
        "EMA of a student the all-reduce should have kept identical, so this means the "
        "student diverged in a way the direct comparison missed."
    )


def test_teacher_and_centring_are_updated_together_and_agree(two_rank_fabric, shared_tmp):
    """``on_step_end`` ran both halves, and both are rank-consistent.

    The centring buffer is created lazily inside ``update_center`` and so cannot be covered by
    the engine's setup-time broadcast. It stays consistent because ``update_center``
    all-reduces a sum and a count instead of averaging per rank -- asserted here rather than
    trusted, since a per-rank centre would make the ranks optimise different objectives while
    every parameter comparison still passed.
    """
    fabric = two_rank_fabric
    module = _dino(n_pairs=2)
    _trainer(
        fabric,
        shared_tmp / "dino_centre",
        module=module,
        loader=fake_loader(steps=8),
        optim__epochs=1,
    ).fit()

    assert module.loss_fn.updates > 0, "update_center never ran"
    assert hasattr(module.loss_fn, "center"), "the centring buffer was never created"
    centre = module.loss_fn.center.detach().reshape(-1)
    gathered = fabric.all_gather(centre)
    spread = float((gathered[0] - gathered[1]).abs().max())
    assert spread == 0.0, (
        f"the two ranks' centring buffers differ by {spread:.3e}: update_center is averaging "
        "per rank instead of all-reducing a sum and a count (loss.py:251-264)."
    )

    # The buffer is in the state dict, which is what makes a resume keep centring.
    assert any(k.endswith("loss_fn.center") for k in module.state_dict()), (
        f"loss_fn.center is not in state_dict: {sorted(module.state_dict())}"
    )


def test_the_accumulation_gate_reduces_once_per_window(two_rank_fabric, shared_tmp, monkeypatch):
    """One reduce per optimizer step, not per microstep and not per view.

    There used to be two nested gates here -- the model's per-view one inside the engine's
    per-microstep one -- and this test existed to prove they composed rather than fought.
    Since 2026-09-10 there is only the engine's: the model sums its views into one loss.
    With ``accumulate=2`` and 2 views, exactly one backward in every two may reduce, and the
    view count must not appear at all. The failure is quiet either way, because the ranks
    stay identical and only the gradient magnitudes differ, by the accumulation factor.
    """
    fabric = two_rank_fabric
    seen = _spy_on_the_engines_gate(monkeypatch)
    module = _dino(n_pairs=2)
    _trainer(
        fabric,
        shared_tmp / "dino_accum",
        module=module,
        loader=fake_loader(steps=8),
        optim__epochs=1,
        optim__accumulate_grad_batches=2,
    ).fit()

    assert seen, "no backward was ever taken"
    fired = [i for i, on in enumerate(seen) if on]
    assert fired, "no backward was ever allowed to reduce"
    assert all(i % 2 == 1 for i in fired), (
        f"reduces fired at {fired} of {len(seen)} backwards; expected only the last "
        "microstep of each accumulate=2 window, and exactly one backward per microstep."
    )

    trainable = torch.cat(
        [p.detach().reshape(-1) for p in module.parameters() if p.requires_grad]
    )
    gathered = fabric.all_gather(trainable)
    spread = float((gathered[0] - gathered[1]).abs().max())
    assert spread == 0.0, f"ranks diverged under accumulation by {spread:.3e}"


def test_a_model_owned_schedule_reads_the_run_geometry_from_ctx_extra(
    two_rank_fabric, shared_tmp
):
    """The teacher momentum is a cosine over ``total_iters``, and ``total_iters`` is
    ``epochs * len(loader)`` -- not knowable when the model is constructed. It arrives through
    ``ctx.extra``, so this asserts the schedule was built from the run's real geometry and
    moved off its start value rather than silently staying constant."""
    fabric = two_rank_fabric
    module = _dino(n_pairs=1, momentum_start=0.9, momentum_end=1.0)
    trainer = _trainer(
        fabric,
        shared_tmp / "dino_mom",
        module=module,
        loader=fake_loader(steps=8),
        optim__epochs=2,
    )
    trainer.fit()

    assert module._momentum is not None, "the momentum schedule was never built"
    assert len(module._momentum) == trainer.epoch_len * 2, (
        f"schedule length {len(module._momentum)} does not match "
        f"epochs*epoch_len = {trainer.epoch_len * 2}: ctx.extra['total_iters'] did not arrive"
    )
    assert module._momentum[0] == pytest.approx(0.9)
    assert module.model.last_momentum > 0.9, (
        f"momentum never moved off its start value ({module.model.last_momentum})"
    )


def test_a_checkpoint_round_trips_under_ddp(two_rank_fabric, shared_tmp):
    """Save on two ranks, resume into a fresh module, and get the same tensors back.

    Two properties, both of which a future Fabric could break silently. The wrapper must not
    leak into ``state_dict()`` keys -- a checkpoint has to stay loadable by an unwrapped
    model, which is what makes resume, ``wcfm eval`` and every downstream reader work
    unchanged. And the **centring buffer** has to come back: it is created lazily inside
    ``update_center``, so it is the one piece of state the engine's setup-time broadcast
    cannot cover, and "a resumed run silently restarts centring" is the bug Checkpoint v2
    exists to fix.
    """
    fabric = two_rank_fabric
    module = _dino(n_pairs=2)
    run_dir = shared_tmp / "dino_ckpt"
    _trainer(
        fabric,
        run_dir,
        module=module,
        loader=fake_loader(steps=8),
        optim__epochs=1,
    ).fit()

    keys = list(module.state_dict())
    assert "model.student.weight" in keys, (
        f"state_dict keys were rewritten by the wrapper: {keys}"
    )
    assert not any("_forward_module" in k for k in keys), (
        "the wrapper leaked into the checkpoint keys: "
        f"{[k for k in keys if '_forward_module' in k]}"
    )
    assert any(k.endswith("loss_fn.center") for k in keys), "centring buffer not saved"

    saved_centre = module.loss_fn.center.detach().clone()
    saved = torch.cat(
        [p.detach().reshape(-1) for p in module.parameters() if p.requires_grad]
    )

    # A fresh module, resumed from what rank 0 wrote.
    resumed = _dino(n_pairs=2)
    result = _trainer(
        fabric,
        run_dir,
        module=resumed,
        loader=fake_loader(steps=8),
        optim__epochs=1,
        run__resume="auto",
    ).fit()

    assert result["epochs_run"] == 0, "a finished run resumed with `auto` has nothing to do"
    reloaded = torch.cat(
        [p.detach().reshape(-1) for p in resumed.parameters() if p.requires_grad]
    )
    assert torch.equal(saved, reloaded), "the student did not come back from the checkpoint"
    assert hasattr(resumed.loss_fn, "center"), (
        "the lazily-created centring buffer did not come back -- a resumed run would restart "
        "centring, silently, which is the bug Checkpoint v2 exists to fix"
    )
    assert torch.equal(saved_centre, resumed.loss_fn.center.detach())


# ------------------------------------------------------- the real module shape, on two ranks


def test_ssl_module_keeps_student_teacher_and_centre_identical_across_ranks(
    two_rank_fabric, shared_tmp
):
    """``SslModule`` over ``LinearBackbone``: the shape Stage 3 ships, on the one DDP path.

    Three things must agree across ranks after an epoch, and each is a different mechanism:
    the **student** (DDP's all-reduce, through ``ctx.module`` per view); the **teacher** (an
    EMA of a student the ranks agree on, itself broadcast at construction -- and computed
    through ``ctx.module(teacher=True)`` under ``no_grad``, which must not arm the reducer or
    the next backward raises); the **centring buffer** (``update_center``'s sum-and-count
    all-reduce). A per-rank teacher or centre would leave the ranks optimising different
    objectives while the student comparison still passed.
    """
    import random

    from wcfm.model.augment import Augment, BlockMasker, Cropper
    from wcfm.model.modules import EmaTeacher, SslModule
    from wcfm.model.terms import ChargeTerm, DinoTerm

    from .fake_backbone import LinearBackbone, batch_loader

    fabric = two_rank_fabric
    random.seed(0)
    torch.manual_seed(0)  # same construction on both ranks; the broadcast is the guarantee
    module = SslModule(
        backbone=LinearBackbone(in_dim=1, hidden=8, out_dim=8),
        terms={
            "dino": DinoTerm(score_injected=True,
                             proj_head={"hidden_dim": 16, "output_dim": 8, "n_layers": 2}),
            "charge": ChargeTerm(weight=0.1),
        },
        augment=Augment(
            cropper=Cropper(image_w=64, image_h=48, n_global=1, n_local=2, min_active_pixels=5,
                            blur_sigma_px=2.0),
            masker=BlockMasker(ratio=0.5, win_ch=2, win_tick=2),
        ),
        teacher=EmaTeacher(0.9, 1.0),
    )
    result = _trainer(
        fabric,
        shared_tmp / "ssl_run",
        module=module,
        loader=batch_loader(steps=8, counts=(60, 50), width=64, height=48, blob=True),
        optim__epochs=1,
    ).fit()
    assert result["steps"] > 0

    def spread(tensors) -> float:
        flat = torch.cat([t.detach().reshape(-1).float() for t in tensors])
        g = fabric.all_gather(flat)
        return float((g[0] - g[1]).abs().max())

    student = [p for p in module.parameters() if p.requires_grad]
    assert spread(student) == 0.0, "the student diverged: a forward bypassed ctx.module"
    assert spread(module.teacher_backbone.parameters()) == 0.0, "the teachers diverged"
    assert spread(module.terms["dino"].teacher_head.parameters()) == 0.0
    centre = module.terms["dino"].loss.center
    assert spread([centre]) == 0.0, "update_center is averaging per rank"
    assert bool(module.terms["dino"].loss.center_initialized)



def test_term_gradients_reach_the_collector_already_reduced(two_rank_fabric, shared_tmp):
    """Spike (e)'s conclusion, enforced end to end on two ranks.

    `autograd.grad` returns RANK-LOCAL vectors -- that is what the spike measured -- so the
    module all-reduces them before they reach `TermGrad`, which declares ``reduce = {}`` and
    performs no collective of its own. This asserts the vectors are identical on both ranks,
    which is the only way ``cos(mean(g_a), mean(g_b))`` is the conflict in the gradient the
    optimizer applies rather than one shard's opinion of it.

    **The control is in the data**: `toy_loader` gives the two ranks different batches, so an
    unreduced vector differs between them and this test fails. `test_ddp_keeps_every_ranks_
    parameters_identical` establishes that the fixture really does shard.

    The collector is configured through `metrics.collectors`, not attached to the trainer by
    hand: `_build_collectors` runs inside `setup()`, which `fit()` calls, so an injected
    collector is silently discarded.
    """
    from .toy import TwoTermToyModule, toy_loader

    module = TwoTermToyModule(opposed=True)
    trainer = _trainer(
        two_rank_fabric,
        shared_tmp / "termgrad",
        module=module,
        loader=toy_loader(steps=4),
        optim__epochs=1,
        metrics__step_cadence=1,
        metrics__collectors={
            "termgrad": {"_target_": "wcfm.metrics.collectors.TermGrad", "cadence": 1}
        },
    )
    trainer.fit()

    # The engine asked on every step, on every rank. A rank that decided otherwise would leave
    # the other inside the module's all_reduce for good.
    assert module.saw_request and all(module.saw_request)
    assert module.last_term_gradients is not None
    assert set(module.last_term_gradients) == {"a", "b"}

    for name, vec in module.last_term_gradients.items():
        gathered = two_rank_fabric.all_gather(vec)
        spread = (gathered[0] - gathered[1]).abs().max()
        assert torch.allclose(gathered[0], gathered[1], atol=1e-6), (
            f"term {name!r} was NOT reduced before the collector saw it -- ranks differ by "
            f"{spread}. A cosine over rank-local vectors is not the conflict the optimizer sees."
        )

    # And the collector actually wrote its columns, so the wiring is exercised end to end.
    if two_rank_fabric.is_global_zero:
        stream = shared_tmp / "termgrad" / "metrics" / "step.jsonl"
        rows = [json.loads(line) for line in stream.read_text().splitlines()]
        assert any("termgrad/min_cos" in r for r in rows), "TermGrad wrote no columns"
        # The toy's terms are built to oppose each other over the trunk.
        worst = min(r["termgrad/min_cos"] for r in rows if "termgrad/min_cos" in r)
        assert worst < 0.0, f"the opposed toy should show a negative cosine, got {worst}"


def test_the_engine_only_asks_for_term_gradients_when_a_collector_needs_them(
    two_rank_fabric, shared_tmp
):
    """The measurement costs ~1.5x a step, so it must not run when nothing reads it -- and the
    decision must be identical on every rank or the module's all-reduce hangs the job."""
    from .toy import TwoTermToyModule, toy_loader

    module = TwoTermToyModule()
    trainer = _trainer(
        two_rank_fabric,
        shared_tmp / "termgrad_off",
        module=module,
        loader=toy_loader(steps=3),
        optim__epochs=1,
        metrics__step_cadence=1,
    )
    trainer.fit()
    # No collector declares `needs = {"term_grads"}`, so the module was never asked.
    assert module.saw_request and not any(module.saw_request)
    assert module.last_term_gradients is None
