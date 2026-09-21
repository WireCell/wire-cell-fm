"""This is the main training loop, on Lightning Fabric.
It owns everything about running a job and knows nothing about what is being trained.

Procedure:

- `setup()` launches Fabric, seeds as `run.seed + rank`, builds the loader, wraps the module
  for DDP and builds the optimizer from its declared `param_groups`, builds the schedules
  from `len(loader)`, instantiates the collectors, resumes if `run.resume` resolves to a
  file, writes the run directory and opens the metrics writer.
- `fit()` walks the epochs: reshuffle the reader, run the epoch, checkpoint on the
  configured cadence, write the epoch record. It stops early on preemption, after writing
  `latest.pt`.
- `_run_epoch()` walks the batches: zero the gradients on an accumulation boundary and apply
  the schedules, move the batch, build the `StepContext`, call `training_step` and backward
  the loss it returns. On the micro-step that closes the window: clip, check the gradients
  are finite, step the optimizer, call `on_step_end`, collect metrics, poll for preemption.
- `engine/collection.py` runs the collectors: `on_step` after each optimizer step and
  `on_epoch` at the end, each building a `StepRecord`, reducing the declared keys and
  writing one flat row.

Fabric gives the same semantics on one GPU and on many, so no call site here branches on
whether the run is distributed.

There is one DDP path, and two handles on the module. The wrapper from `fabric.setup` arms
DDP's reducer and applies `run.precision`, so every forward goes through it as `ctx.module`,
and the accumulation gate applies to it. The unwrapped module keeps unprefixed parameter
names, so `training_step`, `state_dict` and `StepRecord.named_parameters` use that one.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from lightning_fabric.plugins.precision.amp import MixedPrecision
from omegaconf import DictConfig, OmegaConf

from wcfm.config.io import derive, write_run_dir
from wcfm.data.build import build_loader
from wcfm.engine.checkpoint import (
    Checkpoint,
    load_checkpoint,
    resolve_resume,
    rng_state,
    save_checkpoint,
    set_rng_state,
    should_save,
)
from wcfm.engine.collection import Collection, build_collectors
from wcfm.engine.optim import apply_schedules, build_optimizer, build_schedules
from wcfm.engine.preempt import PreemptionGuard
from wcfm.engine.protocol import StepContext, StepOutput
from wcfm.metrics.writer import MetricsWriter

log = logging.getLogger(__name__)

__all__ = ["Trainer", "build_fabric"]


class AutocastOnlyPrecision(MixedPrecision):
    """`16-mixed` and `bf16-mixed` as an autocast around the forward and nothing else.

    Fabric's `MixedPrecision.convert_input` casts every floating tensor the wrapper's forward
    receives to the half type before the forward runs, reaching into dataclasses such as
    `Voxels` and a model's own arguments. A coordinate of order one in bfloat16 has a spacing of
    two to five pixels, which is what a point cloud sees when `ctx.module(points, lengths)`
    is called under it; a charge feature loses its low bits the same way. Autocast alone casts
    the operands of matmuls and convolutions and leaves everything else in the dtype it
    arrived in, which is the behaviour `run.precision` promises.
    `tests/test_engine.py::test_mixed_precision_leaves_the_inputs_alone` pins it.
    """

    def convert_input(self, data: Any) -> Any:
        return data


def build_fabric(cfg: DictConfig) -> Any:
    """A `Fabric` carrying the launch config's DDP flags.

    `fabric.launch()` is called unconditionally by `Trainer.fit`, and is a no-op under
    torchrun: `creates_processes_externally` is true there, so Fabric skips
    `_call_children_scripts()` and just calls the function.

    The import is `lightning_fabric`, the standalone distribution `pyproject.toml` depends
    on, which installs only that top-level package. `lightning.fabric` belongs to the
    umbrella `lightning` package, which pulls in `lightning.pytorch` and torchmetrics, both
    absent from the cluster stack, and raises `ModuleNotFoundError` there.
    """
    from lightning_fabric import Fabric
    from lightning_fabric.strategies import DDPStrategy

    devices = int(cfg.launch.devices)

    # `launch.devices` is the ONE place the rank count is stated (`conf/launch/*.yaml`), and
    # this is where a disagreement becomes visible. Under torchrun the launcher has already
    # forked WORLD_SIZE ranks; if the config says a different number, Fabric builds the wrong
    # world and nothing else complains. Checked before a device is touched, so the job
    # dies in seconds.
    env_world = os.environ.get("WORLD_SIZE")
    if env_world is not None and int(env_world) != devices:
        raise RuntimeError(
            f"launch.devices={devices} but the launcher started WORLD_SIZE={env_world}. "
            "These must agree: `launch.devices` is what Fabric builds and WORLD_SIZE is what "
            "actually exists. Select a launch preset that matches the allocation (e.g. "
            f"`launch.devices={env_world}`), or request a different number of GPUs."
        )

    strategy_name = str(cfg.launch.strategy)
    strategy: Any = strategy_name
    if strategy_name == "ddp" and (devices > 1 or int(cfg.launch.num_nodes) > 1):
        # Built here so the two flags below reach it; `strategy="ddp"` takes Fabric's
        # defaults and drops them.
        strategy = DDPStrategy(
            find_unused_parameters=bool(cfg.launch.find_unused_parameters),
            static_graph=bool(cfg.launch.static_graph),
        )
    elif devices == 1:
        strategy = "auto"

    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    precision = str(cfg.run.precision)
    kwargs: dict[str, Any] = {"precision": precision}
    if precision in ("16-mixed", "bf16-mixed"):
        # Fabric refuses `precision=` together with a precision plugin.
        kwargs = {"plugins": [AutocastOnlyPrecision(precision, accelerator)]}
    return Fabric(
        accelerator=accelerator,
        devices=devices,
        num_nodes=int(cfg.launch.num_nodes),
        strategy=strategy,
        **kwargs,
    )


class Trainer:
    """Owns the loop and nothing about the model.

    `fabric` is injectable so the CPU suite can hand in a one-device `Fabric` and exercise
    accumulation, the guards, checkpointing and resume without a GPU. The backbone is what
    needs CUDA; the loop does not.
    """

    def __init__(
        self,
        cfg: DictConfig,
        module: Any,
        *,
        fabric: Any = None,
        loader: Any = None,
        run_dir: Path | str | None = None,
        argv: list[str] | None = None,
    ):
        self.cfg = cfg
        self.module = module  # UNWRAPPED
        self.fabric = fabric if fabric is not None else build_fabric(cfg)
        self._given_loader = loader
        self.argv = argv or []
        self.run_dir = Path(run_dir) if run_dir else Path(cfg.run.output_root) / cfg.run.name

        # The DDP-wrapped handle, and the only one `no_backward_sync` is applied to.
        # `None` until `setup()` runs, which is also how `fit()` knows to call it.
        self.wrapped: Any = None
        self._ctx_extra: dict[str, Any] = {}
        self.optimizer: torch.optim.Optimizer | None = None
        self.schedules: dict[str, Any] = {}
        # Built in `setup()`, once the writer it needs exists.
        self.collection: Collection | None = None
        self.step = 0
        self.start_epoch = 1
        self.skipped_nonfinite = 0
        self.skipped_oom = 0

    # ------------------------------------------------------------------ setup

    @property
    def world_size(self) -> int:
        return int(getattr(self.fabric, "world_size", 1))

    @property
    def rank(self) -> int:
        return int(getattr(self.fabric, "global_rank", 0))

    def setup(self) -> None:
        self.fabric.launch()  # a no-op under torchrun, which launches the ranks itself
        self.fabric.seed_everything(int(self.cfg.run.seed) + self.rank)

        loader = self._given_loader
        if loader is None:
            loader = build_loader(
                self.cfg.data,
                rank=self.rank,
                world_size=self.world_size,
                num_workers=int(self.cfg.run.num_workers),
                shuffle=True,
                seed=int(self.cfg.run.seed),
            )
        # move_to_device=False: `Batch.to` owns the move.
        self.loader = self.fabric.setup_dataloaders(loader, move_to_device=False)

        self._setup_module_and_optimizer()

        self.epoch_len = len(self.loader)
        self.schedules = build_schedules(self.cfg.optim, self.epoch_len)

        derived = derive(self.cfg, self.epoch_len, self.world_size)
        # Run geometry for `StepContext.extra`: a model that owns a schedule needs the
        # totals, and cannot work them out itself because `epoch_len` is `len(loader)`.
        self._ctx_extra = {
            "epoch_len": derived["epoch_len"],
            "total_iters": derived["total_iters"],
            "world_size": derived["world_size"],
        }
        resume_from = resolve_resume(str(self.cfg.run.resume), self.run_dir / "checkpoints")
        if resume_from is not None:
            self._load(resume_from)

        if self.fabric.is_global_zero:
            # Imported here rather than at module scope: `collect()` imports torch, warpconvnet
            # and flash_attn to read their versions, and paying that at `wcfm.engine` import
            # time would make every CPU-only test drag the GPU stack in.
            from wcfm.cli.env_check import collect

            write_run_dir(
                self.cfg,
                self.run_dir,
                argv=self.argv,
                world_size=self.world_size,
                derived=derived,
                env=collect(),
                module=self._module_provenance(),
            )
        self.writer = MetricsWriter(
            self.run_dir / "metrics",
            enabled=self.fabric.is_global_zero,
            max_record_bytes=int(self.cfg.metrics.max_record_bytes),
            resume_step=self.step if self.step else None,
        )
        self.collection = Collection(
            build_collectors(self.cfg.metrics),
            fabric=self.fabric,
            writer=self.writer,
            module=self.module,
            run_dir=self.run_dir,
            step_cadence=int(self.cfg.metrics.step_cadence),
        )

    def _setup_module_and_optimizer(self) -> None:
        """Wrap the whole module for DDP, then build the optimizer from its declared groups.

        One path, covering every model shape here. A second would mean two sets of
        semantics for `no_sync`, `state_dict` and `ctx.module`, whose multi-GPU halves no
        single-device test can see.

        What the whole-module wrap gives a model:

        - Any number of views, through `ctx.module`. DDP arms its reducer once per
          `forward()` and the engine takes one backward per micro-step, so a model reaches
          every parameter that backward will touch from ONE call to the handle: it loops its
          views inside its own `forward` and returns a single summed loss. Asserted in the
          distributed suite (`views=3`, and `SslModule` itself), on gloo and on nccl.
        - Named submodules behind the module's own `forward`, which is what DDP reduces
          around.
        - A frozen EMA teacher inside the wrapper: its parameters have
          `requires_grad=False` so the reducer skips them, and DDP's construction-time
          `_sync_module_states` broadcasts them from rank 0, keeping every rank's teacher
          identical for free.

        The cost: one reducer covers every trainable parameter, so a view whose forward
        skips a head relies on `find_unused_parameters=True` to mark it ready. That is the
        configured default. The engine suppresses the reduce on every micro-step but the
        last, so each optimizer step reduces once.
        """
        self.optimizer = build_optimizer(self.module, self.cfg.optim)
        self.wrapped, self.optimizer = self.fabric.setup(self.module, self.optimizer)
        # Give `no_sync()` back. `DistributedDataParallel` has it, `_FabricModule` does not
        # forward it: its `__getattr__` falls through to the bare `nn.Module`, so
        # `hasattr(ctx.module, "no_sync")` is False. The engine owns suppression and nothing
        # in `wcfm.model` calls this, but a handle that accepts `no_sync()` and silently
        # does nothing would cost redundant reduces with no sign of it.
        self.wrapped.no_sync = lambda: self.fabric.no_backward_sync(self.wrapped, enabled=True)
        self._broadcast_module_state()

    def _module_provenance(self) -> dict:
        """The module's optional `provenance()`, or nothing. Never fatal.

        Found by `getattr` like the other optional hooks, so the engine names no model type.
        Wrapped because provenance is not worth failing a run over: a worker with no `git`
        on its PATH would otherwise kill the job at setup, after the
        queue wait, for a field nobody reads until afterwards.
        """
        hook = getattr(self.module, "provenance", None)
        if hook is None:
            return {}
        try:
            return dict(hook())
        except Exception as exc:  # pragma: no cover - defensive by intent
            log.warning("module provenance failed: %s: %s", type(exc).__name__, exc)
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _broadcast_module_state(self) -> None:
        """Make every rank agree on every parameter and buffer, from rank 0.

        `DistributedDataParallel.__init__` already broadcasts the wrapped module's
        parameters, and its buffers when `broadcast_buffers` is on. This keeps the guarantee
        anyway, for two reasons worth the one collective:

        - The buffer half of it is a strategy kwarg. `broadcast_buffers` defaults
          to true and nothing here sets it, so a `launch` key that turned it off to save
          bandwidth would silently take a frozen teacher's buffers with it. A teacher starts
          as a copy of the rank-local student and is then EMA-updated, so per-rank
          construction leaves each rank distilling against a different teacher for the whole
          run: an EMA only decays the difference, it never removes it.
        - It holds whatever the caller seeded, and it holds even when construction is
          nondeterministic, which is the stronger guarantee and the cheaper contract.

        It does not cover state a module creates lazily, because that state does not exist
        when this runs; the checkpoint schema is what covers it, by saving the module's own
        `state_dict`.

        One collective per dtype over a flat buffer, since a per-tensor
        loop is O(tensors) latencies and a real backbone has thousands. Tensors are visited
        in sorted-name order so the sequence is identical on every rank -- the same
        discipline as `_reduce_declared`.
        """
        if self.world_size == 1 or not torch.distributed.is_initialized():
            return

        named: list[tuple[str, torch.Tensor]] = [
            *self.module.named_parameters(),
            *self.module.named_buffers(),
        ]
        by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
        for name, tensor in named:
            if tensor is not None and tensor.numel():
                by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

        for dtype in sorted(by_dtype, key=str):
            tensors = [t for _, t in sorted(by_dtype[dtype], key=lambda kv: kv[0])]
            data = [t.data for t in tensors]
            flat = torch._utils._flatten_dense_tensors(data)
            torch.distributed.broadcast(flat, src=0)
            synced = torch._utils._unflatten_dense_tensors(flat, data)
            for tensor, value in zip(tensors, synced, strict=True):
                tensor.data.copy_(value)

    # -------------------------------------------------------------- the loop

    def fit(self) -> dict:
        if self.wrapped is None:
            self.setup()
        epochs = int(self.cfg.optim.epochs)
        accumulate = max(1, int(self.cfg.optim.accumulate_grad_batches))
        stopped_early = False

        if self.start_epoch > epochs:
            # Nothing to do: a finished run resumed with `resume: auto`, or one resumed
            # with fewer `optim.epochs` than the checkpoint reached. A re-submitted sweep
            # hits the first every time, so this returns and says which epoch it found.
            self.fabric.print(f"nothing to do: resumed at epoch {self.start_epoch} of {epochs}")
            self.writer.close()
            return {
                "epochs_run": 0,
                "steps": self.step,
                "preempted": False,
                "skipped_nonfinite": self.skipped_nonfinite,
                "skipped_oom": self.skipped_oom,
            }

        with PreemptionGuard() as guard, self.writer:
            for epoch in range(self.start_epoch, epochs + 1):
                self._set_epoch(epoch)
                self.module.train()
                summary = self._run_epoch(epoch, accumulate, guard)

                if self.fabric.is_global_zero and should_save(
                    epoch,
                    epochs,
                    int(self.cfg.run.save_every),
                    list(self.cfg.run.save_at),
                ):
                    self._save(self.run_dir / "checkpoints" / f"checkpoint_epoch{epoch}.pt", epoch)

                self.writer.write("epoch", {"epoch": epoch, "step": self.step, **summary})
                self.fabric.print(_timing_line(epoch, summary))

                if summary.get("preempted"):
                    # Every rank agrees on this: the flag was all-reduced. `latest.pt` is
                    # written by rank 0 only, which is also the only rank that ever reads it.
                    if self.fabric.is_global_zero:
                        self._save(self.run_dir / "checkpoints" / "latest.pt", epoch)
                    self.fabric.print(f"preempted at epoch {epoch}, step {self.step}")
                    stopped_early = True
                    break

        return {
            "epochs_run": epoch - self.start_epoch + 1,
            "steps": self.step,
            "preempted": stopped_early,
            "skipped_nonfinite": self.skipped_nonfinite,
            "skipped_oom": self.skipped_oom,
        }

    def _run_epoch(self, epoch: int, accumulate: int, guard: PreemptionGuard) -> dict:
        assert self.optimizer is not None
        device = getattr(self.fabric, "device", torch.device("cpu"))
        epoch_t0 = time.perf_counter()
        data_wait = 0.0
        loss_sum = 0.0
        n_steps = 0
        preempted = False
        self.collection.begin_epoch()
        t_fetch = time.perf_counter()

        for batch_idx, batch in enumerate(self.loader):
            data_wait += time.perf_counter() - t_fetch
            step_t0 = time.perf_counter()
            is_last_micro = ((batch_idx + 1) % accumulate) == 0

            if batch_idx % accumulate == 0:
                self.optimizer.zero_grad(set_to_none=True)
                # The schedule is indexed by optimizer step, not by batch. The two are the
                # same sequence only while `accumulate_grad_batches` is 1.
                applied = apply_schedules(self.optimizer, self.schedules, self.step)

            batch = batch.to(device) if hasattr(batch, "to") else batch
            # Decided before the step: the module takes these gradients while it still
            # holds the graph.
            want_term_grads = self.collection.wants_term_grads(self.step)
            ctx = StepContext(
                step=self.step,
                epoch=epoch,
                device=device,
                # The DDP-wrapped handle. `training_step` runs on the unwrapped module,
                # but every forward inside it goes through this one, which arms the reducer
                # and applies precision.
                module=self.wrapped,
                all_reduce=self.fabric.all_reduce,
                is_last_microstep=is_last_micro,
                extra={**self._ctx_extra, "collect_term_gradients": want_term_grads},
            )

            out = self._training_step(batch, ctx)
            if out is None:  # OOM at world_size == 1, counted and skipped
                t_fetch = time.perf_counter()
                continue

            if is_last_micro:
                stepped = self._optimizer_step(batch)
                if stepped:
                    self.module.on_step_end(ctx)
                loss_sum += float(out.scalars.get("loss", 0.0))
                n_steps += 1
                step_time = time.perf_counter() - step_t0

                timing = {
                    "step": step_time,
                    "data_wait": data_wait,
                    "epoch_elapsed": time.perf_counter() - epoch_t0,
                }
                self.collection.on_step(
                    epoch,
                    self.step,
                    out,
                    applied,
                    timing,
                    grad_norm=getattr(self, "_last_grad_norm", None),
                )
                self.step += 1

                if guard.should_stop(self.fabric.all_reduce if self.world_size > 1 else None):
                    preempted = True
                    break

            t_fetch = time.perf_counter()

        elapsed = time.perf_counter() - epoch_t0
        # Epoch-cadence collectors fire here, and their columns land in the epoch record.
        return {
            **self.collection.on_epoch(epoch),
            "loss": loss_sum / n_steps if n_steps else None,
            "steps": n_steps,
            "elapsed_s": elapsed,
            "data_wait_s": data_wait,
            "data_wait_frac": data_wait / elapsed if elapsed > 0 else None,
            "skipped_nonfinite": self.skipped_nonfinite,
            "skipped_oom": self.skipped_oom,
            "preempted": preempted,
        }

    def _training_step(self, batch: Any, ctx: StepContext) -> StepOutput | None:
        """The module's step and its backward, with the OOM guard around both.

        The model computes a loss and returns it on `StepOutput`; this takes it through
        `fabric.backward` under the accumulation gate. Three things therefore live in one
        place: the suppression rule for gradient accumulation, the precision plugin's
        scaler, and the OOM guard -- an OOM lands in the backward far more often than in the
        forward, so the guard has to span it.

        A module may return `loss=None`, meaning it has nothing to differentiate this step
        (an eval-shaped or metrics-only step); the engine then takes no backward.

        Recovery is only attempted on one device. Under DDP an OOM on one rank is not
        recoverable: the ranks that completed their backward are already in the gradient
        all-reduce, so a rank that skips its backward hangs the group rather than skipping a
        step, and the counter would report a skip that never happened while the job sat in
        the queue until its wall clock ran out. So this re-raises, with the batch's event
        keys, and the job dies loudly and diagnosably instead.
        """
        try:
            out = self.module.training_step(batch, ctx)
            if out is not None and out.loss is not None:
                # Suppressed on every micro-step but the last, so DDP reduces once per
                # optimizer step rather than once per batch.
                with self._no_backward_sync(not ctx.is_last_microstep):
                    self.fabric.backward(out.loss)
            return out
        except torch.cuda.OutOfMemoryError:
            self.skipped_oom += 1
            keys = _event_keys(batch)
            self.writer.write(
                "step", {"step": self.step, "epoch": ctx.epoch, "oom": True, "event_keys": keys}
            )
            if self.world_size > 1:
                raise RuntimeError(
                    f"CUDA OOM on rank {self.rank} at step {self.step}, batch event keys "
                    f"{keys}. Not recoverable under DDP: the other ranks are already in the "
                    "gradient all-reduce, so skipping this rank's step would hang the group."
                ) from None
            assert self.optimizer is not None
            self.optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            return None

    def _optimizer_step(self, batch: Any) -> bool:
        """Clip, guard against non-finite gradients, step. Returns whether a step happened."""
        assert self.optimizer is not None
        clip = float(self.cfg.optim.clip_grad_norm)
        grad_norm = None
        if clip > 0.0:
            # The unwrapped module on purpose: `clip_gradients` calls `_unwrap_objects` on
            # whatever it is given, so this is the same call either way and stays correct if
            # a model ever holds a trainable tensor the wrapper does not enumerate.
            grad_norm = self.fabric.clip_gradients(self.module, self.optimizer, max_norm=clip)

        if not self._grads_finite():
            self.skipped_nonfinite += 1
            self.writer.write(
                "step",
                {
                    "step": self.step,
                    "nonfinite_grad": True,
                    "event_keys": _event_keys(batch),
                },
            )
            self.optimizer.zero_grad(set_to_none=True)
            return False

        self.optimizer.step()
        if grad_norm is not None:
            self._last_grad_norm = float(grad_norm)
        return True

    def _grads_finite(self) -> bool:
        """Every rank reaches the same verdict: the check is a collective when it has to be.

        A rank-local answer would let one rank step while another skipped, so the two would
        hold different parameters from then on -- and nothing downstream would say so, because
        only rank 0 writes metrics.
        """
        local = 1.0
        for param in self.module.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                local = 0.0
                break
        if self.world_size == 1:
            return local > 0.0
        local_t = torch.tensor([local], device=self.fabric.device)
        flag = self.fabric.all_reduce(local_t, reduce_op="min")
        return bool(float(flag.item()) > 0.0)

    def _no_backward_sync(self, enabled: bool = True):
        """The accumulation gate: one bool, closed over the wrapped handle.

        Fabric's signature is `no_backward_sync(module, enabled=True)`, and the handle is
        always `self.wrapped`, the one thing there is to suppress. It wraps the single
        `fabric.backward` of a micro-step (`_training_step`), so DDP reduces on the last
        micro-step of an accumulation window.
        """
        return self.fabric.no_backward_sync(self.wrapped, enabled=enabled)

    # ------------------------------------------------------------ checkpoints

    def _set_epoch(self, epoch: int) -> None:
        """The reader's equivalent of `DistributedSampler.set_epoch`, and the sampler's own.

        Without the sampler call each rank sees a fixed 1/N of the pack for the whole run --
        which no metric would obviously reveal.
        """
        for holder in (self.loader, getattr(self.loader, "dataset", None)):
            if holder is not None and hasattr(holder, "set_epoch"):
                holder.set_epoch(epoch)
        sampler = getattr(self.loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _save(self, path: Path, epoch: int) -> Path:
        assert self.optimizer is not None
        return save_checkpoint(
            path,
            Checkpoint(
                epoch=epoch,
                step=self.step,
                cfg=OmegaConf.to_container(self.cfg, resolve=True),  # type: ignore[arg-type]
                model=self.module.state_dict(),  # opaque; carries the centring buffer
                optimizer=self.optimizer.state_dict(),
                rng={f"rank{self.rank}": rng_state(self.rank)},
                meta={
                    "epoch_len": self.epoch_len,
                    "world_size": self.world_size,
                    "skipped_nonfinite": self.skipped_nonfinite,
                    "skipped_oom": self.skipped_oom,
                },
            ),
        )

    def _load(self, path: Path) -> None:
        ckpt = load_checkpoint(path, map_location="cpu")
        # `self.module` is the unwrapped module and nothing replaced its attributes, so
        # this is a plain `nn.Module.load_state_dict` against the keys `state_dict()` wrote.
        self.module.load_state_dict(ckpt.model)
        if ckpt.optimizer and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt.optimizer)
        self.step = int(ckpt.step)
        self.start_epoch = int(ckpt.epoch) + 1
        self.skipped_nonfinite = int(ckpt.meta.get("skipped_nonfinite", 0))
        self.skipped_oom = int(ckpt.meta.get("skipped_oom", 0))

        # Only rank 0's RNG state is in the file, so every other rank reseeds. The resume
        # epoch is mixed in because `seed + rank` alone would hand a rank the stream it
        # already had at the start of the run. No two ranks share a stream.
        restored = set_rng_state(ckpt.rng.get(f"rank{self.rank}"))
        if not restored:
            self.fabric.seed_everything(
                int(self.cfg.run.seed) + self.rank + 1000 * self.start_epoch
            )
        self.fabric.print(
            f"resumed from {path} at epoch {ckpt.epoch}, step {ckpt.step}"
            + ("" if restored else f" (rank {self.rank} reseeded; no RNG state in the file)")
        )


def _event_keys(batch: Any) -> list | None:
    """The offending batch's event keys, so a guard's counter is reproducible rather than
    merely alarming. `Batch` carries meta from day one, which is what makes this free."""
    meta = getattr(batch, "meta", None)
    if not isinstance(meta, dict):
        return None
    for key in ("event_key", "event_id", "key", "entry"):
        if key in meta:
            value = meta[key]
            return value.tolist() if hasattr(value, "tolist") else list(value)
    return None


def _timing_line(epoch: int, summary: dict) -> str:
    """Formats `[timing] epoch=N ...` from the epoch record."""
    parts = [f"[timing] epoch={epoch}"]
    for key in ("loss", "steps", "elapsed_s", "data_wait_frac"):
        value = summary.get(key)
        if value is None:
            continue
        parts.append(f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}")
    return " ".join(parts)
