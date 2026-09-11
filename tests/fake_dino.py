"""A DINO-shaped ``TrainingModule`` the CPU suite can run the whole loop over.

This is the shape Stage 3's native DINO has -- an EMA teacher, a projection head, a backward
per view, a lazily-created centring buffer and a momentum schedule that needs the run's total
length -- with ``nn.Linear`` where the sparse backbone goes, because no sparse convolution
runs on a CPU. It is the closest thing in the repo to a Stage 3 template, so it is written the
way a real model should be written.

**One DDP path, and this is what a multi-submodule model looks like on it.** The reduced unit
is ``forward``: ``DinoShapedModule.forward`` runs the student path (backbone then head), the
step calls it once per view through ``ctx.module``, and each call arms DDP's reducer for the
backward that follows. Anything that must *not* be reduced is computed on ``self`` --
``encode_teacher`` runs under ``no_grad`` on frozen parameters. An earlier design had a second
DDP path for a model that wrapped its own named submodules; it existed because the old repo's
``DINODuneModel`` computes through fixed ``self.student(...)`` call sites, and it went when the
shim that imported that model went.

Two details are kept from the old repo because they are what a real model still has to survive:

* ``FakeLoss.center`` is created **lazily**, exactly as ``loss.py:73`` does, and
  ``update_center`` all-reduces a sum and a count rather than averaging per rank
  (``loss.py:251-264``) -- so the engine's setup-time broadcast cannot cover it and the
  checkpoint schema has to.
* The teacher is a rank-local EMA of a reduced student. That is only safe because the ranks
  start from identical teachers, which is what ``_broadcast_module_state`` and DDP's own
  construction-time sync both guarantee.

``tests/`` rather than ``wcfm/`` for the same reason as ``toy.py``: it knows what a training
step is, which no framework module may.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class FakeStats:
    """``ForwardStats``' fields, as the shim reads them (``dino/model.py:22-49``).

    ``None`` for a quantity this configuration does not produce, which is what
    ``_scalars_from`` turns into an *absent* key rather than a null one."""

    loss: float
    n_pairs: int
    t_ent: float | None = None
    s_ent: float | None = None
    kl: float | None = None
    cov: float | None = None
    var: float | None = None
    loss_masked: float | None = None
    loss_unmasked: float | None = None
    loss_charge: float | None = None
    loss_occ: float | None = None
    s_backbone: Any = None
    t_backbone: Any = None
    s_out: Any = None
    t_out: Any = None


class FakeVoxels:
    """Only what the shim touches: ``feature_tensor``, an ``[N, D]`` matrix."""

    def __init__(self, feature_tensor: Tensor):
        self.feature_tensor = feature_tensor


class FakeLoss(nn.Module):
    """``PixelDINOLoss``' surface: a lazily-created centring buffer and a collective update.

    The laziness is the point -- ``loss.py:73`` creates ``center`` on first use, which is why
    the engine's setup-time broadcast cannot cover it and why it has to be in ``state_dict``.
    ``update_center`` all-reduces a sum and a count rather than averaging per rank
    (``loss.py:251-264``), and does so *before* the empty-batch check, so every rank returns
    together or not at all.
    """

    def __init__(self, dim: int = 4, center_momentum: float = 0.9):
        super().__init__()
        self.center_momentum = float(center_momentum)
        self.dim = dim
        self.updates = 0

    def forward(self, *_args: Any, **_kwargs: Any) -> Tensor:  # pragma: no cover
        raise NotImplementedError("the loss is computed in DinoShapedModule.training_step")

    def update_center(self, teacher_out: Any) -> None:
        import torch.distributed as dist

        if teacher_out is None:
            return
        flat = teacher_out.feature_tensor
        distributed = dist.is_available() and dist.is_initialized()

        feat_sum = flat.sum(dim=0)
        count = torch.tensor([flat.shape[0]], device=feat_sum.device, dtype=feat_sum.dtype)
        if distributed:
            dist.all_reduce(feat_sum)
            dist.all_reduce(count)
        if float(count.item()) <= 0:
            return

        batch_center = feat_sum / count
        if not hasattr(self, "center"):
            # Lazily, exactly as loss.py:73 does.
            self.register_buffer("center", torch.zeros_like(batch_center))
        self.center.mul_(self.center_momentum).add_(
            batch_center.detach(), alpha=1.0 - self.center_momentum
        )
        self.updates += 1


def sync_enabled(module: Any) -> bool:
    """Whether DDP would actually reduce on the next backward.

    ``require_backward_grad_sync`` is the flag ``no_sync()`` toggles and the reducer consults,
    so it is the only observation that sees *both* gates -- the model's per-view one and the
    engine's accumulation one, which nest. Reached through ``_forward_module`` because
    ``_FabricModule.__getattr__`` falls through to the bare module, not to the DDP.
    """
    if module is None:
        return True
    ddp = getattr(module, "_forward_module", module)
    return bool(getattr(ddp, "require_backward_grad_sync", True))


class FakeDinoModel(nn.Module):
    """Student, projection head, EMA teacher: the parameters, and nothing about a step.

    ``forward`` is the student path and therefore the unit of DDP reduction; the teacher is
    reached only through ``encode_teacher``, under ``no_grad``. Owning the parameters here and
    the step in ``DinoShapedModule`` is the split a real model wants too -- the backbone does
    not need to know what a view is.
    """

    def __init__(self, dim: int = 4, use_proj_head: bool = True, n_pairs: int = 3):
        super().__init__()
        self.student = nn.Linear(dim, dim, bias=False)
        self.teacher = nn.Linear(dim, dim, bias=False)
        self.teacher.load_state_dict(self.student.state_dict())
        if use_proj_head:
            self.student_head = nn.Linear(dim, dim, bias=False)
            self.teacher_head = nn.Linear(dim, dim, bias=False)
            self.teacher_head.load_state_dict(self.student_head.state_dict())
        else:
            self.student_head = None
            self.teacher_head = None
        for module in (self.teacher, self.teacher_head):
            if module is not None:
                for p in module.parameters():
                    p.requires_grad_(False)
        self.n_pairs = n_pairs

    # ---------------------------------------------------------------- the surface

    def update_teacher(self, momentum: float) -> None:
        self.last_momentum = float(momentum)
        with torch.no_grad():
            pairs = [(self.student, self.teacher)]
            if self.student_head is not None:
                pairs.append((self.student_head, self.teacher_head))
            for s, t in pairs:
                for sp, tp in zip(s.parameters(), t.parameters(), strict=True):
                    tp.mul_(momentum).add_(sp.detach(), alpha=1.0 - momentum)

    def encode_student(self, rows: Tensor) -> Tensor:
        out = self.student(rows)
        if self.student_head is not None:
            out = self.student_head(out)
        return out

    def encode_teacher(self, rows: Tensor) -> Tensor:
        """The target, on frozen parameters under ``no_grad``.

        Computed on ``self``, never through ``ctx.module``: there is nothing to reduce, and
        `DDP.forward` would not arm the reducer under ``no_grad`` anyway. The parameters are
        still inside the wrapped module, which is what gets them broadcast from rank 0 at
        construction -- the divergence the old repo had to fix by hand.
        """
        with torch.no_grad():
            out = self.teacher(rows)
            if self.teacher_head is not None:
                out = self.teacher_head(out)
            return out.detach()

    def forward(self, rows: Tensor) -> Tensor:
        """The student path: **the unit of DDP reduction.**

        A model with named submodules puts them behind this rather than calling them from the
        step, so one ``ctx.module(...)`` call is one arming and one backward. Dispatching on
        an argument -- a ``tap=`` for the backbone alone -- is how a model with more than one
        reduced shape stays on one path.
        """
        return self.encode_student(rows)


class FakeBatch:
    """``Batch``'s surface: ``.voxels``, ``.meta``, ``.batch_size``, ``.to``.

    ``training_step`` reads ``batch.voxels`` and then ``feature_tensor`` off it -- the same
    attribute ``dino/loss.py:249`` reads off a real ``Voxels``, and the only one a model needs
    when the backbone is a ``Linear``. So it is the whole of this class.
    """

    def __init__(self, feats: Tensor, meta: dict | None = None):
        self.voxels = FakeVoxels(feats)
        self.meta = meta or {}

    @property
    def batch_size(self) -> int:
        return int(self.voxels.feature_tensor.shape[0])

    def to(self, device: Any) -> FakeBatch:
        return FakeBatch(self.voxels.feature_tensor.to(device), self.meta)


class _FakeDataset(torch.utils.data.Dataset):
    def __init__(self, n: int, dim: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, dim, generator=g)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int) -> Tensor:
        return self.x[i]


def fake_loader(steps: int = 4, batch: int = 2, dim: int = 4, seed: int = 0):
    """A real ``DataLoader`` yielding ``FakeBatch``, since ``setup_dataloaders`` takes
    nothing else and the ``_FabricDataLoader`` it returns is what production iterates."""
    return torch.utils.data.DataLoader(
        _FakeDataset(steps * batch, dim, seed),
        batch_size=batch,
        shuffle=False,
        collate_fn=lambda items: FakeBatch(torch.stack(items)),
        drop_last=True,
    )


# `_STAT_SCALARS` and the two mappers below used to live in `wcfm/model/legacy.py`, on a shim
# that imported the old repo. That shim is gone -- comparing the two frameworks by running each
# and comparing evaluations is the only comparison that can actually fail, where a shim sharing
# the old code cannot detect a discrepancy in the code it shares. What the shim exercised about
# the *engine* is kept here, because Stage 3's native DINO has the same shape: a backward per
# view, an EMA teacher no wrapper reduces, a lazily-created centring buffer, and a schedule
# that needs the run's total length.

_STAT_SCALARS = (
    "loss",
    "t_ent",
    "s_ent",
    "kl",
    "cov",
    "var",
    "loss_masked",
    "loss_unmasked",
    "loss_charge",
    "loss_occ",
)


def scalars_from(stats: Any) -> dict[str, float]:
    """Absent-never-null: a quantity this configuration does not produce is left out of the
    record rather than written as a null, which is what lets a reader tell "this run had no
    teacher" from "this step failed to measure entropy"."""
    out = {n: float(v) for n in _STAT_SCALARS if (v := getattr(stats, n, None)) is not None}
    if (n_pairs := getattr(stats, "n_pairs", None)) is not None:
        out["n_pairs"] = float(n_pairs)
    return out


def observables_from(stats: Any) -> dict[str, Tensor]:
    """The feature matrices, detached. A field that is not ``[N, D]`` is skipped rather than
    reshaped -- guessing an orientation is how a diagnostic starts reporting a number that
    means nothing."""
    out: dict[str, Tensor] = {}
    for field, name in (
        ("s_backbone", "student/backbone"),
        ("t_backbone", "teacher/backbone"),
        ("s_out", "student/out"),
        ("t_out", "teacher/out"),
    ):
        voxels = getattr(stats, field, None)
        feats = getattr(voxels, "feature_tensor", None) if voxels is not None else None
        if feats is not None and feats.dim() == 2:
            out[name] = feats.detach().float().clone()
    return out


class DinoShapedModule(nn.Module):
    """The DINO shape as a ``TrainingModule``, on the one DDP path.

    Five things such a model has to get right, and where each of them is:

    * **The reduced unit is ``forward``** -- the student path -- reached through
      ``ctx.module``. DDP arms its reducer inside ``DistributedDataParallel``'s
      forward, so a view that computed on ``self`` would reduce nothing, silently, which is
      the ADR 0001 failure and what the suite's control asserts.
    * **The ENGINE backwards the returned loss** (ADR 0006). Under ``16-mixed`` the scaler
      lives in ``fabric.backward`` and the matching unscale in ``_FabricOptimizer.step``; a
      model calling ``tensor.backward()`` scales neither while the optimizer unscales, so
      gradients come out wrong by the scale factor with no error anywhere. Going through the
      capability is what makes every ``run.precision`` value usable.
    * **No per-view suppression any more.** Every view is summed into one loss and the
      engine takes a single backward under its own accumulation gate, so there is nothing
      left for the model to suppress. Until 2026-09-10 this module backwarded per view and
      nested its own gate inside the engine's; ADR 0001's amendment records why that was
      never DINO's shape.
    * **The teacher is EMA-updated in ``on_step_end``**, after the optimizer step, so it
      distils against a student the ranks agree on. Centring is updated there too and is
      gated on configuration, never on whether a tensor happens to be ``None`` -- it
      all-reduces, so a rank-dependent skip hangs the group.
    * **The momentum schedule comes from ``ctx.extra``**, because ``total_iters`` is
      ``epochs * len(loader)`` and so is unknowable at construction.
    """

    def __init__(
        self,
        *,
        dim: int = 4,
        n_pairs: int = 3,
        use_proj_head: bool = True,
        has_teacher: bool = True,
        momentum_start: float = 0.996,
        momentum_end: float = 1.0,
    ):
        super().__init__()
        self.model = FakeDinoModel(dim=dim, use_proj_head=use_proj_head, n_pairs=n_pairs)
        # A submodule, not a plain attribute: that is what puts the lazily-created centring
        # buffer into `state_dict()` and therefore into the checkpoint.
        self.loss_fn = FakeLoss(dim=dim)
        self.has_teacher = has_teacher
        self.momentum_start = float(momentum_start)
        self.momentum_end = float(momentum_end)
        self._momentum: Any = None
        self._last_teacher_out: Any = None
        self._observables: dict[str, Tensor] = {}
        # Observed by the tests: whether DDP would actually reduce at each backward.
        # `require_backward_grad_sync` is the flag `no_sync()` toggles and the reducer
        # consults, so it is the only observation that sees *both* gates -- the model's
        # per-view one and the engine's accumulation one, which nest.
        # (autocast enabled, autocast dtype) as seen inside `forward`. The output dtype
        # cannot tell you: three separate things upcast a bf16 result on the way out.
        self._autocast: tuple[bool, torch.dtype] = (False, torch.float32)

    # -------------------------------------------------------- the reduced unit

    def forward(self, rows: Tensor) -> Tensor:
        """The student path, which is what ``fabric.setup`` wrapped and ``ctx.module`` reaches."""
        device_type = "cuda" if rows.is_cuda else "cpu"
        self._autocast = (
            torch.is_autocast_enabled(device_type),
            torch.get_autocast_dtype(device_type),
        )
        return self.model(rows)

    # -------------------------------------------------------- TrainingModule

    def training_step(self, batch: Any, ctx: Any) -> Any:
        from wcfm.engine.protocol import StepOutput

        if self._momentum is None:
            self._build_momentum(ctx)
        voxels = batch.voxels if hasattr(batch, "voxels") else batch
        rows = voxels.feature_tensor if hasattr(voxels, "feature_tensor") else voxels

        target = self.model.encode_teacher(rows)

        total = 0.0
        last_out = None
        step_loss = None
        n_pairs = self.model.n_pairs
        for _pair in range(n_pairs):
            # `ctx.module(rows)`, never `self.forward(rows)` or `self.model.student(rows)`:
            # the wrapper is what arms the reducer and applies the configured precision.
            out = ctx.module(rows)
            view_loss = ((out.float() - target.float()) ** 2).mean() / n_pairs
            step_loss = view_loss if step_loss is None else step_loss + view_loss
            total += float(view_loss.detach()) * n_pairs
            last_out = out

        stats = FakeStats(
            loss=total / n_pairs,
            n_pairs=n_pairs,
            kl=0.5,
            s_backbone=FakeVoxels(last_out.detach()),
            t_backbone=FakeVoxels(target.detach()),
            s_out=FakeVoxels(last_out.detach()),
            t_out=FakeVoxels(target.detach()),
        )
        self._last_teacher_out = stats.t_out
        self._observables = observables_from(stats)
        return StepOutput(
            scalars=scalars_from(stats), n_samples=batch.batch_size, loss=step_loss
        )

    def param_groups(self) -> list[dict]:
        """Trainable parameters only; the teacher's are frozen and must not appear."""
        groups = [{"name": "backbone", "params": list(self.model.student.parameters())}]
        if self.model.student_head is not None:
            groups.append(
                {"name": "head", "params": list(self.model.student_head.parameters())}
            )
        return groups

    def observables(self) -> dict[str, Tensor]:
        return self._observables

    def on_step_end(self, ctx: Any) -> None:
        momentum = float(self._momentum[ctx.step])
        if self.has_teacher:
            self.model.update_teacher(momentum)
            self.loss_fn.update_center(self._last_teacher_out)
        self._last_teacher_out = None

    def grad_taxonomy(self) -> dict[str, tuple[str, ...]]:
        """Trailing dots are load-bearing: ``"model.student_head..."`` starts with
        ``"model.student"``, so a boundary-less prefix puts every head parameter in the
        backbone group. The engine takes the longest matching prefix, and these are the
        unwrapped names -- ``StepRecord.named_parameters`` is the unwrapped module's."""
        taxonomy: dict[str, tuple[str, ...]] = {"backbone": ("model.student.",)}
        if self.model.student_head is not None:
            taxonomy["head"] = ("model.student_head.",)
        return taxonomy

    def provenance(self) -> dict:
        return {"module": type(self).__name__, "n_pairs": self.model.n_pairs}

    def load_state_dict(self, sd: dict, strict: bool = True):  # type: ignore[override]
        """Materialise the loss's lazy buffers, then load strictly.

        ``center`` is created on first use, so a module that has not stepped does not have it
        and a strict load rejects it as unexpected -- and "a resumed run silently restarts
        centring" is the bug the checkpoint schema exists to fix, which is worthless if the
        file cannot be read back. Loading non-strictly would "work" and would also swallow a
        genuine architecture mismatch, so only the missing buffers are created.

        **On the module's own device, not the checkpoint's.** The engine loads with
        ``map_location="cpu"`` on purpose -- rank 0's file has to be readable anywhere -- so a
        buffer created from the loaded tensor lands on the CPU while the rest of the module is
        on the GPU, and the next thing that touches both raises "Expected all tensors to be on
        the same device". Invisible on gloo, where both sides are CPU: found by the 2-GPU nccl
        run (cluster 2258) after 22 CPU-rank tests passed.

        This is the shape ``PixelDINOLoss`` is designed out of -- its ``center`` is registered
        eagerly at construction, so ``fabric.setup()`` moves it and there is nothing to
        materialise later. The laziness is reproduced here because it is what the old repo did
        (``loss.py:73``), and a model that keeps it has to do this itself: the engine cannot
        repair it generically, because rebinding ``param.data`` after ``fabric.setup()`` breaks
        DDP's bucket views.
        """
        device = next(self.parameters()).device
        for key, value in sd.items():
            prefix, _, leaf = key.rpartition(".")
            if prefix == "loss_fn" and not hasattr(self.loss_fn, leaf):
                if isinstance(value, Tensor):
                    self.loss_fn.register_buffer(leaf, torch.zeros_like(value, device=device))
        return super().load_state_dict(sd, strict=strict)

    def _build_momentum(self, ctx: Any) -> None:
        """The teacher momentum cosine, from the geometry the engine put in ``ctx.extra``:
        ``total_iters`` is ``epochs * len(loader)`` and so unknowable at construction."""
        from wcfm.engine.optim import CosineScheduler

        total = int(ctx.extra.get("total_iters", 0)) or 1
        self._momentum = CosineScheduler(
            self.momentum_start, self.momentum_end, epochs=1, steps_per_epoch=total
        )
