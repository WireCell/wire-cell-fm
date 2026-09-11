"""A ``TrainingModule`` the CPU suite can run the whole loop over.

It lives in ``tests/``, not under ``wcfm/model/``, for two reasons. It keeps
``test_import_graph.py`` honest -- a toy inside the framework would be a framework module that
knows what a training step is. And "no backbone forward is CPU-testable" is a statement about
the backbone, not about the loop: accumulation, the engine's reduce gating, the non-finite
guard, clipping, the checkpoint round trip and resume are all reachable on a CPU with an
``nn.Linear``, and they are where the engine's own bugs would be.

``ToyModule`` deliberately owns its backward and reports through ``StepOutput``, so it
exercises the contract rather than a convenient subset of it. ``views`` > 1 makes it do a
separate backward per view, which is the DINO shape that made the engine hand over
``backward`` as a capability in the first place.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from wcfm.engine.protocol import StepContext, StepOutput


class ToyBatch:
    """Stands in for ``Batch(voxels, meta)``: carries meta, and moves on ``.to``."""

    def __init__(self, x: Tensor, meta: dict | None = None):
        self.x = x
        self.meta = meta or {}

    def to(self, device: Any) -> ToyBatch:
        return ToyBatch(self.x.to(device), self.meta)

    @property
    def batch_size(self) -> int:
        return int(self.x.shape[0])


class ToyModule(nn.Module):
    """Linear regression onto zero, with a buffer that must survive a checkpoint.

    ``centre`` is the stand-in for the DINO centring buffer -- the thing the old repo created
    lazily and never saved. A resume test that does not check a buffer would pass against the
    bug this schema exists to fix.
    """

    def __init__(self, dim: int = 4, views: int = 1, explode_at: int | None = None):
        super().__init__()
        self.net = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, dim)
        self.register_buffer("centre", torch.zeros(dim))
        self.views = views
        self.explode_at = explode_at
        self.step_ends = 0
        self.last_ctx: StepContext | None = None
        # The last step's features, detached, as an [N, D] matrix -- which is the shape the
        # spectral collectors read. A 1-D buffer is not an observable they can summarise.
        self._feats: Tensor = torch.zeros(1, dim)
        self._feats_dtype: torch.dtype = torch.float32
        # (autocast enabled, autocast dtype) as seen inside `forward`. See its docstring.
        self._autocast: tuple[bool, torch.dtype] = (False, torch.float32)

    # ---------------------------------------------------- TrainingModule

    @staticmethod
    def rows_of(batch: Any) -> Tensor:
        """An ``[N, dim]`` input from either a ``ToyBatch`` or a real ``Batch``.

        A real ``Batch`` carries ``voxels``, not ``x``: the whole point of running this toy
        against the production loader is that the difference shows up here, in the test suite,
        rather than as an ``AttributeError`` inside a real module in Stage 3. The features are
        a single charge column, so they are tiled to the toy's width -- no sparse convolution
        is involved, which is what keeps this CPU-testable.
        """
        if hasattr(batch, "x"):
            return batch.x
        feats = batch.voxels.batched_features.batched_tensor.float()
        return feats.expand(-1, 4) if feats.shape[-1] == 1 else feats[:, :4]

    def forward(self, x: Tensor) -> Tensor:
        """One view's network pass. This is what the DDP wrapper wraps, so it is what
        ``ctx.module(...)`` reaches -- and calling it per view is what arms the reducer once
        per view, the shape a per-view backward needs.

        It also records whether it is running under autocast, which is the **only** reliable
        way to observe the configured precision. Three separate things upcast a bf16 result
        back to fp32 on the way out, and each one hid this in turn: a module's fp32 bias wins
        the type promotion on the add, ``_FabricModule.forward`` calls
        ``precision.convert_output`` (``wrappers.py:138``) which casts to the default dtype,
        and anything the test then does in fp32. So the dtype that comes *out* says nothing;
        whether autocast was *enabled* is the question.
        """
        device_type = "cuda" if x.is_cuda else "cpu"
        self._autocast = (
            torch.is_autocast_enabled(device_type),
            torch.get_autocast_dtype(device_type),
        )
        return self.head(self.net(x))

    def training_step(self, batch: Any, ctx: StepContext) -> StepOutput:
        rows = self.rows_of(batch)
        total: Any = None
        for _view in range(self.views):
            # `ctx.module(rows)`, never `self.forward(rows)`: the wrapper arms DDP's reducer
            # and applies the configured precision. `self` does neither, silently.
            out = ctx.module(rows)
            # What the wrapper's autocast actually produced, before anything upcasts it. The
            # engine's precision setting is only observable here.
            self._feats_dtype = out.dtype
            self._feats = out.detach().float().clone()
            loss = (out.float() ** 2).mean()
            if self.explode_at is not None and ctx.step == self.explode_at:
                loss = loss * float("inf")
            total = loss if total is None else total + loss
        assert total is not None
        self.last_ctx = ctx
        # Every view summed into ONE loss, handed to the engine to differentiate. The
        # accumulation gate is the engine's, not this module's.
        return StepOutput(
            scalars={"loss": float(total.detach()) / self.views, "n_voxels": float(rows.shape[0])},
            n_samples=batch.batch_size,
            loss=total / self.views,
        )

    def param_groups(self) -> list[dict]:
        return [
            {"name": "backbone", "params": list(self.net.parameters())},
            {"name": "head", "params": list(self.head.parameters()), "lr_scale": 2.0},
        ]

    def observables(self) -> dict[str, Tensor]:
        """Detached, and built after backward -- holding live handles pins the autograd graph.
        The names are the module's own; nothing in the framework parses them."""
        return {"toy/feat": self._feats, "toy/centre": self.centre.detach().clone()}

    def on_step_end(self, ctx: StepContext) -> None:
        self.step_ends += 1
        with torch.no_grad():
            self.centre += 1.0


class ToyDataset(Dataset):
    """Map-style, returning ``(tensor, meta)`` -- the shape ``data.collate`` consumes."""

    def __init__(self, n: int, dim: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, dim, generator=g)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, i: int) -> tuple[Tensor, dict]:
        return self.x[i], {"event_key": i}


def toy_collate(items: list[tuple[Tensor, dict]]) -> ToyBatch:
    return ToyBatch(
        torch.stack([x for x, _ in items]),
        {"event_key": [m["event_key"] for _, m in items]},
    )


def toy_loader(steps: int = 4, batch: int = 2, dim: int = 4, seed: int = 0) -> DataLoader:
    """A real ``DataLoader``, not a list of batches.

    ``Fabric.setup_dataloaders`` accepts nothing else, and wrapping the real thing is the
    point: the ``_FabricDataLoader`` it returns is what production iterates, so a test over a
    plain list would skip the one wrapper the engine actually puts in the path.
    """
    return DataLoader(
        ToyDataset(steps * batch, dim, seed),
        batch_size=batch,
        shuffle=False,
        collate_fn=toy_collate,
        drop_last=True,
    )


class TwoPartToyModule(nn.Module):
    """**The control.** A DINO-shaped model that computes on ``self`` and so reduces nothing.

    Named submodules, a per-view backward and an EMA teacher, with the step calling
    ``self.student(...)`` and ``self.student_head(...)`` directly -- exactly as the old
    repo's ``DINODuneModel.forward_backward`` does (``model.py:381,383,386,414``). Under the
    one DDP path that is a **bug**, and this class exists to prove the suite can see it: it
    must diverge across ranks, or every other assertion in the distributed file is
    uninformative, since the correct spelling computes identical numbers and differs only in
    whether DDP was armed.

    ``DinoShapedModule`` in ``fake_dino.py`` is the same shape written correctly -- the
    student path behind ``forward``, reached per view through ``ctx.module`` -- and is the
    Stage 3 template. Read the two together.

    ``teacher`` has ``requires_grad=False`` and is EMA-updated under ``no_grad``, so DDP's
    reducer never includes it while DDP's construction-time broadcast still keeps the ranks
    agreeing on it. That combination is why a frozen teacher needs no special handling at all
    on this path.
    """

    def __init__(self, dim: int = 4, views: int = 1, momentum: float = 0.9):
        super().__init__()
        self.student = nn.Linear(dim, dim, bias=False)
        self.student_head = nn.Linear(dim, dim, bias=False)
        self.teacher = nn.Linear(dim, dim, bias=False)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.register_buffer("centre", torch.zeros(dim))
        self.views = views
        self.momentum = momentum
        self.step_ends = 0
        self._feats: Tensor = torch.zeros(1, dim)

    def encode_student(self, rows: Tensor) -> Tensor:
        """Backbone then head, on the bare attributes -- so no reducer is ever armed.

        The correct form is to put this behind ``forward`` and call ``ctx.module(rows)``. See
        ``fake_dino.DinoShapedModule``.
        """
        return self.student_head(self.student(rows))

    def training_step(self, batch: Any, ctx: StepContext) -> StepOutput:
        rows = ToyModule.rows_of(batch)
        with torch.no_grad():
            target = self.teacher(rows).detach()
        total: Any = None
        for _view in range(self.views):
            out = self.encode_student(rows)
            self._feats = out.detach().float().clone()
            loss = ((out.float() - target.float()) ** 2).mean()
            total = loss if total is None else total + loss
        assert total is not None
        return StepOutput(
            scalars={"loss": float(total.detach()) / self.views},
            n_samples=batch.batch_size,
            loss=total / self.views,
        )

    def param_groups(self) -> list[dict]:
        """Trainable parameters only -- the teacher's are frozen and must not appear.

        Read off the attributes. DDP does not copy parameters, so the tensors the optimizer
        gets are the same objects the reducer writes into.
        """
        return [
            {"name": "backbone", "params": [p for p in self.student.parameters()]},
            {
                "name": "head",
                "params": [p for p in self.student_head.parameters()],
                "lr_scale": 2.0,
            },
        ]

    def observables(self) -> dict[str, Tensor]:
        return {"toy/feat": self._feats}

    def on_step_end(self, ctx: StepContext) -> None:
        self.step_ends += 1
        with torch.no_grad():
            for t, s in zip(self.teacher.parameters(), self.student.parameters(), strict=True):
                t.mul_(self.momentum).add_(s.detach(), alpha=1.0 - self.momentum)
            self.centre += 1.0


class TwoTermToyModule(nn.Module):
    """Two terms over a shared trunk, honouring the engine's `collect_term_gradients` request.

    The shape `SslModule` presents to `TermGrad`, small enough to run on two CPU ranks. Both
    heads run inside the wrapped forward (ADR 0007), the per-term gradients are taken with
    `autograd.grad(..., retain_graph=True)` before the real backward, and they are all-reduced
    here -- not in the collector, which must not perform a collective.
    """

    def __init__(self, dim: int = 4, opposed: bool = False):
        super().__init__()
        self.trunk = nn.Linear(dim, dim, bias=False)
        self.head_a = nn.Linear(dim, dim, bias=False)
        self.head_b = nn.Linear(dim, dim, bias=False)
        # When True the two terms are driven to fight over the trunk, so a test can assert the
        # collector SEES a conflict rather than merely reporting a number.
        self.opposed = opposed
        self.last_term_gradients: dict[str, Tensor] | None = None
        self.saw_request: list[bool] = []

    def forward(self, x):
        h = self.trunk(x)
        return self.head_a(h), self.head_b(h)

    def rows_of(self, batch: Any) -> Tensor:
        return batch.x

    def training_step(self, batch: Any, ctx: StepContext) -> StepOutput:
        want = bool(ctx.extra.get("collect_term_gradients"))
        self.saw_request.append(want)
        shared = [p for p in self.trunk.parameters() if p.requires_grad]

        out_a, out_b = ctx.module(self.rows_of(batch))
        loss_a = (out_a.float() ** 2).mean()
        loss_b = -(out_b.float() ** 2).mean() if self.opposed else (out_b.float() ** 2).mean()

        acc: dict[str, Tensor] = {}
        if want:
            for name, loss in (("a", loss_a), ("b", loss_b)):
                grads = torch.autograd.grad(loss, shared, retain_graph=True, allow_unused=True)
                acc[name] = torch.cat(
                    [
                        (g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(grads, shared, strict=True)
                    ]
                ).detach()


        self.last_term_gradients = (
            {k: ctx.all_reduce(v, reduce_op="mean") for k, v in sorted(acc.items())}
            if want
            else None
        )
        return StepOutput(
            scalars={"loss": float((loss_a + loss_b).detach())},
            n_samples=batch.batch_size,
            loss=loss_a + loss_b,
        )

    def param_groups(self) -> list[dict]:
        return [{"name": "trunk", "params": list(self.parameters())}]

    def observables(self) -> dict[str, Tensor]:
        return {}

    def on_step_end(self, ctx: StepContext) -> None:
        return None
