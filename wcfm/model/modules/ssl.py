"""`SslModule`: the `TrainingModule` for the DINO family, and the reduced unit of DDP.

Every word in `training_step` is model vocabulary:

    normalise(batch)                                         # here, in place
    plan = augment(batch)                                    # crops, then a mask per view
    teacher = [ctx.module(v.clean, teacher=True, terms=..)]  # no_grad; only if a term needs it
    requests = [ViewRequest(view.voxels, inject, terms) for each scored view]
    encoded  = ctx.module(requests, taps=..)                 # ONE forward, every view
    loss     = sum over views, over active terms of
                   t.weight(step) * t.compute(...).loss / t.total_contrib(plan)
    return StepOutput(..., loss=loss)                        # the ENGINE backwards it

There is one DDP path, and this is how a DINO-shaped model fits it. `forward` is the reduced
unit and dispatches on its arguments: it runs the backbone (the student, or the frozen teacher
twin under `teacher=True`) and every named term's head, and returns both. One forward per step
therefore carries every trainable parameter the loss will differentiate. DDP arms its reducer
once per forward and, with `find_unused_parameters`, traverses from the forward's outputs to
see which parameters took part, so a head applied outside the forward is marked unused at the
end of it and then receives a gradient: "Expected to mark a variable ready only once".

The teacher call goes through the same `ctx.module` under `no_grad`, which gives it the
configured precision without arming the reducer (`DistributedDataParallel.forward` skips
`prepare_for_backward` when grad is disabled), so student and teacher compute in the same
dtype. The teacher's parameters are `requires_grad=False` and so sit outside the reducer while
still inside the wrapped module, which is what broadcasts them from rank 0.

The engine sees `training_step`, a `StepOutput`, `param_groups`, `observables`, `on_step_end`
and a state dict. The centring buffer is in that state dict from step 0.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from warpconvnet.geometry.types.voxels import Voxels

from wcfm.data.voxels import Batch
from wcfm.engine.protocol import StepContext, StepOutput

from ..augment.transforms import FeatureLogTransform
from ..augment.views import Augment
from ..backbones.base import Backbone, FeatureBundle, Injection
from ..terms.base import TeacherOutput, Term


class EmaTeacher:
    """Configuration of the EMA teacher: a cosine momentum schedule over the run."""

    def __init__(self, momentum_start: float = 0.996, momentum_end: float = 1.0):
        if not 0.0 <= momentum_start <= 1.0 or not 0.0 <= momentum_end <= 1.0:
            raise ValueError("teacher momentum must be in [0, 1]")
        self.momentum_start = float(momentum_start)
        self.momentum_end = float(momentum_end)

    def __repr__(self) -> str:
        return f"EmaTeacher({self.momentum_start:g} -> {self.momentum_end:g})"


class NoTeacher:
    """The explicit absence of a teacher, so `model/teacher=none` is a real option."""

    def __repr__(self) -> str:
        return "NoTeacher()"


def _inference_context(module, step: int, device) -> StepContext:
    """A `StepContext` for an offline pass: real position, inert capabilities.

    `Term.compute` takes a ctx, so an offline caller needs one. Every capability on it is a
    no-op here by construction -- there is no Fabric and no reducer -- and `module` is the
    bare module rather than a wrapped handle, which is correct precisely because there is
    nothing to arm.
    """
    return StepContext(
        step=step,
        epoch=0,
        device=device,
        module=module,
        all_reduce=lambda t, *a, **k: t,
        is_last_microstep=True,
    )


def _ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for s, t in zip(student.parameters(), teacher.parameters(), strict=True):
            t.mul_(momentum).add_(s.detach(), alpha=1.0 - momentum)


class ViewRequest(NamedTuple):
    """One view's share of the student forward.

    A sequence of these turns the step's view loop into a single `forward`, which is what
    makes one backward enough. `terms` names the heads to run on this view, so a term that does
    not run here costs nothing.

    It is a NamedTuple, and that is load-bearing. Under a mixed precision, Fabric runs
    `_apply_to_collection` over the forward's arguments to cast floating-point tensors, and it
    treats dataclasses specially: a frozen one raises outright (`allow_frozen=False`), and a
    mutable one is deepcopied, which would clone this view's voxels on every step. A NamedTuple
    is rebuilt element by element instead, and `Voxels` and `Injection` pass through untouched.
    The failure only appears under `bf16-mixed`, so an fp32 test suite stays green.
    """

    voxels: Voxels
    inject: Injection | None
    terms: tuple[str, ...]


class SslModule(nn.Module):
    """Backbone + optional EMA teacher + augment stage + a `ModuleDict` of terms."""

    def __init__(
        self,
        *,
        backbone: Backbone,
        terms: Mapping[str, Term],
        augment: Augment,
        teacher: EmaTeacher | NoTeacher | None = None,
        normalize: FeatureLogTransform | None = None,
        observe_taps: Iterable[str] = (),
    ):
        super().__init__()
        if not terms:
            raise ValueError("a model needs at least one term; `model.terms` is empty")
        self.backbone = backbone
        self.augment = augment
        self.normalize = normalize
        self.observe_taps = tuple(observe_taps)
        #: Per-term gradients over the backbone, already all-reduced, from the last step
        #: that asked for them. Read by `Trainer._record` with `getattr`, like the taxonomy.
        self.last_term_gradients: dict[str, Tensor] | None = None
        self.teacher_cfg: EmaTeacher | None = (
            teacher if isinstance(teacher, EmaTeacher) else None
        )

        self.terms = nn.ModuleDict()
        for name, term in terms.items():
            if not isinstance(term, Term):
                raise TypeError(f"model.terms.{name} is {type(term).__name__}, not a Term")
            term.name = str(name)
            term.build(backbone)
            self.terms[str(name)] = term

        # A term that READS a tap must be given it, whatever the metrics config asked for:
        # `observe_taps` is an observation request and a term's is a data dependency. Sorted
        # so the tuple -- and therefore the backbone's work -- is identical on every rank.
        self.required_taps = tuple(
            sorted(set(self.observe_taps) | {tap for x in terms.values() for tap in x.wants_taps()})
        )

        self.teacher_backbone: Backbone | None = None
        if self.teacher_cfg is not None:
            self.teacher_backbone = copy.deepcopy(backbone)
            for p in self.teacher_backbone.parameters():
                p.requires_grad_(False)
            self.teacher_backbone.eval()

        self._momentum: Any = None
        self._last_student: FeatureBundle | None = None
        self._last_teacher: FeatureBundle | None = None
        self.validate()

    # ------------------------------------------------------------------- validation

    def validate(self) -> None:
        """Cross-term rules, each message naming the fix. Per-term rules live on the term."""
        has_teacher = self.teacher_backbone is not None
        needs_teacher = [n for n, t in self.terms.items() if t.requires_teacher]
        if needs_teacher and not has_teacher:
            raise ValueError(
                f"terms {needs_teacher} distil from a teacher and model.teacher is not ema"
            )
        if has_teacher and not needs_teacher:
            # A never-trained teacher in a checkpoint lets `--source=teacher` extraction
            # quietly return features from initialisation weights.
            raise ValueError(
                "model.teacher is ema but no term uses a teacher; select model/teacher=none"
            )
        masker = self.augment.masker
        for name, term in self.terms.items():
            if term.requires_masking and masker is None:
                raise ValueError(f"term {name!r} needs a masker and none is configured")
            missing = sorted(set(term.inject_roles) - set(self.backbone.inject_roles))
            if missing:
                raise ValueError(
                    f"term {name!r} injects roles {missing} but backbone "
                    f"{type(self.backbone).__name__} holds tokens only for "
                    f"{list(self.backbone.inject_roles)}"
                )
            term.validate(self.backbone, self.augment, has_teacher)
            # Optional hook, the same shape as `provenance` and `observables`. `Term.validate`
            # is not handed the charge transform, so a term that has to agree with it -- one
            # distilling from a checkpoint trained on another production's bounds -- asks here.
            check = getattr(term, "validate_normalize", None)
            if check is not None:
                check(self.normalize)
        if (
            masker is not None
            and masker.requires_full_canvas
            and self.augment.cropper is not None
        ):
            # The grid is defined on the canvas, and under cropping the canvas the masker
            # sees is the crop, whose size it was not built for.
            raise ValueError(
                f"{type(masker).__name__} masks on a grid over the full canvas and cannot run "
                "on a crop; select model/augment=mask_only, or a pixel or block masker"
            )
        self._validate_occupancy_grid(masker)

    def _validate_occupancy_grid(self, masker: Any) -> None:
        """The masker's cell must tile the resolution the occupancy question is asked at.

        A cell that does not divide the read stride puts part of a wiped cell in a
        half-resolution voxel the rest of which survived: the candidate at that voxel is
        labelled from the removed set, but the feature it reads was never fully removed, so the
        label and the input disagree on a rim of every cell. Nothing in the loss shows it --
        the run trains on slightly wrong targets.
        """
        for name, term in self.terms.items():
            stride = getattr(term, "READ_STRIDE", None)
            if stride is None or masker is None:
                continue
            for axis in ("cell_w", "cell_h"):
                cell = getattr(masker, axis, None)
                if cell is not None and int(cell) % int(stride):
                    raise ValueError(
                        f"term {name!r} reads occupancy at stride {stride} and the masker's "
                        f"{axis} is {cell}, which it does not divide; every cell would carry "
                        f"a mislabelled rim. Set model.augment.masker.{axis} to a multiple "
                        f"of {stride}"
                    )

    # ---------------------------------------------------------------- the reduced unit

    def forward(
        self,
        voxels: Voxels | Sequence[ViewRequest],
        inject: Injection | None = None,
        taps: Iterable[str] = (),
        *,
        teacher: bool = False,
        terms: Iterable[str] = (),
    ) -> tuple[FeatureBundle, dict[str, Any]] | list[tuple[FeatureBundle, dict[str, Any]]]:
        """The DDP-wrapped forward: the backbone, then every named term's head.

        Two shapes, dispatched on the first argument, because there is one DDP path and a
        model with more than one reduced shape dispatches rather than reaching past the
        wrapper:

        - a `Voxels` -> `(bundle, {term_name: head_output})` for that one view. This is the
          teacher's shape, and it runs under `no_grad`.
        - a sequence of `ViewRequest` -> one `(bundle, heads)` per request, every view of the
          step inside a single `forward`. This is the student's shape.

        The second exists so the step needs exactly one backward. DDP arms its reducer once per
        `forward()`, so several forwards feeding one backward over-reduces. Keeping the view
        loop in here is what avoids that.

        `Term.compute` is parameter-free, so `find_unused_parameters` traversing from these
        outputs reaches every parameter the loss will differentiate.
        """
        if not isinstance(voxels, Voxels):
            requests = list(voxels)
            return [
                self._encode(r.voxels, r.inject, taps, teacher=teacher, terms=r.terms)
                for r in requests
            ]
        return self._encode(voxels, inject, taps, teacher=teacher, terms=terms)

    def _encode(
        self,
        voxels: Voxels,
        inject: Injection | None,
        taps: Iterable[str],
        *,
        teacher: bool,
        terms: Iterable[str],
    ) -> tuple[FeatureBundle, dict[str, Any]]:
        """One view: the backbone, then every named term's head.

        Called only from `forward`, so everything here is inside the wrapper's arming.
        """
        if teacher:
            assert self.teacher_backbone is not None, "no teacher configured"
            bundle = self.teacher_backbone(voxels, None, taps)
            heads = {n: self.terms[n].teacher_head_forward(bundle) for n in terms}
        else:
            bundle = self.backbone(voxels, inject, taps)
            heads = {n: self.terms[n].head_forward(bundle) for n in terms}
        return bundle, heads

    # --------------------------------------------------------------- TrainingModule

    def training_step(self, batch: Batch, ctx: StepContext) -> StepOutput:
        if self._momentum is None and self.teacher_cfg is not None:
            self._build_momentum(ctx)

        if self.normalize is not None:
            self.normalize(batch.voxels)  # in place, before the masker
        plan = self.augment(batch)
        n_voxels = int(batch.voxels.coordinate_tensor.shape[0])

        teacher: list[TeacherOutput] | None = None
        if self.teacher_backbone is not None:
            needs = [n for n, t in self.terms.items() if t.requires_teacher]
            with torch.no_grad():
                teacher = [
                    ctx.module(v.clean, None, self.required_taps, teacher=True, terms=needs)
                    for v in plan.globals
                ]
            self._last_teacher = teacher[0][0]
        for term in self.terms.values():
            term.begin_step(plan, teacher)

        active = [
            [t for t in self.terms.values() if t.runs_on(plan, k)] for k in range(plan.n_views)
        ]
        scored = [k for k, a in enumerate(active) if a]
        if not scored:
            raise RuntimeError("no term runs on any view of this plan; check the augment config")
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        # Asked for by the engine only on a step where a collector declaring `needs =
        # {"term_grads"}` fires -- `needs` is a class attribute, so every rank agrees on which
        # steps those are, which is what makes the all-reduce at the end safe.
        want_grads = bool(ctx.extra.get("collect_term_gradients"))
        shared_params = [p for p in self.backbone.parameters() if p.requires_grad]

        # ONE forward over every scored view: one arming, one backward. `forward`'s docstring
        # says why the view loop lives in there. `requests` is built in `scored` order and the
        # outputs come back in that order, which is what lets the loss loop below pair them up.
        requests = [
            ViewRequest(
                voxels=plan.views[k].voxels,
                inject=Injection.merge(t.injection_request(plan.views[k]) for t in active[k]),
                terms=tuple(t.name for t in active[k]),
            )
            for k in scored
        ]
        encoded = ctx.module(requests, taps=self.required_taps)

        # Per-term contributions summed ACROSS views before anything differentiates them: a
        # term scored on three views contributes three times, exactly as when each view was
        # backwarded separately.
        loss: Tensor | None = None
        term_contribs: dict[str, Tensor] = {}
        for k, (bundle, heads) in zip(scored, encoded, strict=True):
            for term in active[k]:
                out = term.compute(bundle, heads[term.name], k, plan, teacher, ctx)
                contrib = term.weight(ctx.step) * out.loss / term.total_contrib(plan)
                loss = contrib if loss is None else loss + contrib
                term_contribs[term.name] = (
                    contrib
                    if term.name not in term_contribs
                    else term_contribs[term.name] + contrib
                )
                for key, value in out.scalars.items():
                    sums[key] = sums.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + out.counts.get(key, 1)
            self._last_student = bundle  # the last SCORED view, as before
        assert loss is not None
        total = float(loss.detach())

        # Per-term gradients, taken BEFORE the real backward and only when a collector firing
        # this step asked for them. Spike (e) measured that `autograd.grad` with
        # `retain_graph` leaves `.grad` untouched, performs no reduction, and leaves this
        # backward reducing correctly -- so this cannot corrupt accumulation, and the vectors
        # it returns are rank-local and have to be reduced (below) before anyone measures an
        # angle between them.
        #
        # One call per TERM, not per term per view: with every view in one graph the summed
        # contribution differentiates in a single pass, and the gradient of a sum is the sum
        # of the gradients.
        grad_acc: dict[str, Tensor] = {}
        if want_grads:
            for name in sorted(term_contribs):
                grads = torch.autograd.grad(
                    term_contribs[name], shared_params, retain_graph=True, allow_unused=True
                )
                grad_acc[name] = torch.cat(
                    [
                        (g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(grads, shared_params, strict=True)
                    ]
                ).detach()

        # No backward here: `loss` goes back on the StepOutput and the ENGINE differentiates
        # it. One forward, one loss, one backward, and the model never learns about the
        # accumulation gate or the gradient scaler.

        # The reduction the collector is forbidden to do. `cos(mean(g_a), mean(g_b))` is the
        # conflict in the gradient the optimizer actually applies; a mean of per-rank cosines is
        # a different and noisier quantity, so this has to happen on the vectors, here, where
        # `ctx.all_reduce` is legal. Every rank runs it for every term in the same order,
        # because `active` is a function of the plan and the plan is the same shape on each.
        self.last_term_gradients = (
            {name: ctx.all_reduce(g, reduce_op="mean") for name, g in sorted(grad_acc.items())}
            if want_grads
            else None
        )

        scalars: dict[str, float] = {"loss": total, "n_voxels": float(n_voxels)}
        for key, s in sums.items():
            n = counts[key]
            scalars[key] = s / n if n > 0 else s  # a count of 0 marks a total, not a mean
        if self._momentum is not None:
            scalars["teacher_momentum"] = float(self._momentum[ctx.step])
        return StepOutput(scalars=scalars, n_samples=batch.batch_size, loss=loss)

    def term_gradients(
        self, batch: Batch, *, step: int = 0, params: list[Tensor] | None = None
    ) -> dict[str, Tensor]:
        """Per-term gradient of this batch's loss with respect to `params`, one flat vector
        each.

        The offline counterpart of `TermGrad`, and an optional hook like `inference_step`:
        `wcfm/eval/gradients.py` reaches it with `getattr` and never imports this module. It
        answers whether two terms are pulling the backbone in the same direction.

        It changes no training behaviour and is never called from `training_step`. It runs
        single-process, under `wcfm eval extract`, where there is no DDP reducer for
        `torch.autograd.grad` to disturb; `TermGrad` is the in-loop counterpart and reduces its
        vectors in `training_step` instead.

        Gradients are taken against the backbone by default because that is the shared thing
        terms compete over. A term's own head is not contested, so a cosine involving it would
        measure nothing.
        """
        params = list(self.backbone.parameters()) if params is None else list(params)
        ctx = _inference_context(self, step, batch.voxels.coordinate_tensor.device)

        if self.normalize is not None:
            self.normalize(batch.voxels)
        plan = self.augment(batch)

        teacher: list[TeacherOutput] | None = None
        if self.teacher_backbone is not None:
            needs = [n for n, t in self.terms.items() if t.requires_teacher]
            with torch.no_grad():
                teacher = [
                    self(v.clean, None, self.required_taps, teacher=True, terms=needs)
                    for v in plan.globals
                ]
        for term in self.terms.values():
            term.begin_step(plan, teacher)

        out: dict[str, Tensor] = {}
        for k in range(plan.n_views):
            active = [t for t in self.terms.values() if t.runs_on(plan, k)]
            if not active:
                continue
            view = plan.views[k]
            inject = Injection.merge(t.injection_request(view) for t in active)
            names = [t.name for t in active]
            bundle, heads = self(view.voxels, inject, self.required_taps, terms=names)
            for term in active:
                res = term.compute(bundle, heads[term.name], k, plan, teacher, ctx)
                contrib = term.weight(step) * res.loss / term.total_contrib(plan)
                # `retain_graph` because the next term shares this view's forward; `create_graph`
                # stays False -- these are read, never differentiated through.
                grads = torch.autograd.grad(
                    contrib, params, retain_graph=True, allow_unused=True
                )
                flat = torch.cat(
                    [
                        (g if g is not None else torch.zeros_like(p)).reshape(-1)
                        for g, p in zip(grads, params, strict=True)
                    ]
                )
                # Summed across views, which is what the optimizer would have seen: a term
                # scored on three views contributes three times and its gradient is their sum.
                out[term.name] = flat if term.name not in out else out[term.name] + flat
        return {k: v.detach() for k, v in out.items()}

    def on_step_end(self, ctx: StepContext) -> None:
        if self.teacher_backbone is not None:
            m = float(self._momentum[ctx.step])
            _ema(self.backbone, self.teacher_backbone, m)
            for term in self.terms.values():
                for s, t in term.ema_pairs():
                    _ema(s, t, m)
        for term in self.terms.values():
            term.on_step_end(ctx)

    def param_groups(self) -> list[dict]:
        groups = [{"name": "backbone", "params": list(self.backbone.parameters())}]
        for term in self.terms.values():
            groups.extend(term.param_groups())
        return groups

    def observables(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        if self._last_student is not None:
            out["student/out"] = self._last_student.out.feature_tensor.detach().float().clone()
            for tap, vox in self._last_student.taps.items():
                out[f"student/{tap}"] = vox.feature_tensor.detach().float().clone()
        if self._last_teacher is not None:
            out["teacher/out"] = self._last_teacher.out.feature_tensor.detach().float().clone()
        for term in self.terms.values():
            out.update(term.observables())
        return out

    # ---------------------------------------------------------------- optional hooks

    def inference_sources(self) -> tuple[str, ...]:
        """Which branches offline extraction can score. `teacher` only if one was trained."""
        return ("student", "teacher") if self.teacher_backbone is not None else ("student",)

    def inference_step(
        self, voxels: Voxels, sources: Iterable[str], taps: Iterable[str] = ()
    ) -> dict[str, FeatureBundle]:
        """One clean image, every requested branch: `{source: FeatureBundle}`.

        An optional hook, the same shape as `grad_taxonomy` and `provenance`, found with
        `getattr` by a framework that must not import this package. It is what lets
        `wcfm/eval/extract.py` score a checkpoint without ever naming `backbone` or
        `teacher_backbone`: the words for the branches are model vocabulary and they stay on
        this side of the boundary.

        Both branches come back from one call because the charge transform is applied in place
        and must happen once. A per-branch callable would either transform the same voxels
        twice -- log of a log, a silently wrong feature at every pixel -- or make its caller
        responsible for a step it is not supposed to know about. Taking the sources together
        removes the choice, and the backbone runs twice while the loader, the collate and the
        host-to-device copy run once.

        No injection and no term heads. Extraction reads what the backbone makes of a clean
        image; a masked view would measure the augmentation instead, and a head output is a
        term's business, not a feature.
        """
        sources = tuple(sources)
        available = self.inference_sources()
        unknown = [s for s in sources if s not in available]
        if unknown:
            raise ValueError(
                f"this run has no {unknown} branch to extract; it has {list(available)}. A "
                "checkpoint trained with model/teacher=none holds no teacher weights, and "
                "extracting one would return features from initialisation."
            )
        if self.normalize is not None:
            self.normalize(voxels)  # in place, once, before either forward
        out: dict[str, FeatureBundle] = {}
        for source in sources:
            bundle, _ = self.forward(voxels, None, taps, teacher=source == "teacher")
            out[source] = bundle
        return out

    def grad_taxonomy(self) -> dict[str, tuple[str, ...]]:
        taxonomy = {
            group: tuple(f"backbone.{p}" for p in prefixes)
            for group, prefixes in self.backbone.GRAD_GROUPS.items()
        }
        for name, term in self.terms.items():
            taxonomy.update(term.grad_groups(f"terms.{name}."))
        return taxonomy

    def provenance(self) -> dict:
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())

        return {
            "module": type(self).__name__,
            "backbone": type(self.backbone).__name__,
            "backbone_params": n(self.backbone),
            "terms": {name: type(t).__name__ for name, t in self.terms.items()},
            "term_params": {name: n(t) for name, t in self.terms.items()},
            "teacher": repr(self.teacher_cfg) if self.teacher_cfg else None,
            "augment": repr(self.augment),
            "normalize": repr(self.normalize) if self.normalize else None,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher_backbone is not None:
            self.teacher_backbone.eval()
        for term in self.terms.values():
            term.train(mode)  # terms keep their own teacher halves in eval
        return self

    def _build_momentum(self, ctx: StepContext) -> None:
        """The teacher momentum cosine over `total_iters`, which is `epochs * len(loader)` and
        so arrives through `ctx.extra` rather than the config."""
        from wcfm.engine.optim import CosineScheduler

        assert self.teacher_cfg is not None
        total = int(ctx.extra.get("total_iters", 0)) or 1
        self._momentum = CosineScheduler(
            self.teacher_cfg.momentum_start,
            self.teacher_cfg.momentum_end,
            epochs=1,
            steps_per_epoch=total,
        )
