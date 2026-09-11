"""The DINO term: per-pixel distillation from an EMA teacher, per (student view, teacher
global) pair. `hybrid` is this term with `score_injected=True`.

Which pairs reach the loss. A same-index pair compares a view with the teacher's copy of the
same crop, so it is skipped whenever another pair exists -- unless the student received mask
tokens at the removed coordinates, in which case the pair is a masked-prediction task and is
kept. `score_injected` decides both things: whether this term requests injection, and whether
the same-index pair counts. With one global view and no injection, the global view pairs with
nothing and is not encoded at all.

Injected positions are excluded explicitly when they are not scored. Another term may inject
at the same coordinates on the same view, so `score_injected=False` filters the tagged rows
out of the loss rather than relying on their being absent from the intersection.

The projection head runs in `head_forward`, inside the wrapped forward, and `compute` receives
its output. `terms/base.py` says why that split is not optional under DDP.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor
from warpconvnet.geometry.types.voxels import Voxels

from ..augment.views import Augment, View, ViewPlan
from ..backbones.base import Backbone, FeatureBundle, Injection, InjectionGroup
from .base import TeacherOutput, Term, TermOutput
from .gather import match_and_gather
from .heads import DINOProjectionHead
from .losses import PixelDINOLoss

DEFAULT_HEAD: dict = {"hidden_dim": 256, "output_dim": 128, "n_layers": 2}
_DEFAULT = object()  # "not given": distinguishes the default head from an explicit `null`


class DinoTerm(Term):
    requires_teacher: ClassVar[bool] = True

    def __init__(
        self,
        *,
        weight: float = 1.0,
        score_injected: bool = False,
        proj_head: dict | None | object = _DEFAULT,
        center_momentum: float = 0.9,
        use_centering: bool = True,
        teacher_temp: float = 0.07,
        student_temp: float = 0.1,
        cov_penalty: dict | None = None,
        var_penalty: dict | None = None,
    ):
        super().__init__(weight=weight)
        self.score_injected = bool(score_injected)
        # `proj_head: null` in config removes the head; leaving it unset keeps the default
        # one, so a `DinoTerm()` built in a test has the production shape.
        if proj_head is _DEFAULT:
            proj_head = dict(DEFAULT_HEAD)
        self._use_head = proj_head is not None
        self._head_cfg = dict(proj_head) if proj_head is not None else {}
        cov = dict(cov_penalty or {})
        var = dict(var_penalty or {})
        self._loss_kwargs = dict(
            center_momentum=center_momentum,
            use_centering=use_centering,
            teacher_temp=teacher_temp,
            student_temp=student_temp,
            use_cov_penalty=bool(cov.get("enabled", False)),
            cov_penalty_weight=float(cov.get("weight", 1.0)),
            use_var_penalty=bool(var.get("enabled", False)),
            var_penalty_weight=float(var.get("weight", 1.0)),
            var_gamma=float(var.get("gamma", 1.0)),
        )
        self.head: DINOProjectionHead | None = None
        self.teacher_head: DINOProjectionHead | None = None
        self.loss: PixelDINOLoss | None = None
        self._inject_taps: tuple[str, ...] = ()
        self._teacher_out: list[Voxels] = []
        self._last_student_head: Tensor | None = None
        self._last_teacher_head: Tensor | None = None

    # ----------------------------------------------------------------- construction

    def build(self, backbone: Backbone) -> None:
        D = backbone.out_dim
        if self._use_head:
            cfg = self._head_cfg
            args = (D, cfg["hidden_dim"], cfg["output_dim"], cfg.get("n_layers", 4))
            self.head = DINOProjectionHead(*args)
            self.teacher_head = DINOProjectionHead(*args)
            self.teacher_head.load_state_dict(self.head.state_dict())
            for p in self.teacher_head.parameters():
                p.requires_grad_(False)
            self.teacher_head.eval()
            loss_dim = self.head.out_dim
        else:
            loss_dim = D
        # normalize_features is the negation of having a head: the head's L2 norm already
        # puts the features on the sphere.
        self.loss = PixelDINOLoss(
            loss_dim, normalize_features=not self._use_head, **self._loss_kwargs
        )
        # Inject where the pair is scored. `match_and_gather` intersects the two branches'
        # full-resolution outputs, so a stride-1 site is the one that decides whether an
        # injected position is scoreable at all. Requesting the same coordinates at a coarser
        # skip as well puts them on the grid an occupancy term enumerates its candidates on,
        # where one coordinate would carry two roles and the backbone refuses the step.
        self._inject_taps = tuple(
            t for t in backbone.INJECT_TAPS if backbone.TAP_STRIDE.get(t, 1) == 1
        )

    @property
    def inject_roles(self) -> tuple[str, ...]:
        return ("masked",) if self.score_injected else ()

    def validate(self, backbone: Backbone, augment: Augment, has_teacher: bool) -> None:
        if not has_teacher:
            raise ValueError(
                f"term {self.name!r} distils from a teacher and none is configured; select "
                "model/teacher=ema, or remove the term"
            )
        if augment.masker is None and augment.cropper is None:
            raise ValueError(
                f"term {self.name!r} with neither masking nor cropping compares each view "
                "against itself and learns nothing; enable at least one"
            )
        if self.score_injected:
            if augment.masker is None:
                raise ValueError(
                    f"term {self.name!r} has score_injected=true but no masker is configured: "
                    "with nothing masked there is nothing to reinject. Select an augment with a "
                    "masker, or set score_injected=false (plain dino)"
                )
            if not backbone.supports_injection or "masked" not in backbone.inject_roles:
                raise ValueError(
                    f"term {self.name!r} has score_injected=true but backbone "
                    f"{type(backbone).__name__} has no 'masked' token (inject_roles="
                    f"{list(backbone.inject_roles)}); select model/backbone=attn_mae, or set "
                    "score_injected=false"
                )

    # ------------------------------------------------------ inside the wrapped forward

    def head_forward(self, bundle: FeatureBundle) -> Voxels:
        return self.head(bundle.out) if self.head is not None else bundle.out

    def teacher_head_forward(self, bundle: FeatureBundle) -> Voxels:
        return self.teacher_head(bundle.out) if self.teacher_head is not None else bundle.out

    # ---------------------------------------------------------------------- per step

    def pairs_for(self, plan: ViewPlan, k: int) -> list[int]:
        """The teacher globals student view `k` is scored against."""
        keep_same_index = self.score_injected and plan.masked
        return [
            g
            for g in range(plan.n_global)
            if not (k == g and plan.n_views > 1 and not keep_same_index)
        ]

    def runs_on(self, plan: ViewPlan, view_idx: int) -> bool:
        return bool(self.pairs_for(plan, view_idx))

    def total_contrib(self, plan: ViewPlan) -> int:
        return sum(len(self.pairs_for(plan, k)) for k in range(plan.n_views))

    def injection_request(self, view: View) -> Injection | None:
        if not self.score_injected or view.mask is None:
            return None
        return Injection(
            [
                InjectionGroup(tap, view.mask.masked_coords, "masked", 1)
                for tap in self._inject_taps
            ]
        )

    def begin_step(self, plan: ViewPlan, teacher: list[TeacherOutput] | None) -> None:
        assert teacher is not None, "DinoTerm.validate requires a teacher"
        self._teacher_out = [outs[self.name] for _bundle, outs in teacher]
        if self._teacher_out:
            self._last_teacher_head = self._teacher_out[0].feature_tensor.detach()

    def compute(self, bundle, head_out, view_idx, plan, teacher, ctx) -> TermOutput:
        assert self.loss is not None, "build() was not called"
        view = plan.views[view_idx]
        s_out: Voxels = head_out
        masked = view.masked_coords

        loss_sum: Tensor | None = None
        scalars: dict[str, float] = {}
        counts: dict[str, int] = {}
        n_pairs = 0

        def add(key: str, value: float | None) -> None:
            if value is None or value != value:  # None or NaN: not measured this pair
                return
            scalars[key] = scalars.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1

        for g in self.pairs_for(plan, view_idx):
            t_out = self._teacher_out[g]
            s, s_bb, t, cnt, is_masked = match_and_gather(s_out, bundle.out, t_out, masked)
            if is_masked is not None and not self.score_injected:
                # Injected by another term: present in the output, not ours to score.
                keep = ~is_masked
                b_idx = torch.repeat_interleave(
                    torch.arange(cnt.shape[0], device=cnt.device), cnt
                )
                s, s_bb, t = s[keep], s_bb[keep], t[keep]
                cnt = torch.bincount(b_idx[keep], minlength=cnt.shape[0])
                is_masked = None
            out = self.loss(s, s_bb, t, cnt, is_masked=is_masked)
            loss_sum = out.loss if loss_sum is None else loss_sum + out.loss
            n_pairs += 1
            add("loss_dino", float(out.loss.detach()))
            add("teacher_entropy", out.teacher_entropy)
            add("student_entropy", out.student_entropy)
            add("kl", out.kl)
            add("cov_penalty", out.cov_penalty)
            add("var_penalty", out.var_penalty)
            add("loss_masked", out.loss_masked)
            add("loss_unmasked", out.loss_unmasked)

        assert loss_sum is not None and n_pairs > 0, "runs_on said yes but no pair was scored"
        scalars["n_pairs"] = float(n_pairs)
        counts["n_pairs"] = 0  # a total, not a mean: the module sums these
        self._last_student_head = s_out.feature_tensor.detach()
        return TermOutput(loss=loss_sum, scalars=scalars, counts=counts)

    def on_step_end(self, ctx: Any) -> None:
        # Gated on configuration (a teacher always exists for this term), never on whether a
        # tensor happens to be None: update_center all-reduces, so a rank-dependent skip hangs.
        assert self.loss is not None
        if self._teacher_out:
            self.loss.update_center(self._teacher_out[0].feature_tensor)
        self._teacher_out = []

    # ---------------------------------------------------------------- bookkeeping

    def ema_pairs(self):
        return [(self.head, self.teacher_head)] if self.head is not None else []

    def observables(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        if self.loss is not None:
            out[f"{self.name}/center"] = self.loss.center.detach().clone()
        if self.head is not None and self._last_student_head is not None:
            out["student/head"] = self._last_student_head.float().clone()
        if self.teacher_head is not None and self._last_teacher_head is not None:
            out["teacher/head"] = self._last_teacher_head.float().clone()
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher_head is not None:
            self.teacher_head.eval()
        return self
