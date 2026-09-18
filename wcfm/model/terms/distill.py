"""Distillation: match a frozen teacher's per-voxel features, cosine, at the coordinates both
sides hold.

The teacher is a `wcfm` checkpoint. `load_module` rebuilds the module that wrote it from the
file's own `cfg.model`, so its architecture, its feature width and the data convention it was
trained under all come out of the file and none of them is typed into a config. The term keeps
one branch's backbone, frozen, as a submodule, which is what gets it device placement, rank-0
broadcast and a self-contained checkpoint.

`conf/model/kd.yaml` runs this term alone with no cropper and no masker: the whole image goes
in on both sides and every pixel is scored. The join is by coordinate, so under a masker the
term scores only the pixels the student's output carries -- the teacher still runs on the whole
input and the extra rows do not match. `distill_matched` is where that shows up.

Two things a caller has to know:

- Do not reach the teacher through `load_backbone`. Its closure runs `SslModule.inference_step`,
  which normalises charge in place, and `SslModule.training_step` has already normalised the
  batch in place before the augment. The log transform would be applied twice, which is a
  plausible number at every pixel and no error anywhere.
- A teacher whose own `cfg.model` carries a `distill` term loads its own teacher when
  `load_module` instantiates it, and so on down the chain.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.mlp import Linear

from ..augment.transforms import FeatureLogTransform
from ..augment.views import ViewPlan
from ..backbones.base import Backbone, FeatureBundle
from ..checkpoint import inference_sources, load_module
from .base import TeacherOutput, Term, TermOutput
from .gather import match_and_gather
from .losses import distill_loss


class DistillTerm(Term):
    """Regress the student's projected features onto a frozen checkpoint's.

    A distillation run carries this term alone.

    `requires_teacher` stays False: it names the EMA twin `model/teacher=ema` builds, which this
    term neither needs nor uses, so `model/teacher=none` remains valid.
    """

    requires_teacher: ClassVar[bool] = False
    requires_masking: ClassVar[bool] = False

    def __init__(
        self, *, checkpoint: str = "", source: str = "student", weight: float = 1.0
    ) -> None:
        super().__init__(weight=weight)
        if not checkpoint:
            raise ValueError(
                "distill needs a teacher: set model.terms.<key>.checkpoint to a wcfm "
                "checkpoint written by an earlier run"
            )
        self.checkpoint = str(checkpoint)
        self.source = str(source)

        try:
            module, ckpt = load_module(self.checkpoint)
        except ModuleNotFoundError as exc:
            # `cfg.model` names classes that are not in this tree: a checkpoint written by
            # another codebase. `load_module`'s own errors (no `cfg.model`, weights that do not
            # match it) already name the file, so only this one needs saying.
            raise ModuleNotFoundError(
                f"{self.checkpoint} records an architecture this tree cannot import ({exc}); "
                "distill reads a wcfm checkpoint, whose `cfg.model` is instantiated to rebuild "
                "the teacher"
            ) from exc
        available = inference_sources(module)
        if self.source not in available:
            raise ValueError(
                f"{self.checkpoint} has no {self.source!r} branch; it has {list(available)}. A "
                "run trained with model/teacher=none holds no teacher weights, and distilling "
                "from one would regress onto initialisation."
            )
        self.teacher: Backbone = (
            module.teacher_backbone if self.source == "teacher" else module.backbone
        )
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()
        # Kept for `validate_normalize`, which compares the charge convention this teacher
        # was trained under against the run's. The rest of the rebuilt module is dropped here:
        # its terms' heads and its EMA twin would ride in every checkpoint this run writes for
        # nothing.
        self.teacher_cfg: dict = dict(ckpt.cfg or {})

        self.head: Linear | None = None
        self._teacher_out: list[Voxels] = []

    # -------------------------------------------------------------- construction

    def build(self, backbone: Backbone) -> None:
        # Eager, like every other head: one built on first use receives no gradient on the
        # steps it does not run and is decayed by the optimizer anyway. Both widths are
        # attributes, so a mismatch is a construction-time error rather than a first-forward one.
        self.head = Linear(backbone.out_dim, self.teacher.out_dim, bias=True)

    def validate_normalize(self, normalize: FeatureLogTransform | None) -> None:
        """Refuse a teacher trained on a different charge scale.

        A teacher trained where `min_val`/`max_val` differ is handed inputs outside the
        distribution it learned and answers confidently anyway. Nothing downstream can see
        that, which is why it is refused rather than warned about. The teacher's own constants
        come from `cfg.model.normalize` in its checkpoint, so this is decidable rather than a
        caveat in a docstring.

        Found by `SslModule.validate` with `getattr`, because `Term.validate` is not handed the
        transform. A checkpoint recording no transform says nothing to disagree with.
        """
        theirs = ((self.teacher_cfg.get("model") or {}).get("normalize")) or {}
        if normalize is None or not theirs:
            return
        for key, ours in (("min_val", normalize.min_val), ("max_val", normalize.max_val)):
            other = theirs.get(key)
            if other is not None and float(other) != float(ours):
                raise ValueError(
                    f"term {self.name!r} distils from {self.checkpoint}, trained with "
                    f"normalize.{key}={other}, while this run uses {ours}. The charge "
                    f"transform would hand the teacher inputs it never saw: match "
                    f"data.feat_{key}, or pick a teacher trained on this production."
                )

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    # ------------------------------------------------------ inside the wrapped forward

    def head_forward(self, bundle: FeatureBundle) -> Voxels:
        assert self.head is not None, "build() was not called"
        return self.head(bundle.out)

    # ------------------------------------------------------------------- the step

    def begin_step(self, plan: ViewPlan, teacher: list[TeacherOutput] | None) -> None:
        # `self.teacher`, not `ctx.module`: the rule on StepContext.module covers every forward
        # whose gradients matter, and a frozen no_grad forward arms no reducer. `.clean` is the
        # crop before masking, which is what the teacher was trained to see.
        with torch.no_grad():
            # One entry per view, in view order, so `compute` indexes with `view_idx`.
            self._teacher_out = [self.teacher(v.clean, None, ()).out for v in plan.views]

    def compute(
        self,
        bundle: FeatureBundle,
        head_out: Voxels,
        view_idx: int,
        plan: ViewPlan,
        teacher: list[TeacherOutput] | None,
        ctx: Any,
    ) -> TermOutput:
        assert self._teacher_out, "begin_step() was not called"
        t_out = self._teacher_out[view_idx]
        s, _s_bb, t, counts, _is_masked = match_and_gather(head_out, bundle.out, t_out)
        loss = distill_loss(s, t.to(s.dtype), counts)
        matched = int(counts.sum().item())
        return TermOutput(
            loss=loss,
            scalars={"loss_distill": float(loss.detach()), "distill_matched": float(matched)},
            # 0 marks a total rather than a mean: a matched count averaged over views is not a
            # count of anything.
            counts={"distill_matched": 0},
        )

    def on_step_end(self, ctx: Any) -> None:
        self._teacher_out = []
