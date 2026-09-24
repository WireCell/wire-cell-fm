"""What a term is: one question asked of the student's output, with its own head and its own
loss. Two shapes share the bookkeeping in `TermBase`:

- `Term` is asked of a `FeatureBundle` on a pixel view and runs under `SslModule`.
- `GroupTerm` is asked of the masked tokens of a point cloud and runs under `PointMaeModule`.

A term never sees the loop. The module owns the schedule: it decides which terms run, unions
their requests, weights and sums their losses, and takes the step's single backward.

Everything with trainable parameters runs inside the wrapped forward. A term's head is applied
by `head_forward`, which the module's `forward` calls, so the head's parameters are reachable
from the forward's outputs, which is what DDP's unused-parameter pass traverses. A head applied
outside that forward is marked unused at the end of it and then receives a gradient in the
backward, and DDP raises "Expected to mark a variable ready only once". One forward computes
everything the loss will differentiate, and `compute` -- the loss itself -- is parameter-free.
`tests/test_engine_distributed.py` pins the neighbouring rule on two ranks: a view computed on
`self` rather than `ctx.module` arms nothing, and the ranks diverge in silence.

Heads are built eagerly, in `build(backbone)`, never lazily on first use: a head that exists
but is never called receives zero gradients under DDP and is then decayed by the optimizer, so
its checkpointed value depends on history nothing records. Everything a term owns is in
`state_dict` from step 0, buffers included.

Weights under `SslModule`: `total = sum_t weight_t * (loss_t / total_contrib_t)`.
`total_contrib` is the number of contributions the term makes over the whole plan -- `n_pairs`
for DINO, the number of views a primary-scope term runs on -- so the sum over views is the mean
over that term's contributions. Dividing by it is what makes the DINO sum a `/n_pairs` mean and
the charge contribution `lambda_charge * charge`, and it leaves `lambda_dino` explicit rather
than an implied 1. A `GroupTerm` contributes once per step, so its weight multiplies its loss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from torch import Tensor, nn

if TYPE_CHECKING:  # pragma: no cover
    from ..augment.views import Augment, View, ViewPlan
    from ..backbones.base import Backbone, FeatureBundle, Injection
    from ..backbones.polarmae import MaskedTokens, PolarMAEBackbone

# One teacher forward: the teacher backbone's bundle and each term's teacher-head output.
TeacherOutput = tuple["FeatureBundle", dict[str, Any]]


@dataclass
class TermOutput:
    loss: Tensor
    """Attached to the graph; the module weights, divides and sums it."""
    scalars: dict[str, float] = field(default_factory=dict)
    """Sums over this view's contributions, so the module can average exactly."""
    counts: dict[str, int] = field(default_factory=dict)
    """How many contributions each scalar sums over; absent means one; 0 marks a total."""


class TermBase(nn.Module, ABC):
    """Weight, name and parameter bookkeeping shared by `Term` and `GroupTerm`."""

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.loss_weight = float(weight)
        self.name = type(self).__name__.removesuffix("Term").lower()

    def weight(self, step: int) -> float:
        return self.loss_weight

    def on_step_end(self, ctx: Any) -> None:
        """After the optimizer step: centring, per-term schedules."""

    def param_groups(self) -> list[dict]:
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"name": f"{self.name}_head", "params": params}] if params else []

    def grad_groups(self, prefix: str) -> dict[str, tuple[str, ...]]:
        """Taxonomy entries for this term's trainable parameters, under `prefix`."""
        if not any(p.requires_grad for p in self.parameters()):
            return {}
        return {f"{self.name}_head": (prefix,)}

    def observables(self) -> dict[str, Tensor]:
        return {}


class Term(TermBase):
    """A question asked of a pixel view's `FeatureBundle`."""

    requires_teacher: ClassVar[bool] = False
    requires_masking: ClassVar[bool] = False

    # -------------------------------------------------------------- construction

    @abstractmethod
    def build(self, backbone: Backbone) -> None:
        """Construct heads and buffers against the backbone. Called once, before validation."""

    def validate(self, backbone: Backbone, augment: Augment, has_teacher: bool) -> None:
        """Refuse a configuration that would train the wrong thing. Messages name the fix."""

    @property
    def inject_roles(self) -> tuple[str, ...]:
        """The roles this term's `injection_request` uses, for validation up front."""
        return ()

    # ------------------------------------------------------ inside the wrapped forward

    def head_forward(self, bundle: FeatureBundle) -> Any:
        """What this term computes from its trainable parameters. Runs inside
        `SslModule.forward`, so it is armed, autocast and reduced like the backbone."""
        return None

    def teacher_head_forward(self, bundle: FeatureBundle) -> Any:
        """The teacher-side counterpart, on the frozen twin. Runs under `no_grad`."""
        return None

    # -------------------------------------------------------------------- per step

    def begin_step(self, plan: ViewPlan, teacher: list[TeacherOutput] | None) -> None:
        """Per-step preparation, before any student view runs."""

    def runs_on(self, plan: ViewPlan, view_idx: int) -> bool:
        return True

    def total_contrib(self, plan: ViewPlan) -> int:
        return sum(1 for k in range(plan.n_views) if self.runs_on(plan, k))

    def injection_request(self, view: View) -> Injection | None:
        return None

    def wants_taps(self) -> tuple[str, ...]:
        """Named intermediates this term reads. The module unions these into every forward,
        so a term reading a tap does not depend on the metrics config having asked for it."""
        return ()

    @abstractmethod
    def compute(
        self,
        bundle: FeatureBundle,
        head_out: Any,
        view_idx: int,
        plan: ViewPlan,
        teacher: list[TeacherOutput] | None,
        ctx: Any,
    ) -> TermOutput:
        """The loss. Parameter-free: everything differentiable was computed in the forward."""

    # ---------------------------------------------------------------- bookkeeping

    def ema_pairs(self) -> list[tuple[nn.Module, nn.Module]]:
        """`(student, teacher)` module pairs the module's EMA must cover."""
        return []


class GroupTerm(TermBase):
    """A question asked of the masked tokens of one point cloud.

    `head_forward` receives the `MaskedTokens` of the step and runs inside
    `PointMaeModule.forward`; `compute` scores its output against the groups the tokens stand
    for, and is parameter-free.
    """

    @abstractmethod
    def build(self, backbone: PolarMAEBackbone) -> None:
        """Construct heads against the backbone's `out_dim` and group size. Called once."""

    def validate(self, backbone: PolarMAEBackbone) -> None:
        """Refuse a configuration that would train the wrong thing. Messages name the fix."""

    def head_forward(self, tokens: MaskedTokens) -> Any:
        return None

    @abstractmethod
    def compute(self, tokens: MaskedTokens, head_out: Any, ctx: Any) -> TermOutput:
        """The loss. Parameter-free: everything differentiable was computed in the forward."""
