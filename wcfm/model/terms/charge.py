"""The charge term: regress the (log-space) charge at the pixels masking removed.

Supervised entirely by the input, with no teacher. It requests injection of the `masked` role
at every skip so the decoder emits a feature at each removed coordinate, reads its 1x1 head
there (inside the wrapped forward, through `head_forward`), and takes an L1 against the
masker's `masked_feats`, which are already normalised because the transform runs before the
masker.

The gather is `on_miss="raise"`: every masked coordinate is injected at the full-resolution
skip and the decoder's output geometry is that skip's, so a miss is a bug in the backbone
rather than a condition to absorb.

A charge head and a DINO term have never trained together, so this combination has no
reference loss curve to be judged against and is gated on probe numbers instead.
"""

from __future__ import annotations

from typing import Any, ClassVar

from torch import Tensor
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import SparseConv2d

from ..augment.views import Augment, View, ViewPlan
from ..backbones.base import Backbone, FeatureBundle, Injection, InjectionGroup
from .base import TeacherOutput, Term, TermOutput
from .gather import gather_at_coords
from .losses import charge_loss


class ChargeTerm(Term):
    requires_masking: ClassVar[bool] = True

    def __init__(self, *, weight: float = 0.1):
        super().__init__(weight=weight)
        self.head: SparseConv2d | None = None
        self._inject_taps: tuple[str, ...] = ()
        self._last_pred: Tensor | None = None

    def build(self, backbone: Backbone) -> None:
        self.head = SparseConv2d(backbone.out_dim, 1, kernel_size=1, bias=True)
        # Inject where it reads. The head runs on `bundle.out`, whose geometry is the
        # full-resolution skip's, so a stride-1 site is what makes `on_miss="raise"` hold.
        # Requesting the same coordinates at a coarser skip as well lands them on the grid an
        # occupancy term enumerates its candidates on -- and since those candidates cover every
        # cell the masker wiped, every projected coordinate is also a candidate, so the
        # backbone refuses the step for a coordinate carrying two roles.
        self._inject_taps = tuple(
            t for t in backbone.INJECT_TAPS if backbone.TAP_STRIDE.get(t, 1) == 1
        )

    @property
    def inject_roles(self) -> tuple[str, ...]:
        return ("masked",)

    def validate(self, backbone: Backbone, augment: Augment, has_teacher: bool) -> None:
        if augment.masker is None:
            raise ValueError(
                f"term {self.name!r} regresses the charge masking removed and no masker is "
                "configured; select an augment with a masker, or remove the term"
            )
        if not self._inject_taps:
            sites = {t: backbone.TAP_STRIDE.get(t) for t in backbone.INJECT_TAPS}
            raise ValueError(
                f"term {self.name!r} regresses the charge at full resolution and needs a "
                f"stride-1 injection site; {type(backbone).__name__} injects at {sites}"
            )
        if not backbone.supports_injection or "masked" not in backbone.inject_roles:
            raise ValueError(
                f"term {self.name!r} needs the backbone to inject at the masked coordinates, and "
                f"{type(backbone).__name__} has no 'masked' token (inject_roles="
                f"{list(backbone.inject_roles)}); select model/backbone=attn_mae"
            )

    def head_forward(self, bundle: FeatureBundle) -> Voxels:
        assert self.head is not None, "build() was not called"
        return self.head(bundle.out)

    def injection_request(self, view: View) -> Injection | None:
        if view.mask is None:
            return None
        return Injection(
            [
                InjectionGroup(tap, view.mask.masked_coords, "masked", 1)
                for tap in self._inject_taps
            ]
        )

    def compute(
        self,
        bundle: FeatureBundle,
        head_out: Voxels,
        view_idx: int,
        plan: ViewPlan,
        teacher: list[TeacherOutput] | None,
        ctx: Any,
    ) -> TermOutput:
        view = plan.views[view_idx]
        assert view.mask is not None, "validate() requires a masker"
        feats = view.mask.masked_feats
        if feats and feats[0].shape[-1] != 1:
            raise ValueError(
                f"charge target has {feats[0].shape[-1]} channels; the charge term regresses "
                "the single charge channel"
            )
        pred, target, counts, _ = gather_at_coords(
            head_out, view.mask.masked_coords, feats, on_miss="raise"
        )
        loss = charge_loss(pred, target, counts)
        self._last_pred = pred.detach()
        return TermOutput(loss=loss, scalars={"loss_charge": float(loss.detach())})
