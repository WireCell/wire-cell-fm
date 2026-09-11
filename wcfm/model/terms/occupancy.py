"""The occupancy term: at each candidate coordinate, is there a voxel there or not?

Supervised entirely by the input. The masker enumerates candidates and labels each one
positive exactly when it coincides with a voxel masking removed (`label_candidates` in
`augment/masking.py`); the term asks the backbone to inject a `candidate` token at those
coordinates, reads a 1x1 head on the half-resolution decoder output, and takes a focal BCE.

It reads `dec_half` rather than `out`. `READ_TAP` and `READ_STRIDE` together ask one question
per 2x2 block of full-resolution pixels, and the candidate coordinates are in those units.
That resolution is a property of the question rather than of the backbone, so the term owns
it.

The role matters. DINO's placeholders and occupancy's candidates are two different questions
-- "predict the feature of a pixel I removed" and "is anything here at all" -- so the request
carries `role="candidate"` and the backbone holds a token of its own for it. Tokens indexed by
resolution instead of by purpose would give both questions the same learned vector. A
coordinate requested under both roles at one tap is refused by the backbone rather than
silently resolved.

The gather is `on_miss="drop"` here and `"raise"` in the charge term. The charge term injects
at full resolution and reads the full-resolution output, so every request must come back and a
miss is a bug. Occupancy's candidates are deduped against the skip: one that coincides with a
surviving voxel is already in the geometry and is not re-injected, and after cropping some can
fall outside the crop entirely. Those are conditions rather than bugs, and they are not silent
either -- the term intersects its labelled candidates against what the backbone reports it
placed, and counts every survivor it had to drop into `occ_dropped`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.sparse_conv import SparseConv2d

from ..augment.views import Augment, View, ViewPlan
from ..backbones.base import Backbone, FeatureBundle, Injection, InjectionGroup
from ..backbones.blocks import ResidualSparseBlock2D
from .base import TeacherOutput, Term, TermOutput
from .gather import gather_at_coords
from .losses import occupancy_loss


class OccupancyTerm(Term):
    requires_masking: ClassVar[bool] = True

    #: The decoder resolution the question is asked at.
    READ_TAP: ClassVar[str] = "dec_half"
    READ_STRIDE: ClassVar[int] = 2
    #: Candidates are injected at the skip of the same stride, so the decoder emits a feature
    #: at each of them by the time `READ_TAP` is produced.
    INJECT_TAP: ClassVar[str] = "enc1"

    def __init__(self, *, weight: float = 1.0, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__(weight=weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.block: ResidualSparseBlock2D | None = None
        self.head: SparseConv2d | None = None
        self._dropped = 0

    # -------------------------------------------------------------- construction

    def build(self, backbone: Backbone) -> None:
        dim = backbone.tap_dim(self.READ_TAP)
        self.block = ResidualSparseBlock2D(dim, dim, kernel_size=3)
        self.head = SparseConv2d(dim, 1, kernel_size=1, bias=True)

    @property
    def inject_roles(self) -> tuple[str, ...]:
        return ("candidate",)

    def validate(self, backbone: Backbone, augment: Augment, has_teacher: bool) -> None:
        masker = augment.masker
        if masker is None:
            raise ValueError(
                f"term {self.name!r} asks whether a masked coordinate was occupied and no "
                "masker is configured; select an augment with a masker, or remove the term"
            )
        if not getattr(masker, "builds_candidates", False):
            raise ValueError(
                f"term {self.name!r} needs the masker to enumerate occupancy candidates, and "
                f"{type(masker).__name__} produces none. Select model/masker=region with "
                "build_candidates: true. The block and pixel maskers have no candidate source: "
                "only the region masker knows the geometry of what it removed well enough to "
                "enumerate one."
            )
        if self.READ_TAP not in backbone.TAPS:
            raise ValueError(
                f"term {self.name!r} reads tap {self.READ_TAP!r} and "
                f"{type(backbone).__name__} has {list(backbone.TAPS)}"
            )
        stride = backbone.TAP_STRIDE.get(self.READ_TAP)
        if stride != self.READ_STRIDE:
            raise ValueError(
                f"term {self.name!r} enumerates candidates in units of stride "
                f"{self.READ_STRIDE} but {type(backbone).__name__} produces tap "
                f"{self.READ_TAP!r} at stride {stride}; the labels and the predictions would "
                "be at different resolutions"
            )
        if not backbone.supports_injection or "candidate" not in backbone.inject_roles:
            raise ValueError(
                f"term {self.name!r} needs a 'candidate' token on the backbone, and "
                f"{type(backbone).__name__} has inject_roles="
                f"{list(backbone.inject_roles)}; add 'candidate' to model.backbone.inject_roles"
            )
        # Both messages name the fix rather than restating the rule: each of these
        # mismatches is expensive to debug from the loss curve alone.
        if getattr(masker, "flavor", "wipe") != "wipe":
            raise ValueError(
                f"term {self.name!r} labels a candidate positive when it coincides with a "
                "removed voxel, which is only a well-posed question if the cell was wiped; "
                f"masker flavor is {getattr(masker, 'flavor', None)!r}, set flavor: wipe"
            )
        if not getattr(masker, "neg_per_pos", None) and not getattr(masker, "max_neg", None):
            raise ValueError(
                f"term {self.name!r} runs on a candidate set that is overwhelmingly empty and "
                "no negatives cap is set; set model.augment.masker.neg_per_pos or .max_neg, "
                "choosing the cap against the positive rate this masker and grid produce"
            )

    # ------------------------------------------------------ inside the wrapped forward

    def head_forward(self, bundle: FeatureBundle) -> Voxels:
        assert self.block is not None and self.head is not None, "build() was not called"
        tap = bundle.taps.get(self.READ_TAP)
        if tap is None:
            raise KeyError(
                f"term {self.name!r} reads tap {self.READ_TAP!r} and the bundle carries "
                f"{sorted(bundle.taps)}; SslModule must request it"
            )
        return self.head(self.block(tap))

    def wants_taps(self) -> tuple[str, ...]:
        return (self.READ_TAP,)

    # -------------------------------------------------------------------- per step

    def injection_request(self, view: View) -> Injection | None:
        if view.mask is None or view.mask.cand_coords is None:
            return None
        return Injection(
            [
                InjectionGroup(
                    self.INJECT_TAP, view.mask.cand_coords, "candidate", self.READ_STRIDE
                )
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
        coords, targets = view.mask.cand_coords, view.mask.occ_targets
        if coords is None or targets is None:
            raise ValueError(
                f"term {self.name!r} ran on a view whose masker produced no candidates; "
                "validate() should have refused this configuration"
            )
        coords, targets, n_unplaced = self._intersect_with_placed(bundle, coords, targets)
        pred, target, counts, n_missed = gather_at_coords(
            head_out, coords, targets, on_miss="drop"
        )
        assert target is not None
        loss = occupancy_loss(pred, target, counts, alpha=self.alpha, gamma=self.gamma)
        self._dropped = n_unplaced + n_missed
        n_pos = int(target.sum().item()) if target.numel() else 0
        return TermOutput(
            loss=loss,
            scalars={
                "loss_occ": float(loss.detach()),
                "occ_candidates": float(target.numel()),
                "occ_positive_rate": (n_pos / target.numel()) if target.numel() else 0.0,
                "occ_dropped": float(self._dropped),
            },
        )

    def _intersect_with_placed(
        self, bundle: FeatureBundle, coords: list[Tensor], targets: list[Tensor]
    ) -> tuple[list[Tensor], list[Tensor], int]:
        """Keep only the candidates the backbone reports it actually injected.

        A candidate that already coincides with a surviving voxel is deduped against the skip
        and never injected; it is still a legitimate question, but the backbone did not place
        a token for it, so scoring it would be scoring an ordinary feature as though it were a
        prediction. Every candidate dropped here is counted into `occ_dropped`.
        """
        if bundle.injected is None:
            # The backbone does not report what it placed, so there is nothing to intersect
            # against and dropping would be guessing. Backbones that inject must report.
            return coords, targets, 0
        placed: list[Tensor] | None = None
        for g in bundle.injected.groups:
            if g.tap == self.INJECT_TAP and g.role == "candidate":
                placed = g.coords
                break
        if placed is None:
            # It reported, and it placed no `candidate` token anywhere, not even an empty
            # group, so every candidate is unplaced. Returning them unchanged would score
            # ordinary features as predictions: a well-formed scalar computed from the wrong
            # rows, which nothing downstream can tell from a real one.
            placed = []
        kept_c: list[Tensor] = []
        kept_t: list[Tensor] = []
        dropped = 0
        for b, (c, tgt) in enumerate(zip(coords, targets, strict=True)):
            p = placed[b] if b < len(placed) else c.new_zeros(0, c.shape[-1])
            if c.numel() == 0:
                kept_c.append(c)
                kept_t.append(tgt)
                continue
            keep = _rows_in(c, p)
            dropped += int((~keep).sum().item())
            kept_c.append(c[keep])
            kept_t.append(tgt[keep])
        return kept_c, kept_t, dropped

    # ---------------------------------------------------------------- bookkeeping

    def observables(self) -> dict[str, Tensor]:
        return {}


def _rows_in(rows: Tensor, other: Tensor) -> Tensor:
    """Boolean mask over `rows`: which of them appear in `other`. Both `[N, 2]` ints.

    Keyed on a single integer per coordinate rather than a pairwise compare, so this stays
    linear in the candidate count -- a region-masked image enumerates tens of thousands.
    """
    if other.numel() == 0:
        return torch.zeros(rows.shape[0], dtype=torch.bool, device=rows.device)
    span = int(max(int(rows[:, 0].max()), int(other[:, 0].max()))) + 1
    key_r = rows[:, 1].long() * span + rows[:, 0].long()
    key_o = other[:, 1].long() * span + other[:, 0].long()
    return torch.isin(key_r, key_o)
