"""`PointMaeModule`: the `TrainingModule` for the Point-MAE family, on a `PolarMAEBackbone`.

Every word in `training_step` is model vocabulary:

    normalise(batch)                                   # here, in place
    points, lengths = backbone.points_from(voxels)     # (B, N, 4), thinned if configured
    tokens, heads   = ctx.module(points, lengths, terms=..)   # ONE forward: tokenize, mask,
                                                        # encode, decode, every term's head
    loss = sum over terms of t.weight(step) * t.compute(tokens, heads[t]).loss
    return StepOutput(..., loss=loss)                  # the ENGINE backwards it

`forward` is the reduced unit: it tokenises, hides `mask_ratio` of each event's tokens, runs
the encoder over the visible ones and the decoder over all with the mask token in the hidden
slots, then every named term's head. One forward per step carries every trainable parameter
the loss differentiates, which is what DDP's reducer needs; `wcfm.model.modules.ssl` says why.

`max_points` and `charge_threshold` thin the cloud before tokenisation: pixels at or below the
threshold in raw ADC are dropped, then each event is cut to a uniform random subset of
`max_points`, order preserved. Both are off by default. They change what the backbone is
trained on, so a checkpoint records them in `cfg.model`; extraction runs on whole events.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from warpconvnet.geometry.types.voxels import Voxels

from wcfm.data.voxels import Batch
from wcfm.engine.protocol import StepContext, StepOutput

from ..augment.transforms import FeatureLogTransform
from ..backbones.base import FeatureBundle
from ..backbones.polarmae import MaskedTokens, PolarMAEBackbone, random_token_mask
from ..terms.base import GroupTerm


class PointMaeModule(nn.Module):
    """`PolarMAEBackbone` + a `ModuleDict` of `GroupTerm`s + the charge transform."""

    def __init__(
        self,
        *,
        backbone: PolarMAEBackbone,
        terms: Mapping[str, GroupTerm],
        normalize: FeatureLogTransform | None = None,
        mask_ratio: float = 0.6,
        max_points: int | None = None,
        charge_threshold: float | None = None,
    ):
        super().__init__()
        if not terms:
            raise ValueError("a model needs at least one term; `model.terms` is empty")
        if not isinstance(backbone, PolarMAEBackbone):
            raise TypeError(
                f"PointMaeModule tokenises, masks and decodes through a PolarMAEBackbone; "
                f"{type(backbone).__name__} has no tokens to mask. Select model/backbone=polarmae"
            )
        if not 0.0 <= float(mask_ratio) < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
        self.backbone = backbone
        self.normalize = normalize
        self.mask_ratio = float(mask_ratio)
        self.max_points = None if max_points is None else int(max_points)
        self.charge_threshold = None if charge_threshold is None else float(charge_threshold)
        self.terms = nn.ModuleDict()
        for name, term in terms.items():
            if not isinstance(term, GroupTerm):
                raise TypeError(
                    f"model.terms.{name} is {type(term).__name__}, not a GroupTerm; the pixel "
                    "terms run under model/module=ssl"
                )
            term.name = str(name)
            term.build(backbone)
            term.validate(backbone)
            self.terms[str(name)] = term
        self._last: MaskedTokens | None = None

    # ---------------------------------------------------------------- the reduced unit

    def forward(
        self, points: Tensor, lengths: Tensor, *, terms: Iterable[str] = ()
    ) -> tuple[MaskedTokens, dict[str, Any]]:
        """Tokenise, mask, encode, decode, then every named term's head."""
        bundle = self.backbone.tokenize(points, lengths)
        masked, visible = random_token_mask(bundle.lengths, bundle.tokens.shape[1], self.mask_ratio)
        encoded = self.backbone.encode(bundle, visible)
        decoded = self.backbone.decode(bundle, encoded, visible, masked)
        tokens = MaskedTokens(bundle, masked, visible, encoded, decoded)
        heads = {n: self.terms[n].head_forward(tokens) for n in terms}
        return tokens, heads

    # ------------------------------------------------------------------ thinning

    def thin(self, points: Tensor, lengths: Tensor, keep: Tensor) -> tuple[Tensor, Tensor]:
        """Drop the slots where `keep` is False and, with `max_points`, a uniform random part
        of the rest, then compact every event to the front with its order preserved."""
        B, N, _ = points.shape
        device = points.device
        valid = torch.arange(N, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        keep = keep & valid
        if self.max_points is not None:
            scores = torch.rand(B, N, device=device).masked_fill(~keep, 2.0)
            rank = torch.empty_like(scores, dtype=torch.int64)
            rank.scatter_(1, scores.argsort(1), torch.arange(N, device=device).expand(B, N))
            keep &= rank < self.max_points
        order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
        points = points.gather(1, order.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
        lengths = keep.sum(1)
        return points[:, : max(int(lengths.max()), 1)], lengths

    # --------------------------------------------------------------- TrainingModule

    def training_step(self, batch: Batch, ctx: StepContext) -> StepOutput:
        raw = (
            batch.voxels.feature_tensor[:, 0].clone() if self.charge_threshold is not None else None
        )
        if self.normalize is not None:
            self.normalize(batch.voxels)  # in place
        points, lengths, (event, slot) = self.backbone.points_from(batch.voxels)
        n_voxels = int(points.shape[0] and lengths.sum())
        if raw is not None or self.max_points is not None:
            keep = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
            if raw is not None:
                keep[event, slot] = raw > self.charge_threshold
            points, lengths = self.thin(points, lengths, keep)

        names = tuple(self.terms)
        tokens, heads = ctx.module(points, lengths, terms=names)
        self._last = tokens

        loss: Tensor | None = None
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for name in names:
            term = self.terms[name]
            out = term.compute(tokens, heads[name], ctx)
            contrib = term.weight(ctx.step) * out.loss
            loss = contrib if loss is None else loss + contrib
            for key, value in out.scalars.items():
                sums[key] = sums.get(key, 0.0) + value
                counts[key] = counts.get(key, 0) + out.counts.get(key, 1)
        assert loss is not None

        scalars: dict[str, float] = {
            "loss": float(loss.detach()),
            "n_voxels": float(n_voxels),
            "n_points": float(lengths.sum()),
            "n_tokens": float(tokens.bundle.lengths.sum()),
            "n_masked": float(tokens.masked.sum()),
        }
        for key, s in sums.items():
            n = counts[key]
            scalars[key] = s / n if n > 0 else s
        return StepOutput(scalars=scalars, n_samples=batch.batch_size, loss=loss)

    def on_step_end(self, ctx: StepContext) -> None:
        for term in self.terms.values():
            term.on_step_end(ctx)

    def param_groups(self) -> list[dict]:
        groups = [{"name": "backbone", "params": list(self.backbone.parameters())}]
        for term in self.terms.values():
            groups.extend(term.param_groups())
        return groups

    def observables(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        if self._last is not None:
            t = self._last
            out["student/tokens"] = t.encoded[t.visible].detach().float().clone()
            out["student/decoded"] = t.decoded[t.masked].detach().float().clone()
        for term in self.terms.values():
            out.update(term.observables())
        return out

    # ---------------------------------------------------------------- optional hooks

    def inference_sources(self) -> tuple[str, ...]:
        return ("student",)

    def inference_step(
        self, voxels: Voxels, sources: Iterable[str], taps: Iterable[str] = ()
    ) -> dict[str, FeatureBundle]:
        """One clean image through the backbone's own `forward`: every token visible, no
        decoder, no heads, no thinning. The charge transform is applied in place, once."""
        sources = tuple(sources)
        unknown = [s for s in sources if s != "student"]
        if unknown:
            raise ValueError(f"this run has no {unknown} branch to extract; it has ['student']")
        if self.normalize is not None:
            self.normalize(voxels)
        return {s: self.backbone(voxels, None, taps) for s in sources}

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
            "mask_ratio": self.mask_ratio,
            "max_points": self.max_points,
            "charge_threshold": self.charge_threshold,
            "normalize": repr(self.normalize) if self.normalize else None,
        }
