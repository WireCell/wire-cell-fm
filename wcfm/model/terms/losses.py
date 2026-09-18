"""The loss of each term: DINO per pixel, charge reconstruction, occupancy, and distillation.

`PixelDINOLoss` creates its centring buffer eagerly, at construction, from the feature
dimension the term knows once it has built its head. A buffer created lazily on first use does
not exist when the setup-time broadcast runs and never reaches a checkpoint, so a resumed run
restarts centring in silence and the loss curve recovers with nothing saying why. A second
buffer, `center_initialized`, makes the first update copy the batch mean rather than decay
towards it from zero.

`update_center` all-reduces a sum and a count rather than averaging per rank, and tests for an
empty batch after the collectives, so every rank returns together or not at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class DinoLossOutput:
    loss: Tensor
    teacher_entropy: float
    student_entropy: float
    kl: float
    cov_penalty: float | None = None
    var_penalty: float | None = None
    loss_masked: float | None = None
    loss_unmasked: float | None = None


def _img_mean(loss_px: Tensor, batch_idx: Tensor, B: int) -> Tensor | None:
    """Per-image means via scatter, then the mean over non-empty images."""
    if loss_px.numel() == 0:
        return None
    per_img = torch.zeros(B, device=loss_px.device, dtype=loss_px.dtype)
    per_img.scatter_add_(0, batch_idx, loss_px)
    cnt = torch.zeros(B, device=loss_px.device, dtype=loss_px.dtype)
    cnt.scatter_add_(0, batch_idx, torch.ones_like(loss_px))
    per_img = per_img / cnt.clamp(min=1.0)
    valid = cnt > 0
    return per_img[valid].mean() if valid.any() else None


class PixelDINOLoss(nn.Module):
    """Cross-entropy between `softmax(teacher / tau_t)` and `log_softmax(student / tau_s)` per
    matched pixel, with the D-dim feature vector treated as logits, reduced per image and then
    over the batch."""

    def __init__(
        self,
        dim: int,
        *,
        center_momentum: float = 0.9,
        use_centering: bool = True,
        teacher_temp: float = 1.0,
        student_temp: float = 1.0,
        normalize_features: bool = True,
        use_cov_penalty: bool = False,
        cov_penalty_weight: float = 1e-3,
        use_var_penalty: bool = False,
        var_penalty_weight: float = 1.0,
        var_gamma: float = 1.0,
    ):
        super().__init__()
        self.center_momentum = float(center_momentum)
        self.use_centering = bool(use_centering)
        self.teacher_temp = float(teacher_temp)
        self.student_temp = float(student_temp)
        self.normalize_features = bool(normalize_features)
        self.use_cov_penalty = bool(use_cov_penalty)
        self.cov_penalty_weight = float(cov_penalty_weight)
        self.use_var_penalty = bool(use_var_penalty)
        self.var_penalty_weight = float(var_penalty_weight)
        self.var_gamma = float(var_gamma)
        self.register_buffer("center", torch.zeros(int(dim)))
        self.register_buffer("center_initialized", torch.tensor(False))

    def forward(
        self,
        s: Tensor,
        s_backbone: Tensor,
        t: Tensor,
        counts: Tensor,
        is_masked: Tensor | None = None,
    ) -> DinoLossOutput:
        B = counts.shape[0]
        device = s.device
        if self.use_centering:
            t = t - self.center.to(t.dtype)
        if self.normalize_features:
            s = F.normalize(s, dim=-1)
            t = F.normalize(t, dim=-1)

        t_prob = F.softmax(t / self.teacher_temp, dim=-1)
        s_logp = F.log_softmax(s / self.student_temp, dim=-1)
        t_logp = F.log_softmax(t / self.teacher_temp, dim=-1)
        loss = -(t_prob * s_logp).sum(dim=-1)  # H(P_t, P_s)
        s_prob = F.softmax(s / self.student_temp, dim=-1)
        t_ent_px = -(t_prob * t_logp).sum(dim=-1)
        s_ent_px = -(s_prob * s_logp).sum(dim=-1)
        kl_px = loss - t_ent_px

        cov = self._cov_penalty(s_backbone) if self.use_cov_penalty else None
        var = self._var_penalty(s_backbone) if self.use_var_penalty else None

        batch_idx = torch.repeat_interleave(torch.arange(B, device=device), counts.to(device))
        scalar = _img_mean(loss, batch_idx, B)
        if scalar is None:  # nothing matched: keep the graph attached, contribute zero
            scalar = s.sum() * 0.0
        if cov is not None:
            scalar = scalar + self.cov_penalty_weight * cov
        if var is not None:
            scalar = scalar + self.var_penalty_weight * var

        def _f(x: Tensor | None) -> float:
            return float(x.item()) if x is not None else float("nan")

        out = DinoLossOutput(
            loss=scalar,
            teacher_entropy=_f(_img_mean(t_ent_px, batch_idx, B)),
            student_entropy=_f(_img_mean(s_ent_px, batch_idx, B)),
            kl=_f(_img_mean(kl_px, batch_idx, B)),
            cov_penalty=float(cov.item()) if cov is not None else None,
            var_penalty=float(var.item()) if var is not None else None,
        )
        if is_masked is not None:
            with torch.no_grad():
                unmasked = ~is_masked
                m = _img_mean(loss[is_masked], batch_idx[is_masked], B)
                um = _img_mean(loss[unmasked], batch_idx[unmasked], B)
                out.loss_masked = float(m.item()) if m is not None else None
                out.loss_unmasked = float(um.item()) if um is not None else None
        return out

    @staticmethod
    def _cov_penalty(s: Tensor) -> Tensor:
        """VICReg-style: squared off-diagonal covariance, over D."""
        N, D = s.shape
        if N < 2:
            return s.new_tensor(0.0)
        z = s - s.mean(dim=0)
        C = (z.T @ z) / (N - 1)
        return (C.pow(2).sum() - C.diagonal().pow(2).sum()) / D

    def _var_penalty(self, s: Tensor) -> Tensor:
        """VICReg-style hinge keeping per-dimension std above `var_gamma`."""
        N, _D = s.shape
        if N < 2:
            return s.new_tensor(0.0)
        std = torch.sqrt(s.var(dim=0) + 1e-4)
        return torch.mean(torch.clamp(self.var_gamma - std, min=0.0))

    @torch.no_grad()
    def update_center(self, teacher_feats: Tensor) -> None:
        """EMA of the teacher features at active positions, over the whole step.

        A sum and a count, so under DDP the centre is the mean over the step rather than one
        mean per rank. The empty check comes after the collectives: an early return on a
        locally empty batch would leave the other ranks in an all-reduce that never comes.
        """
        distributed = dist.is_available() and dist.is_initialized()
        feat_sum = teacher_feats.sum(dim=0).to(self.center.dtype)
        count = torch.tensor([teacher_feats.shape[0]], device=feat_sum.device, dtype=feat_sum.dtype)
        if distributed:
            dist.all_reduce(feat_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        if count.item() == 0:
            return
        batch_mean = feat_sum / count
        if not bool(self.center_initialized):
            self.center.copy_(batch_mean)
            self.center_initialized.fill_(True)
        else:
            self.center.mul_(self.center_momentum).add_(
                batch_mean, alpha=1.0 - self.center_momentum
            )


def two_stage_mean(per_voxel: Tensor, counts: Tensor) -> Tensor:
    """Mean within each image, then over images -- `PixelDINOLoss`'s reduction, so a busy image
    cannot set the scale for the batch. Stays attached to the graph even when empty."""
    if per_voxel.numel() == 0:
        return per_voxel.sum()
    B = counts.shape[0]
    device = per_voxel.device
    batch_idx = torch.repeat_interleave(torch.arange(B, device=device), counts.to(device))
    per_img = torch.zeros(B, device=device, dtype=per_voxel.dtype)
    per_img.scatter_add_(0, batch_idx, per_voxel)
    counts_f = counts.to(device=device, dtype=per_voxel.dtype)
    per_img = per_img / counts_f.clamp(min=1.0)
    valid = counts_f > 0
    if not bool(valid.any()):
        return per_voxel.sum() * 0.0
    return per_img[valid].mean()


def charge_loss(pred: Tensor, target: Tensor, counts: Tensor) -> Tensor:
    """L1 on the charge masking removed, normalised per image. Both sides are already in the
    normaliser's log space, so no further transform belongs here."""
    if pred.numel() == 0:
        return pred.sum()
    return two_stage_mean((pred - target).abs(), counts)


def occupancy_loss(
    logits: Tensor, target: Tensor, counts: Tensor, *, alpha: float = 0.25, gamma: float = 2.0
) -> Tensor:
    """Focal binary cross-entropy over occupancy candidates, normalised per image.

    The candidate set is mostly empty, so a plain BCE converges to "predict empty everywhere"
    and stays there. `gamma` discounts examples the model already gets right and `alpha`
    reweights the positive class. Both are exposed rather than fixed at their standard values
    because the positive rate depends on how the candidate set was built, and a term that
    cannot be retuned would hide that.

    `two_stage_mean` is per image then over images, as everywhere else, so an image with more
    candidates does not weigh more.
    """
    if logits.numel() == 0:
        return logits.sum()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)  # probability of the true class
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return two_stage_mean(alpha_t * (1.0 - p_t).pow(gamma) * bce, counts)


def distill_loss(pred: Tensor, target: Tensor, counts: Tensor) -> Tensor:
    """Cosine distance to a teacher's features, per image then over images.

    Direction only: the teacher's feature magnitudes need not be commensurate with the
    projected student's, so nothing here depends on the two having been trained to the same
    scale. `F.cosine_similarity` normalises both sides itself.
    """
    if pred.numel() == 0:
        return pred.sum()
    return two_stage_mean(1.0 - F.cosine_similarity(pred, target, dim=-1), counts)
