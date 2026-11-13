"""`sinkhorn_knopp` and its use as `PixelDINOLoss`' teacher target."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from wcfm.model.terms import losses  # noqa: E402
from wcfm.model.terms.losses import PixelDINOLoss, sinkhorn_knopp  # noqa: E402


def skewed_logits(n=500, k=16) -> torch.Tensor:
    torch.manual_seed(0)
    t = torch.nn.functional.normalize(torch.randn(n, k), dim=-1)
    t[:, 0] += 0.5  # one prototype favoured by every pixel
    return torch.nn.functional.normalize(t, dim=-1)


def test_rows_are_distributions_and_the_prototypes_even_out_with_iterations():
    t = skewed_logits()
    usage = torch.softmax(t / 0.07, dim=-1).mean(dim=0)  # 1/K each once balanced
    spread = [float(usage.max() / usage.min())]
    for n_iters in (1, 3, 10):
        q = sinkhorn_knopp(t, 0.07, n_iters)
        torch.testing.assert_close(q.sum(dim=1), torch.ones(len(t)))
        assert (q >= 0).all()
        usage = q.mean(dim=0)
        spread.append(float(usage.max() / usage.min()))
    assert spread[0] > 2.0
    assert spread[1] < spread[0] and spread[2] < spread[1] and spread[3] < spread[2]
    assert spread[3] < 1.05


def test_a_constant_shift_of_the_logits_changes_nothing():
    t = skewed_logits()
    torch.testing.assert_close(sinkhorn_knopp(t + 3.0, 0.07, 3), sinkhorn_knopp(t, 0.07, 3))


def test_no_pixels_gives_no_rows():
    assert sinkhorn_knopp(torch.zeros(0, 16), 0.07, 3).shape == (0, 16)


def test_the_loss_scores_the_student_against_the_balanced_teacher():
    t = skewed_logits()
    s = skewed_logits().flip(0).requires_grad_(True)
    counts = torch.tensor([200, 300])
    loss = PixelDINOLoss(
        16,
        use_centering=False,
        sinkhorn_iters=3,
        teacher_temp=0.07,
        student_temp=0.1,
        normalize_features=False,
    )
    out = loss(s, s, t, counts)
    q = sinkhorn_knopp(t, 0.07, 3)
    per_px = -(q * torch.log_softmax(s / 0.1, dim=-1)).sum(dim=-1)
    expected = (per_px[:200].mean() + per_px[200:].mean()) / 2
    torch.testing.assert_close(out.loss, expected)
    out.loss.backward()
    assert s.grad is not None


def test_sinkhorn_with_centring_is_refused():
    with pytest.raises(ValueError, match="use_centering"):
        PixelDINOLoss(16, use_centering=True, sinkhorn_iters=3)


def on_one_rank(monkeypatch, t: torch.Tensor) -> torch.Tensor:
    """`sinkhorn_knopp` over all of `t` as one process would run it, collectives off."""
    with monkeypatch.context() as m:
        m.setattr(losses, "_all_reduce", lambda x, op="sum": x)
        return sinkhorn_knopp(t, 0.07, 3)


@pytest.mark.distributed
@pytest.mark.parametrize("split", [200, 500], ids=["uneven", "rank1-empty"])
def test_two_ranks_balance_the_prototypes_over_both_ranks_pixels(
    two_rank_fabric, monkeypatch, split
):
    t = skewed_logits().to(two_rank_fabric.device)  # NCCL takes CUDA tensors only
    mine = t[:split] if two_rank_fabric.global_rank == 0 else t[split:]
    expected = on_one_rank(monkeypatch, t)
    expected = expected[:split] if two_rank_fabric.global_rank == 0 else expected[split:]
    torch.testing.assert_close(sinkhorn_knopp(mine, 0.07, 3), expected)
