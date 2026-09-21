"""The pure-torch point-set operations agree with brute-force references.

The reference for `cnms` is the sequential greedy pass the compiled extension performs; the
production version runs in rounds, and this file is what pins the two to the same retained set.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from wcfm.model.backbones.polarmae.ops import (  # noqa: E402
    ball_query,
    chamfer_distance,
    cnms,
    knn_points,
    sample_farthest_points,
)


def cloud(B=3, P=60, seed=0, lengths=(60, 41, 7)):
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(B, P, 3, generator=g)
    pts[..., 2] = 0.0
    return pts, torch.tensor(lengths)


def test_knn_matches_brute_force():
    p1, l1 = cloud(seed=1)
    p2, l2 = cloud(seed=2, lengths=(60, 3, 30))
    dists, idx = knn_points(p1, p2, lengths1=l1, lengths2=l2, K=4)
    for b in range(3):
        d = torch.cdist(p1[b, : l1[b]], p2[b, : l2[b]]) ** 2
        k = min(4, int(l2[b]))
        ref_d, ref_i = d.topk(k, dim=1, largest=False)
        assert torch.allclose(dists[b, : l1[b], :k], ref_d, atol=1e-5)
        assert torch.equal(idx[b, : l1[b], :k], ref_i)
        if k < 4:  # a short row pads with inf and index 0
            assert torch.isinf(dists[b, : l1[b], k:]).all()
            assert (idx[b, : l1[b], k:] == 0).all()
        assert (idx[b, l1[b] :] == 0).all()


def test_ball_query_returns_the_first_k_members_in_index_order():
    p1, l1 = cloud(seed=3)
    idx = ball_query(p1, p1, K=5, radius=0.2, lengths1=l1, lengths2=l1)
    for b in range(3):
        d = torch.cdist(p1[b, : l1[b]], p1[b, : l1[b]])
        for i in range(int(l1[b])):
            members = torch.nonzero(d[i] <= 0.2).flatten()[:5]
            got = idx[b, i]
            assert torch.equal(got[: len(members)], members)
            assert (got[len(members) :] == -1).all()
        assert (idx[b, l1[b] :] == -1).all()


def test_fps_matches_a_sequential_reference():
    pts, lengths = cloud(seed=4, lengths=(60, 10, 3))
    idx = sample_farthest_points(pts, lengths, K=6)
    for b in range(3):
        n = int(lengths[b])
        sel = [0]
        closest = torch.full((n,), float("inf"))
        for _ in range(1, min(6, n)):
            d = ((pts[b, :n] - pts[b, sel[-1]]) ** 2).sum(-1)
            closest = torch.minimum(closest, d)
            sel.append(int(closest.argmax()))
        assert idx[b, : len(sel)].tolist() == sel
        assert (idx[b, len(sel) :] == -1).all()


def sequential_cnms(centroids, radius, overlap, K, lengths):
    """The compiled extension's algorithm, one candidate at a time."""
    idx = ball_query(
        centroids, centroids, K=K, radius=2 * radius * overlap, lengths1=lengths, lengths2=lengths
    )
    counts = idx.ge(0).sum(-1)
    _, order = counts.sort(dim=-1, descending=True, stable=True)
    B, P = counts.shape
    retain = torch.zeros(B, P, dtype=torch.bool)
    for b in range(B):
        suppressed = torch.zeros(P, dtype=torch.bool)
        for i in order[b].tolist():
            if i >= int(lengths[b]) or suppressed[i]:
                continue
            retain[b, i] = True
            nb = idx[b, i]
            suppressed[nb[nb >= 0]] = True
    return retain


@pytest.mark.parametrize("seed,K,radius", [(5, 64, 0.08), (6, 4, 0.15), (7, 200, 0.3)])
def test_cnms_matches_the_sequential_greedy_pass(seed, K, radius):
    """Including a neighbour list truncated at a small `K`, where the suppression relation is
    not symmetric and a centroid can be missing from its own list."""
    pts, lengths = cloud(B=4, P=80, seed=seed, lengths=(80, 55, 12, 0))
    centres, n = cnms(pts, radius=radius, overlap_factor=0.6, K=K, lengths=lengths)
    ref = sequential_cnms(pts, radius, 0.6, K, lengths)
    assert n.tolist() == ref.sum(1).tolist()
    for b in range(4):
        kept = pts[b][ref[b]]  # ascending index order, as the reorder keeps it
        assert torch.equal(centres[b, : n[b]], kept)


def test_chamfer_matches_brute_force_and_is_differentiable():
    g = torch.Generator().manual_seed(8)
    x = torch.rand(3, 7, 4, generator=g).requires_grad_(True)
    y = torch.rand(3, 9, 4, generator=g)
    lx = torch.tensor([7, 4, 1])
    ly = torch.tensor([9, 2, 5])
    loss = chamfer_distance(x, y, lx, ly)
    ref = 0.0
    for b in range(3):
        d = torch.cdist(x[b, : lx[b]], y[b, : ly[b]]) ** 2
        ref = ref + d.min(1).values.mean() + d.min(0).values.mean()
    assert torch.allclose(loss, ref / 3, atol=1e-5)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert (x.grad[1, 4:] == 0).all(), "padded points take no gradient"
