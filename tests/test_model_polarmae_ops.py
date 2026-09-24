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
    grid_ball_query,
    knn_points,
    sample_farthest_points,
)
from wcfm.model.backbones.polarmae.tokenizer import PointcloudGrouping  # noqa: E402

PITCH = 1.0 / 600.0
SHARD = "/gpfs01/lbne/users/fm/cffm-data/shards_fhdh_sparse_200k_mixed_apa0W/shard_00000.h5"


def lattice_cloud(B=3, P=500, seed=0, lengths=(500, 320, 9), span=60):
    """Distinct integer pixels in a `span x span` window, sorted by `(channel, tick)`, scaled
    by `PITCH`, third coordinate zero. Dense enough that radius-5 balls hold tens of points."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.zeros(B, P, 3)
    for b in range(B):
        pix = torch.unique(torch.randint(0, span, (P * 3, 2), generator=g), dim=0)[:P]
        pts[b, : pix.shape[0], :2] = (pix.float() + 200.0) * PITCH
    return pts, torch.tensor(lengths)


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


def test_queries_ignore_autocast():
    """Pixel clouds at the production scale, `1/600` per pixel, under bf16 autocast: every
    query has to return what it returns in fp32."""
    g = torch.Generator().manual_seed(3)
    pix = torch.randint(0, 1000, (2, 400, 2), generator=g).float()
    pts = torch.zeros(2, 400, 3)
    pts[..., :2] = (pix - 500.0) / 600.0
    lengths = torch.tensor([400, 250])
    radius = 5.0 / 600.0
    idx32 = ball_query(pts, pts, K=64, radius=radius, lengths1=lengths, lengths2=lengths)
    d32, k32 = knn_points(pts, pts, lengths1=lengths, lengths2=lengths, K=5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        idx_ac = ball_query(pts, pts, K=64, radius=radius, lengths1=lengths, lengths2=lengths)
        d_ac, k_ac = knn_points(pts, pts, lengths1=lengths, lengths2=lengths, K=5)
    assert torch.equal(idx_ac, idx32)
    assert torch.equal(k_ac, k32)
    assert d_ac.dtype == torch.float32 and torch.equal(d_ac, d32)


@pytest.mark.parametrize("K", [4, 64, 256])
@pytest.mark.parametrize("radius_px", [2.5, 5.0, 7.0])
def test_grid_ball_query_matches_ball_query(K, radius_px):
    pts, lengths = lattice_cloud()
    radius = radius_px * PITCH
    ref = ball_query(pts, pts, K=K, radius=radius, lengths1=lengths, lengths2=lengths)
    got = grid_ball_query(
        pts, pts, K=K, radius=radius, pitch=PITCH, lengths1=lengths, lengths2=lengths
    )
    assert torch.equal(got, ref)
    assert (ref[0].ge(0).sum(1) > 1).any(), "the lattice is too sparse to test anything"
    # Queries from a subset of the cloud, as the member query runs from the centres.
    sub = pts[:, ::3]
    l1 = (lengths + 2) // 3
    ref = ball_query(sub, pts, K=K, radius=radius, lengths1=l1, lengths2=lengths)
    got = grid_ball_query(sub, pts, K=K, radius=radius, pitch=PITCH, lengths1=l1, lengths2=lengths)
    assert torch.equal(got, ref)


def test_grid_ball_query_agrees_on_the_radius_boundary():
    """Offsets `(3, 4)` and `(5, 0)` sit exactly on a radius-5 ball. Both queries must decide
    them the same way, which is pytorch3d's: the fp32 square of the radius, strict `<`."""
    pix = torch.tensor([[0, 0], [3, 4], [5, 0], [4, 3], [0, 5], [1, 1], [4, 4], [0, 6]])
    pts = torch.zeros(1, pix.shape[0], 3)
    pts[0, :, :2] = (pix.float() + 400.0) * PITCH
    lengths = torch.tensor([pix.shape[0]])
    for radius_px in (5.0, 5.0 + 1e-3, 5.0 - 1e-3):
        radius = radius_px * PITCH
        ref = ball_query(pts, pts, K=8, radius=radius, lengths1=lengths, lengths2=lengths)
        got = grid_ball_query(
            pts, pts, K=8, radius=radius, pitch=PITCH, lengths1=lengths, lengths2=lengths
        )
        assert torch.equal(got, ref), radius_px


def test_grid_ball_query_refuses_shared_sites():
    pts = torch.zeros(1, 3, 3)
    pts[0, :, :2] = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 1.0]]) * PITCH
    lengths = torch.tensor([3])
    with pytest.raises(ValueError, match="lattice site"):
        grid_ball_query(pts, pts, K=4, radius=PITCH, pitch=PITCH, lengths1=lengths)
    # The same site in the padded tail is not a point and does not count.
    two = torch.tensor([2])
    grid_ball_query(pts, pts, K=4, radius=PITCH, pitch=PITCH, lengths1=two, lengths2=two)


def test_cnms_and_grouping_with_pitch_match_without():
    pts, lengths = lattice_cloud(seed=4)
    radius = 5.0 * PITCH
    a = cnms(pts, radius=radius, overlap_factor=0.5, K=256, lengths=lengths)
    b = cnms(pts, radius=radius, overlap_factor=0.5, K=256, lengths=lengths, pitch=PITCH)
    assert torch.equal(a[1], b[1]) and torch.equal(a[0], b[0])
    points = torch.cat([pts, torch.rand(pts.shape[0], pts.shape[1], 1)], dim=-1)
    kw = dict(
        num_groups=256,
        group_max_points=32,
        group_radius=radius,
        group_upscale_points=256,
        overlap_factor=0.5,
        context_length=2048,
    )
    dense = PointcloudGrouping(**kw)(points, lengths)
    grid = PointcloudGrouping(**kw, pitch=PITCH)(points, lengths)
    for name in ("groups", "centers", "emb_mask", "point_mask", "idx"):
        assert torch.equal(getattr(dense, name), getattr(grid, name)), name


@pytest.mark.needs_data
def test_grid_queries_match_on_production_events():
    """The two queries on the first eight events of a production shard, on the tokenizer's
    footing: cnms neighbours at radius 5 px and group members at radius 5 px."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(SHARD) as f:
        coords = torch.from_numpy(f["coords"][()]).float()
        off = torch.from_numpy(f["offsets"][()]).long()
    B = 8
    counts = off[1 : B + 1] - off[:B]
    pts = torch.zeros(B, int(counts.max()), 3)
    for b in range(B):
        pts[b, : counts[b], :2] = (
            coords[off[b] : off[b + 1]] - torch.tensor([480.0, 563.0])
        ) * PITCH
    radius = 5.0 * PITCH
    ref = ball_query(pts, pts, K=256, radius=radius, lengths1=counts, lengths2=counts)
    got = grid_ball_query(
        pts, pts, K=256, radius=radius, pitch=PITCH, lengths1=counts, lengths2=counts
    )
    assert torch.equal(got, ref)
    centres, n = cnms(pts, radius=radius, overlap_factor=0.5, K=256, lengths=counts, pitch=PITCH)
    ref_c, ref_n = cnms(pts, radius=radius, overlap_factor=0.5, K=256, lengths=counts)
    assert torch.equal(n, ref_n) and torch.equal(centres, ref_c)
    ref = ball_query(centres, pts, K=256, radius=radius, lengths1=n, lengths2=counts)
    got = grid_ball_query(
        centres, pts, K=256, radius=radius, pitch=PITCH, lengths1=n, lengths2=counts
    )
    assert torch.equal(got, ref)


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
