"""Coordinate matching: ``match_and_gather`` against a per-image reference, exactly; and
``gather_at_coords`` with the ``on_miss`` contract that closes the ``occ_coords`` hole.

The reference is the straightforward per-image intersection the vectorised version replaced.
Both *select* rows of the same tensors, so the bar is ``torch.equal``, not ``allclose``. The
old repo ran this on a GPU for the timing; the correctness half runs here on a CPU.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402
from warpconvnet.geometry.coords.integer import IntCoords  # noqa: E402
from warpconvnet.geometry.features.cat import CatFeatures  # noqa: E402
from warpconvnet.geometry.types.voxels import Voxels  # noqa: E402

from wcfm.model.terms import gather_at_coords, match_and_gather  # noqa: E402

pytestmark = pytest.mark.stack

IMAGE_H, IMAGE_W = 150, 105


def _reference(s_out, s_backbone, t_out, masked=None):
    B = len(s_out.offsets) - 1
    device = s_out.feature_tensor.device
    W = 1
    for v in (s_out, t_out):
        if v.coordinate_tensor.shape[0] > 0:
            W = max(W, int(v.coordinate_tensor[:, 0].max()) + 1)
    s_idx_all, t_idx_all, counts, tags = [], [], [], [] if masked is not None else None
    for b in range(B):
        ss, se = int(s_out.offsets[b]), int(s_out.offsets[b + 1])
        ts, te = int(t_out.offsets[b]), int(t_out.offsets[b + 1])
        sc, tc = s_out.coordinate_tensor[ss:se], t_out.coordinate_tensor[ts:te]
        if sc.shape[0] == 0 or tc.shape[0] == 0:
            counts.append(0)
            continue
        sk = sc[:, 1].long() * W + sc[:, 0].long()
        tk = tc[:, 1].long() * W + tc[:, 0].long()
        t_sorted, t_order = tk.sort()
        pos = torch.searchsorted(t_sorted, sk).clamp(max=t_sorted.shape[0] - 1)
        valid = t_sorted[pos] == sk
        s_local = valid.nonzero(as_tuple=False).squeeze(1)
        if s_local.numel() == 0:
            counts.append(0)
            continue
        s_idx_all.append(s_local + ss)
        t_idx_all.append(t_order[pos[valid]] + ts)
        counts.append(s_local.shape[0])
        if tags is not None:
            m = masked[b]
            if m.shape[0] > 0:
                mk = m[:, 1].long() * W + m[:, 0].long()
                tags.append(torch.isin(sk[s_local], mk))
            else:
                tags.append(torch.zeros(s_local.shape[0], dtype=torch.bool, device=device))
    counts = torch.tensor(counts, dtype=torch.int64, device=device)
    if s_idx_all:
        s_idx, t_idx = torch.cat(s_idx_all), torch.cat(t_idx_all)
        out = (
            s_out.feature_tensor[s_idx],
            s_backbone.feature_tensor[s_idx],
            t_out.feature_tensor[t_idx],
        )
    else:
        out = (
            s_out.feature_tensor.new_zeros(0, s_out.feature_tensor.shape[1]),
            s_backbone.feature_tensor.new_zeros(0, s_backbone.feature_tensor.shape[1]),
            t_out.feature_tensor.new_zeros(0, t_out.feature_tensor.shape[1]),
        )
    return (*out, counts, torch.cat(tags) if tags else None)


def _voxels(coords_per_image, feats_per_image):
    counts = torch.tensor([c.shape[0] for c in coords_per_image], dtype=torch.int64)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    coords = (
        torch.cat(coords_per_image) if coords_per_image else torch.zeros(0, 2, dtype=torch.int32)
    )
    feats = torch.cat(feats_per_image)
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


def _case(sizes, overlap=0.6, d_head=16, d_bb=8, seed=0, masked_frac=0.5):
    g = torch.Generator().manual_seed(seed)
    s_c, t_c, s_f, s_bb, t_f, masked = [], [], [], [], [], []
    for n_s, n_t in sizes:
        pool = torch.randperm(IMAGE_H * IMAGE_W, generator=g)[: n_s + n_t]
        s_flat = pool[:n_s]
        n_shared = int(overlap * min(n_s, n_t))
        t_flat = torch.cat([s_flat[:n_shared], pool[n_s : n_s + n_t - n_shared]])
        to_xy = lambda f: torch.stack([f % IMAGE_W, f // IMAGE_W], dim=1).int()  # noqa: E731
        s_c.append(to_xy(s_flat))
        t_c.append(to_xy(t_flat))
        s_f.append(torch.randn(n_s, d_head, generator=g))
        s_bb.append(torch.randn(n_s, d_bb, generator=g))
        t_f.append(torch.randn(t_flat.shape[0], d_head, generator=g))
        masked.append(to_xy(s_flat[: int(masked_frac * n_s)]))
    return _voxels(s_c, s_f), _voxels(s_c, s_bb), _voxels(t_c, t_f), masked


def _assert_same(fast, ref):
    for name, a, b in zip(("s", "s_bb", "t", "counts", "is_masked"), fast, ref, strict=True):
        if a is None or b is None:
            assert a is None and b is None, f"{name} is None in only one version"
            continue
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert torch.equal(a, b), f"{name} differs"


@pytest.mark.parametrize(
    "sizes,kw",
    [
        ([(90, 110), (150, 140), (70, 70), (200, 180)], {}),
        ([(0, 80), (90, 0), (0, 0), (120, 100)], {}),
        ([(60, 60), (80, 80)], {"overlap": 0.0}),
        ([(50, 0), (70, 0)], {}),
        ([(100, 100)], {}),
        ([(80, 80), (90, 90)], {"masked_frac": 0.0}),
    ],
    ids=["ragged", "empty_images", "no_overlap", "teacher_empty", "B1", "no_masked"],
)
def test_match_and_gather_equals_the_per_image_reference(sizes, kw):
    s, s_bb, t, m = _case(sizes, **kw)
    _assert_same(match_and_gather(s, s_bb, t, m), _reference(s, s_bb, t, m))
    _assert_same(match_and_gather(s, s_bb, t), _reference(s, s_bb, t))


def test_masked_tags_are_all_false_when_nothing_was_masked():
    s, s_bb, t, m = _case([(80, 80)], masked_frac=0.0)
    out = match_and_gather(s, s_bb, t, m)
    assert out[4] is not None and not out[4].any()


# ------------------------------------------------------------------ gather_at_coords


def make_vox(coords, values=None):
    coords = torch.as_tensor(coords, dtype=torch.int32)
    offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int64)
    feats = values if values is not None else torch.zeros(coords.shape[0], 1)
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


def test_gather_matches_by_coordinate_not_row():
    vox = make_vox([[1, 1], [2, 2], [3, 3]], torch.tensor([[10.0], [20.0], [30.0]]))
    pred, target, counts, missed = gather_at_coords(
        vox, [torch.tensor([[3, 3], [1, 1]])], [torch.tensor([300.0, 100.0])]
    )
    assert pred.tolist() == [30.0, 10.0] and target.tolist() == [300.0, 100.0]
    assert counts.tolist() == [2] and missed == 0


def test_a_missing_prediction_raises_by_default():
    """The old drop is what let the occupancy term score a rim-biased sliver for a campaign."""
    vox = make_vox([[1, 1]], torch.tensor([[10.0]]))
    with pytest.raises(RuntimeError, match="no prediction"):
        gather_at_coords(vox, [torch.tensor([[9, 9], [1, 1]])], [torch.tensor([999.0, 100.0])])


def test_on_miss_drop_keeps_labels_on_their_own_coordinates_and_counts_the_drop():
    """The pairing bug that trains happily on wrong answers: a dropped request must drop its
    target too, and the caller must be told."""
    vox = make_vox([[3, 3], [1, 1]], torch.tensor([[33.0], [11.0]]))
    cands = torch.tensor([[1, 1], [2, 2], [3, 3]])
    labels = torch.tensor([1.0, 0.0, 1.0])
    pred, target, counts, missed = gather_at_coords(vox, [cands], [labels], on_miss="drop")
    assert missed == 1 and int(counts[0]) == 2
    assert {(round(float(p)), float(t)) for p, t in zip(pred, target, strict=True)} == {
        (11, 1.0),
        (33, 1.0),
    }


def test_gather_empty_request_and_empty_output():
    vox = make_vox([[1, 1]], torch.tensor([[10.0]]))
    pred, tgt, counts, missed = gather_at_coords(vox, [torch.zeros(0, 2, dtype=torch.int64)], None)
    assert pred.numel() == 0 and tgt is None and counts.tolist() == [0] and missed == 0
    empty = make_vox(torch.zeros(0, 2), torch.zeros(0, 1))
    with pytest.raises(RuntimeError, match="no rows"):
        gather_at_coords(empty, [torch.tensor([[1, 1]])])
    pred, _, _, missed = gather_at_coords(empty, [torch.tensor([[1, 1]])], on_miss="drop")
    assert pred.numel() == 0 and missed == 1
