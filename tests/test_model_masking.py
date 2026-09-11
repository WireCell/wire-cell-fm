"""The maskers, ported from ``test_masker_contract.py`` and ``test_region_masking.py`` in the
old repo -- all three under the same checks, since they are drop-in replacements for each
other -- plus the one thing that is new: ``keep``, the boolean that lets truth ride along.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402
from warpconvnet.geometry.coords.integer import IntCoords  # noqa: E402
from warpconvnet.geometry.features.cat import CatFeatures  # noqa: E402
from warpconvnet.geometry.types.voxels import Voxels  # noqa: E402

from wcfm.model.augment import (  # noqa: E402
    BlockMasker,
    PixelMasker,
    RegionMasker,
    cap_negatives,
)

from .fake_backbone import as_pairs, rows_of  # noqa: E402

pytestmark = pytest.mark.stack

CANVAS_W, CANVAS_H = 60, 40
CELL_W, CELL_H = 10, 10
STRIDE = 2


def make_batch(counts=(120, 0, 80), seed: int = 0) -> Voxels:
    """Scattered active pixels whose middle image is empty -- the alignment case that bites.
    The feature value encodes the coordinate so a misalignment is detectable."""
    g = torch.Generator().manual_seed(seed)
    coords, feats = [], []
    for n in counts:
        c = torch.stack(
            [
                torch.randint(0, CANVAS_W, (n,), generator=g),
                torch.randint(0, CANVAS_H, (n,), generator=g),
            ],
            dim=1,
        )
        c = torch.unique(c, dim=0)
        coords.append(c)
        feats.append((c[:, 0] * 1000 + c[:, 1]).float().unsqueeze(1))
    offsets = torch.tensor(
        [0] + list(torch.tensor([c.shape[0] for c in coords]).cumsum(0)), dtype=torch.int64
    )
    return Voxels(
        batched_coordinates=IntCoords(torch.cat(coords).to(torch.int32), offsets=offsets),
        batched_features=CatFeatures(torch.cat(feats), offsets=offsets),
        offsets=offsets,
    )


def feature_for(coord) -> float:
    return float(coord[0]) * 1000 + float(coord[1])


def maskers(**kw):
    torch.manual_seed(0)
    yield "pixel", PixelMasker(ratio=0.5)
    torch.manual_seed(0)
    yield "block", BlockMasker(ratio=0.5, win_ch=3, win_tick=3)
    torch.manual_seed(0)
    yield (
        "region",
        RegionMasker(image_w=CANVAS_W, image_h=CANVAS_H, cell_w=CELL_W, cell_h=CELL_H, **kw),
    )


def region(**kw) -> RegionMasker:
    torch.manual_seed(0)
    return RegionMasker(image_w=CANVAS_W, image_h=CANVAS_H, cell_w=CELL_W, cell_h=CELL_H, **kw)


# ------------------------------------------------------------------ the shared contract


@pytest.mark.parametrize(
    "name,masker", list(maskers()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_kept_and_masked_partition_the_input_exactly(name, masker):
    vox = make_batch()
    r = masker(vox)
    for b in range(len(vox.offsets) - 1):
        orig = as_pairs(rows_of(vox, b))
        kept = as_pairs(rows_of(r.student, b))
        gone = as_pairs(r.masked_coords[b])
        assert kept | gone == orig, f"{name} image {b}: kept+masked != original"
        assert not (kept & gone), f"{name} image {b}: pixels in both"


@pytest.mark.parametrize(
    "name,masker", list(maskers()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_masked_features_belong_to_their_coordinates_row_for_row(name, masker):
    """A target misaligned by one row trains perfectly happily and means nothing."""
    r = masker(make_batch())
    assert len(r.masked_feats) == len(r.masked_coords)
    for b, (c, f) in enumerate(zip(r.masked_coords, r.masked_feats, strict=True)):
        assert c.shape[0] == f.shape[0]
        for row in range(c.shape[0]):
            assert abs(float(f[row, 0]) - feature_for(c[row])) < 1e-6, f"{name} image {b} row {row}"


@pytest.mark.parametrize(
    "name,masker", list(maskers()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_keep_indexes_the_input_rows_that_survived(name, masker):
    """The new field: ``keep`` over the input rows must reproduce the student exactly, so
    per-pixel truth indexed by it is aligned with the student's coordinates."""
    vox = make_batch()
    r = masker(vox)
    assert r.keep.shape[0] == vox.coordinate_tensor.shape[0]
    assert torch.equal(vox.coordinate_tensor[r.keep], r.student.coordinate_tensor), name
    assert torch.equal(vox.feature_tensor[r.keep], r.student.feature_tensor), name


@pytest.mark.parametrize(
    "name,masker", list(maskers()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_an_empty_image_stays_aligned(name, masker):
    vox = make_batch(counts=(30, 0, 20))
    r = masker(vox)
    B = len(vox.offsets) - 1
    assert len(r.student.offsets) - 1 == B, f"{name}: offsets lost an image"
    assert len(r.masked_coords) == B and len(r.masked_feats) == B
    assert r.masked_coords[1].shape[0] == 0 and r.masked_feats[1].shape[0] == 0


@pytest.mark.parametrize(
    "name,masker", list(maskers()), ids=lambda x: x if isinstance(x, str) else ""
)
def test_every_non_empty_image_keeps_at_least_one_voxel(name, masker):
    vox = make_batch(counts=(3, 1, 50))
    r = masker(vox)
    for b in (0, 1, 2):
        assert rows_of(r.student, b).shape[0] >= 1, f"{name} image {b} emptied"


def test_an_entirely_empty_batch_is_handled():
    vox = make_batch(counts=(0, 0))
    for name, m in maskers(build_candidates=True):
        r = m(vox)
        assert r.student is vox and len(r.masked_coords) == 2, name
        if name == "region":
            assert r.cand_coords is not None and len(r.cand_coords) == 2


def test_only_the_region_masker_builds_candidates():
    vox = make_batch()
    for name, m in maskers():
        r = m(vox)
        assert r.cand_coords is None and r.occ_targets is None, name
    r = region(build_candidates=True, cand_stride=STRIDE)(vox)
    assert r.cand_coords is not None and r.occ_targets is not None


# ------------------------------------------------------------------- region: geometry


def cell_of(x: int, y: int) -> tuple:
    return (x // CELL_W, y // CELL_H)


def test_wipe_empties_whole_cells():
    vox = make_batch()
    r = region(flavor="wipe", wipe_max=0.75)(vox)
    for b in range(len(vox.offsets) - 1):
        kept = {cell_of(*c) for c in as_pairs(rows_of(r.student, b))}
        gone = {cell_of(*c) for c in as_pairs(r.masked_coords[b])}
        assert not (kept & gone), f"image {b}: cells partly masked, partly kept"


@pytest.mark.parametrize("wipe_max", [0.25, 0.5, 0.75])
def test_the_wipe_ceiling_holds(wipe_max):
    vox = make_batch()
    r = region(flavor="wipe", wipe_max=wipe_max)(vox)
    for b in range(len(vox.offsets) - 1):
        n = rows_of(vox, b).shape[0]
        if n == 0:
            continue
        assert r.masked_coords[b].shape[0] <= int(wipe_max * n)
        assert rows_of(r.student, b).shape[0] > 0


def test_wipe_masks_something_in_every_non_empty_image():
    for seed in range(5):
        for counts in ((120, 0, 80), (400, 0, 30), (12, 0, 9)):
            vox = make_batch(counts=counts, seed=seed)
            torch.manual_seed(seed)
            r = RegionMasker(
                image_w=CANVAS_W,
                image_h=CANVAS_H,
                cell_w=CELL_W,
                cell_h=CELL_H,
                flavor="wipe",
                wipe_max=0.75,
            )(vox)
            for b in range(len(vox.offsets) - 1):
                if rows_of(vox, b).shape[0]:
                    assert r.masked_coords[b].shape[0] > 0, f"seed {seed} {counts} image {b}"


def test_surplus_release_never_labels_visible_charge_empty():
    """A cell dense enough to bust the ceiling on its own is still taken and the surplus is
    released back -- visible charge inside a taken cell. Only fully emptied cells may be
    enumerated."""
    dense = torch.stack(
        torch.meshgrid(torch.arange(0, 10), torch.arange(0, 10), indexing="ij"), -1
    ).reshape(-1, 2)
    coords = torch.cat([dense, torch.tensor([[25, 25], [26, 26], [27, 27]])])
    offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int64)
    vox = Voxels(
        batched_coordinates=IntCoords(coords.to(torch.int32), offsets=offsets),
        batched_features=CatFeatures(torch.ones(coords.shape[0], 1), offsets=offsets),
        offsets=offsets,
    )
    fired = False
    for seed in range(8):
        torch.manual_seed(seed)
        r = RegionMasker(
            image_w=CANVAS_W,
            image_h=CANVAS_H,
            cell_w=CELL_W,
            cell_h=CELL_H,
            flavor="wipe",
            wipe_max=0.5,
            build_candidates=True,
            cand_stride=STRIDE,
        )(vox)
        n_masked = r.masked_coords[0].shape[0]
        if 0 < n_masked < dense.shape[0]:
            fired = True
        survivors = {
            (int(x) // STRIDE, int(y) // STRIDE) for x, y in r.student.coordinate_tensor.tolist()
        }
        for c in as_pairs(r.cand_coords[0]):
            assert c not in survivors, f"seed {seed}: candidate {c} covers visible charge"
    assert fired, "the surplus-release branch never ran; this test proved nothing"


def test_candidates_are_exactly_the_wiped_cells_at_the_candidate_stride():
    vox = make_batch()
    r = region(flavor="wipe", wipe_max=0.75, build_candidates=True, cand_stride=STRIDE)(vox)
    for b in range(len(vox.offsets) - 1):
        wiped = {cell_of(*c) for c in as_pairs(r.masked_coords[b])}
        for cx, cy in as_pairs(r.cand_coords[b]):
            assert cell_of(cx * STRIDE, cy * STRIDE) in wiped
        expected = {
            (x // STRIDE, y // STRIDE)
            for gx, gy in wiped
            for x in range(gx * CELL_W, (gx + 1) * CELL_W)
            for y in range(gy * CELL_H, (gy + 1) * CELL_H)
        }
        assert as_pairs(r.cand_coords[b]) == expected
        assert r.cand_coords[b].shape[0] == len(expected), "duplicates"


def test_labels_match_the_pre_mask_image_and_never_a_survivor():
    vox = make_batch()
    r = region(flavor="wipe", wipe_max=0.75, build_candidates=True, cand_stride=STRIDE)(vox)
    for b in range(len(vox.offsets) - 1):
        active = {(int(x) // STRIDE, int(y) // STRIDE) for x, y in rows_of(vox, b).tolist()}
        survivors = {
            (int(x) // STRIDE, int(y) // STRIDE) for x, y in rows_of(r.student, b).tolist()
        }
        for row, (cx, cy) in enumerate((int(x), int(y)) for x, y in r.cand_coords[b].tolist()):
            assert float(r.occ_targets[b][row]) == (1.0 if (cx, cy) in active else 0.0)
            assert (cx, cy) not in survivors


def test_both_classes_are_present():
    r = region(flavor="wipe", wipe_max=0.75, build_candidates=True, cand_stride=STRIDE)(
        make_batch()
    )
    for b in (0, 2):
        n_pos = int(r.occ_targets[b].sum())
        assert n_pos > 0 and r.occ_targets[b].numel() - n_pos > 0


def test_randomize_leaks_as_documented():
    """It is EXPECTED to leak -- that is why it is refused as a reconstruction target."""
    r = region(flavor="randomize", r1=0.9, r2=0.5, build_candidates=True, cand_stride=STRIDE)(
        make_batch()
    )
    leaks = 0
    for b in range(3):
        survivors = {
            (int(x) // STRIDE, int(y) // STRIDE) for x, y in rows_of(r.student, b).tolist()
        }
        for row, c in enumerate((int(x), int(y)) for x, y in r.cand_coords[b].tolist()):
            leaks += c in survivors and float(r.occ_targets[b][row]) == 0.0
    assert leaks > 0


# ---------------------------------------------------------------- region: negative caps


def test_cap_keeps_every_positive():
    cand_b = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    occ = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    keep = cap_negatives(cand_b, occ, B=2, max_neg=1)
    assert bool(keep[occ > 0.5].all())
    for b in (0, 1):
        assert int(((cand_b == b) & keep & (occ < 0.5)).sum()) <= 1


def test_cap_by_ratio_and_off_by_default():
    cand_b = torch.zeros(21, dtype=torch.int64)
    occ = torch.cat([torch.ones(3), torch.zeros(18)])
    assert int((cap_negatives(cand_b, occ, B=1, neg_per_pos=2.0) & (occ < 0.5)).sum()) == 6
    assert bool(cap_negatives(cand_b, occ, B=1).all())


def test_a_capped_run_sub_samples_and_never_relabels():
    vox = make_batch()
    r = region(flavor="wipe", wipe_max=0.75, build_candidates=True, cand_stride=STRIDE, max_neg=5)(
        vox
    )
    for b in range(3):
        active = {(int(x) // STRIDE, int(y) // STRIDE) for x, y in rows_of(vox, b).tolist()}
        n_neg = 0
        for row, c in enumerate((int(x), int(y)) for x, y in r.cand_coords[b].tolist()):
            want = 1.0 if c in active else 0.0
            assert float(r.occ_targets[b][row]) == want
            n_neg += want == 0.0
        assert n_neg <= 5


# ----------------------------------------------------------------- construction rules


def test_an_indivisible_canvas_is_refused():
    with pytest.raises(ValueError, match="evenly"):
        RegionMasker(image_w=1050, image_h=1500, cell_w=64, cell_h=100)


def test_the_cell_must_divide_the_candidate_stride():
    with pytest.raises(ValueError, match="stride"):
        RegionMasker(image_w=105, image_h=100, cell_w=7, cell_h=10, cand_stride=2)


def test_the_production_grid_is_exact():
    m = RegionMasker(image_w=1050, image_h=1500, cell_w=70, cell_h=100)
    assert (m.n_cols, m.n_rows) == (15, 15)
    assert m.requires_full_canvas, "ADR 0004: this is the masker that cannot run on a crop"
    assert not PixelMasker().requires_full_canvas and not BlockMasker().requires_full_canvas


def test_ratios_are_range_checked():
    with pytest.raises(ValueError):
        PixelMasker(ratio=1.0)
    with pytest.raises(ValueError):
        BlockMasker(ratio=0.0)
