"""``OccupancyTerm``: the question the shadowed ``occ_coords`` meant nobody ever asked.

Everything here is CPU. The term's own geometry -- projecting candidates onto a half-resolution
skip -- is ``inject_into_skip``'s and is tested with the backbone; what is tested here is the
term's *orchestration*, which is where the old repo went wrong: which candidates get scored,
what happens to the ones the backbone did not place, and which configurations are refused
rather than silently trained.

The term is pinned to the fake backbone's taps through a subclass. ``READ_TAP`` and friends are
class attributes precisely so that reading at a different resolution is a subclass rather than a
flag, and this is the first use of that.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402

from wcfm.model.augment import Augment, BlockMasker, Cropper, RegionMasker  # noqa: E402
from wcfm.model.backbones.base import Injection, InjectionGroup  # noqa: E402
from wcfm.model.modules import NoTeacher, SslModule  # noqa: E402
from wcfm.model.terms import ChargeTerm, OccupancyTerm  # noqa: E402
from wcfm.model.terms.losses import occupancy_loss  # noqa: E402

from .fake_backbone import LinearBackbone, make_batch  # noqa: E402

pytestmark = pytest.mark.stack


class FlatOccupancyTerm(OccupancyTerm):
    """The same term reading the fake backbone's stride-1 tap."""

    READ_TAP = "hidden"
    READ_STRIDE = 1
    INJECT_TAP = "enc0"


def region_masker(**kw) -> RegionMasker:
    opts = dict(
        image_w=8,
        image_h=8,
        cell_w=4,
        cell_h=4,
        flavor="wipe",
        build_candidates=True,
        cand_stride=1,
        neg_per_pos=3.0,
    )
    opts.update(kw)
    return RegionMasker(**opts)


def backbone(roles=("masked", "candidate")) -> LinearBackbone:
    return LinearBackbone(inject_roles=roles)


# --------------------------------------------------------------------------- the loss


def test_focal_loss_discounts_the_examples_already_right():
    """The candidate set is overwhelmingly empty, so a plain BCE settles on "predict empty".
    The focal weighting is what stops that, and it is the reason this is not `charge_loss`."""
    counts = torch.tensor([4])
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    confident = torch.tensor([6.0, -6.0, -6.0, -6.0])  # right, and sure
    unsure = torch.tensor([0.1, -0.1, -0.1, -0.1])  # right, barely
    assert occupancy_loss(confident, target, counts) < occupancy_loss(unsure, target, counts)


def test_an_empty_candidate_set_is_a_zero_that_still_carries_grad():
    """A view can legitimately enumerate nothing. That must not be a NaN, and must not detach
    the graph -- under DDP a rank returning a detached loss desynchronises the reducer."""
    logits = torch.zeros(0, requires_grad=True)
    loss = occupancy_loss(logits, torch.zeros(0), torch.tensor([0]))
    assert loss.requires_grad and float(loss.detach()) == 0.0


# ------------------------------------------------------------------- what gets scored


def _module(term, masker=None, cropper=None, roles=("masked", "candidate"), bb=None):
    return SslModule(
        backbone=bb or backbone(roles),
        terms={"occupancy": term},
        augment=Augment(cropper=cropper, masker=masker or region_masker()),
        teacher=NoTeacher(),
    )


def test_only_the_candidates_the_backbone_placed_are_scored():
    """A candidate coinciding with a surviving voxel is deduped against the skip and no token
    is placed for it. Scoring it would read an ordinary feature as though it were a prediction
    -- which is what the old repo did with its grown set, without saying so."""
    term = FlatOccupancyTerm()
    term.build(backbone())
    coords = [torch.tensor([[0, 0], [1, 1], [2, 2]])]
    targets = [torch.tensor([1.0, 0.0, 1.0])]

    class Bundle:
        placed = [torch.tensor([[0, 0], [2, 2]])]
        injected = Injection([InjectionGroup("enc0", placed, "candidate", 1)])

    kept_c, kept_t, dropped = term._intersect_with_placed(Bundle(), coords, targets)
    assert dropped == 1, "the candidate at (1, 1) was never injected"
    assert kept_c[0].tolist() == [[0, 0], [2, 2]]
    assert kept_t[0].tolist() == [1.0, 1.0]


def test_nothing_placed_at_all_drops_every_candidate_rather_than_scoring_them():
    term = FlatOccupancyTerm()
    term.build(backbone())
    coords = [torch.tensor([[0, 0], [1, 1]])]
    targets = [torch.tensor([1.0, 0.0])]

    class Bundle:
        nothing = [torch.zeros(0, 2, dtype=torch.long)]
        injected = Injection([InjectionGroup("enc0", nothing, "candidate", 1)])

    kept_c, _, dropped = term._intersect_with_placed(Bundle(), coords, targets)
    assert dropped == 2 and kept_c[0].shape[0] == 0


def test_the_masked_role_does_not_satisfy_a_candidate_request():
    """The whole point of role-typing: a `masked` token at a coordinate is a different
    question from a `candidate` token there, and must not be read as one."""
    term = FlatOccupancyTerm()
    term.build(backbone())
    coords = [torch.tensor([[0, 0]])]

    class Bundle:
        injected = Injection([InjectionGroup("enc0", [torch.tensor([[0, 0]])], "masked", 1)])

    _, _, dropped = term._intersect_with_placed(Bundle(), coords, [torch.tensor([1.0])])
    assert dropped == 1


def test_the_term_asks_for_the_tap_it_reads_whatever_the_metrics_config_wants():
    """`observe_taps` is an observation request; a term's tap is a data dependency. A term
    silently getting no tap would raise deep inside the forward, or worse, read a stale one."""
    module = _module(FlatOccupancyTerm())
    assert "hidden" in module.required_taps
    assert not module.observe_taps, "nothing observed, and the tap is still requested"


# --------------------------------------------------------------------- what is refused


def test_a_masker_with_no_candidate_source_is_refused_with_the_reason():
    """The block masker is what all 31 archived configs ran, and in the old repo its
    occupancy candidates came from the grown set the shadowed `occ_coords` forced. That path
    is not ported, so this must fail loudly rather than train on an empty candidate set."""
    with pytest.raises(ValueError, match="enumerate occupancy candidates"):
        _module(FlatOccupancyTerm(), masker=BlockMasker(ratio=0.5))


def test_a_backbone_without_a_candidate_token_is_refused():
    with pytest.raises(ValueError, match="candidate"):
        _module(FlatOccupancyTerm(), roles=("masked",))


def test_an_uncapped_candidate_set_is_refused():
    """`validate_config` mandated the cap in the old repo, where it could not have been doing
    what its message claimed because the capped set never reached the decoder."""
    with pytest.raises(ValueError, match="negatives cap"):
        _module(FlatOccupancyTerm(), masker=region_masker(neg_per_pos=None, max_neg=None))


def test_a_non_wipe_flavor_is_refused():
    with pytest.raises(ValueError, match="flavor: wipe"):
        _module(FlatOccupancyTerm(), masker=region_masker(flavor="randomize"))


def test_a_cell_that_does_not_tile_the_read_stride_is_refused():
    """A cell that does not divide the read stride puts part of a wiped cell in a
    half-resolution voxel the rest of which survived: the label says removed, the input says
    not. Invisible in the loss -- it just trains on a mislabelled rim of every cell."""

    class HalfRes(FlatOccupancyTerm):
        READ_STRIDE = 2

    class HalfResBackbone(LinearBackbone):
        TAP_STRIDE = {"enc0": 1, "hidden": 2}

    # 3 tiles a 12-wide canvas evenly -- so the masker's own invariant passes, and so does the
    # term's own stride check -- and does not divide a stride-2 read, which is the one left.
    with pytest.raises(ValueError, match="mislabelled rim"):
        _module(
            HalfRes(),
            masker=region_masker(image_w=12, cell_w=3, cell_h=4),
            bb=HalfResBackbone(inject_roles=("masked", "candidate")),
        )


def test_region_occupancy_still_refuses_cropping():
    """ADR 0004, and the reason the old repo's region arm trained on a rim-biased sliver."""
    with pytest.raises(ValueError, match="cannot run\n?.*on a crop|on a crop"):
        _module(FlatOccupancyTerm(), cropper=Cropper(image_w=8, image_h=8))


# ------------------------------------------------------------------ alongside charge


def test_charge_and_occupancy_ask_for_different_roles():
    """The `mae` shape: two terms, two roles, one backbone. If these collided on one role the
    two questions would share a learned vector, which is the design hole role-typing closes."""
    occ, charge = FlatOccupancyTerm(), ChargeTerm()
    module = SslModule(
        backbone=backbone(),
        terms={"charge": charge, "occupancy": occ},
        augment=Augment(cropper=None, masker=region_masker()),
        teacher=NoTeacher(),
    )
    assert set(charge.inject_roles) == {"masked"}
    assert set(occ.inject_roles) == {"candidate"}
    assert set(module.backbone.inject_roles) == {"masked", "candidate"}


def test_charge_does_not_inject_where_occupancy_enumerates(tmp_path):
    """The production `mae` shape, which the fake backbone above cannot express: two injection
    sites at two strides, occupancy on the coarser one.

    Charge's coordinates projected onto a stride-2 skip are a SUBSET of the candidates there,
    because the candidates cover every 2x2 block of a wiped cell. Injecting `masked` at both
    sites therefore hands the backbone a coordinate under two roles on every step, and it
    refuses. Charge injects where it reads instead, which is the stride-1 site.
    """
    from wcfm.data.voxels import voxels_from
    from wcfm.model.backbones.base import inject_into_skip

    class TwoSiteBackbone(LinearBackbone):
        TAPS = ("enc0", "enc1", "hidden")
        TAP_STRIDE = {"enc0": 1, "enc1": 2, "hidden": 1}
        INJECT_TAPS = ("enc0", "enc1")

        def tap_dim(self, tap: str) -> int:
            return super().tap_dim("enc0" if tap == "enc1" else tap)

    class HalfResOcc(FlatOccupancyTerm):
        READ_STRIDE = 2
        READ_TAP = "enc1"
        INJECT_TAP = "enc1"

    bb = TwoSiteBackbone(inject_roles=("masked", "candidate"))
    charge, occ = ChargeTerm(), HalfResOcc()
    charge.build(bb)
    occ.build(bb)

    masker = region_masker(image_w=8, image_h=8, cell_w=4, cell_h=4, cand_stride=2)
    plan = Augment(cropper=None, masker=masker)(make_batch(counts=(24, 24), width=8, height=8))
    view = plan.views[0]
    assert view.mask is not None and view.mask.cand_coords is not None

    charge_groups = charge.injection_request(view).groups
    assert {g.tap for g in charge_groups} == {"enc0"}, "charge injects only where it reads"

    # Both requests, at the tap occupancy owns: the backbone accepts them now.
    both = Injection(charge_groups + occ.injection_request(view).groups)
    half = voxels_from(
        (view.voxels.coordinate_tensor // 2).int(),
        view.voxels.feature_tensor,
        view.voxels.offsets,
    )
    width = half.feature_tensor.shape[1]
    tokens = {r: torch.zeros(width) for r in ("masked", "candidate")}
    inject_into_skip(half, "enc1", 2, both, tokens)


def test_a_step_of_mae_runs_and_both_terms_reach_the_loss():
    """End to end on the fake backbone: the two terms compose, the occupancy head runs on its
    tap, and the total is finite and attached."""
    torch.manual_seed(0)
    module = SslModule(
        backbone=backbone(),
        terms={"charge": ChargeTerm(weight=0.1), "occupancy": FlatOccupancyTerm(weight=1.0)},
        augment=Augment(cropper=None, masker=region_masker()),
        teacher=NoTeacher(),
    )
    batch = make_batch(counts=(24, 24), width=8, height=8)
    plan = module.augment(batch)
    assert plan.views, "the region masker produced no view"
    view = plan.views[0]
    assert view.mask is not None and view.mask.cand_coords is not None


# ------------------------------------------------------------------- grown candidates

from wcfm.data.voxels import voxels_from  # noqa: E402
from wcfm.model.terms.occupancy import grow_candidates, occupied_at  # noqa: E402

from .fake_backbone import as_pairs  # noqa: E402


def _vox(*images, stride=None):
    coords = [torch.as_tensor(c, dtype=torch.int32).reshape(-1, 2) for c in images]
    counts = [c.shape[0] for c in coords]
    offsets = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int64)
    v = voxels_from(torch.cat(coords), torch.ones(sum(counts), 1), offsets)
    return v


def _box(x0, x1, y0, y1):
    return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


def test_growth_from_a_stride8_bottleneck_doubles_twice_then_dilates():
    """One visible pixel: bottleneck cell (2, 1). x2 and 3x3 gives x 3..5, y 1..3; x2 again
    and 3x3 gives x 5..11, y 1..7 at stride 2. Each further iteration dilates by one."""
    vox = _vox([[17, 9]])
    one = grow_candidates(vox, 8, 2, iters=1)[0]
    assert as_pairs(one) == _box(5, 11, 1, 7)
    three = grow_candidates(vox, 8, 2, iters=3)[0]
    assert as_pairs(three) == _box(3, 13, 0, 9)  # clamped at 0, not above


def test_growth_is_per_image_and_an_empty_image_grows_nothing():
    grown = grow_candidates(_vox([[17, 9]], [], [[0, 0]]), 4, 2, iters=1)
    assert [g.shape[0] > 0 for g in grown] == [True, False, True]
    assert as_pairs(grown[2]) == _box(0, 1, 0, 1)


def test_a_row_is_occupied_when_its_block_holds_any_clean_pixel():
    rows = _vox([[0, 0], [1, 0], [5, 5]], [[0, 0]])
    clean = _vox([[1, 1], [11, 11]], [[3, 3]])
    assert occupied_at(rows, clean, 2).tolist() == [True, False, True, False]


class BottleneckBackbone(LinearBackbone):
    TAP_STRIDE = {"enc0": 1, "hidden": 1, "bottleneck_attn": 4}


def test_grown_mode_refuses_a_masker_that_also_enumerates():
    with pytest.raises(ValueError, match="build_candidates: false"):
        _module(
            FlatOccupancyTerm(candidates="grown"),
            bb=BottleneckBackbone(inject_roles=("masked", "candidate")),
        )


def test_an_unknown_candidate_source_is_refused():
    with pytest.raises(ValueError, match="'masker' or 'grown'"):
        OccupancyTerm(candidates="dense")


# ------------------------------------------------------------------ copies and charge sites


def test_copies_are_independently_masked_whole_images():
    torch.manual_seed(0)
    aug = Augment(masker=region_masker(image_w=64, image_h=48, cell_w=8, cell_h=8,
                                       build_candidates=False), copies=3)
    plan = aug(make_batch(width=64, height=48))
    assert plan.n_views == plan.n_global == 3
    masks = [as_pairs(v.mask.masked_coords[0]) for v in plan.views]
    assert len({frozenset(m) for m in masks}) > 1
    assert all(torch.equal(v.clean.coordinate_tensor, plan.source.voxels.coordinate_tensor)
               for v in plan.views)


def test_copies_need_a_masker_and_no_cropper():
    with pytest.raises(ValueError, match="no cropper"):
        Augment(copies=2)


def test_charge_refuses_an_injection_site_the_backbone_lacks():
    with pytest.raises(ValueError, match="injects at"):
        SslModule(
            backbone=backbone(),
            terms={"charge": ChargeTerm(inject_taps=("enc2",))},
            augment=Augment(masker=region_masker()),
            teacher=NoTeacher(),
        )


def test_copies_share_one_teacher_encode():
    """The copies share their clean image, so the teacher runs on it once, not once per copy."""
    from wcfm.model.modules.ssl import _per_clean_view

    aug = Augment(
        masker=region_masker(image_w=64, image_h=48, cell_w=8, cell_h=8, build_candidates=False),
        copies=3,
    )
    plan = aug(make_batch(width=64, height=48))
    calls = []
    out = _per_clean_view(plan, lambda c: calls.append(c) or len(calls))
    assert len(calls) == 1 and out == [1, 1, 1]
