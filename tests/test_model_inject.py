"""Role-typed injection: the coordinate arithmetic in front of every backbone.

``project_coords`` carries the old repo's ``test_maskproj_keys.py``: the key-wrap bug where a
masked coordinate to the right of every skip coordinate was silently dropped as a duplicate,
leaving no token there. ``inject_into_skip`` is new and is what the union of terms relies on.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402
from warpconvnet.geometry.coords.integer import IntCoords  # noqa: E402
from warpconvnet.geometry.features.cat import CatFeatures  # noqa: E402
from warpconvnet.geometry.types.voxels import Voxels  # noqa: E402

from wcfm.model.backbones import (  # noqa: E402
    Injection,
    InjectionGroup,
    inject_into_skip,
    project_coords,
)

from .fake_backbone import as_pairs  # noqa: E402

pytestmark = pytest.mark.stack


def skip_of(coords, feats=None, stride=None) -> Voxels:
    coords = torch.as_tensor(coords, dtype=torch.int32)
    offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int64)
    feats = torch.zeros(coords.shape[0], 4) if feats is None else feats
    return Voxels(
        batched_coordinates=IntCoords(coords, offsets=offsets, tensor_stride=stride),
        batched_features=CatFeatures(feats, offsets=offsets),
        offsets=offsets,
    )


# --------------------------------------------------------------------- project_coords


def test_a_far_right_coordinate_is_not_swallowed_by_a_key_wrap():
    """(100, 0) sets the skip's largest x; (105, 3) wraps onto (3, 4)'s key if W is sized from
    the skip alone: 3*102+105 == 4*102+3."""
    skip = skip_of([[100, 0], [3, 4], [7, 9]])
    out = project_coords([torch.tensor([[105, 3]])], skip, 1)[0]
    assert (105, 3) in as_pairs(out), "dropped as a duplicate it is not"


def test_a_real_duplicate_is_still_dropped():
    skip = skip_of([[100, 0], [3, 4], [7, 9]])
    out = as_pairs(project_coords([torch.tensor([[7, 9], [105, 3]])], skip, 1)[0])
    assert (7, 9) not in out and (105, 3) in out


def test_projection_floor_divides_and_dedupes():
    skip = skip_of([[50, 0], [1, 2]])
    out = project_coords([torch.tensor([[211, 7], [210, 6]])], skip, 2)[0]
    # both land on (105, 3): one row, not two
    assert as_pairs(out) == {(105, 3)} and out.shape[0] == 1


def test_empty_inputs_do_not_raise():
    skip = skip_of([[5, 5]])
    assert project_coords([torch.zeros(0, 2, dtype=torch.int64)], skip, 1)[0].shape[0] == 0
    empty_skip = skip_of(torch.zeros(0, 2))
    out = project_coords([torch.tensor([[3, 4]])], empty_skip, 1)
    assert isinstance(out, list) and as_pairs(out[0]) == {(3, 4)}


def test_a_request_list_longer_than_the_skip_is_truncated_to_its_offsets():
    """WarpConvNet's strided conv drops trailing empty images from its offsets."""
    skip = skip_of([[1, 1]])  # B = 1
    out = project_coords([torch.tensor([[2, 2]]), torch.tensor([[9, 9]])], skip, 1)
    assert len(out) == 1


# -------------------------------------------------------------------- inject_into_skip


def _tokens(dim=4):
    return {"masked": torch.full((dim,), 7.0), "candidate": torch.full((dim,), -3.0)}


def test_injected_rows_carry_their_roles_token_and_the_skip_keeps_its_rows():
    skip = skip_of([[1, 1], [2, 2]], feats=torch.ones(2, 4))
    inj = Injection([InjectionGroup("enc0", [torch.tensor([[5, 5], [6, 6]])], "masked", 1)])
    aug, injected = inject_into_skip(skip, "enc0", 1, inj, _tokens())
    assert as_pairs(aug.coordinate_tensor) == {(1, 1), (2, 2), (5, 5), (6, 6)}
    feats = aug.feature_tensor
    assert torch.equal(feats[:2], torch.ones(2, 4)), "the skip's own rows are untouched"
    assert torch.equal(feats[2:], torch.full((2, 4), 7.0)), "injected rows are the token"
    assert len(injected) == 1 and injected[0].role == "masked" and injected[0].stride == 1
    assert as_pairs(injected[0].coords[0]) == {(5, 5), (6, 6)}


def test_a_coordinate_the_skip_already_has_is_not_injected_and_is_reported_absent():
    skip = skip_of([[1, 1], [2, 2]])
    inj = Injection([InjectionGroup("enc0", [torch.tensor([[2, 2], [5, 5]])], "masked", 1)])
    aug, injected = inject_into_skip(skip, "enc0", 1, inj, _tokens())
    assert aug.coordinate_tensor.shape[0] == 3, "(2, 2) must not be doubled"
    assert as_pairs(injected[0].coords[0]) == {(5, 5)}, "the report is what was placed"


def test_two_terms_requesting_the_same_role_get_one_token_per_coordinate():
    """DINO and charge both ask for the masked coordinates: union, not two copies."""
    skip = skip_of([[1, 1]])
    a = InjectionGroup("enc0", [torch.tensor([[5, 5], [6, 6]])], "masked", 1)
    b = InjectionGroup("enc0", [torch.tensor([[6, 6], [7, 7]])], "masked", 1)
    aug, injected = inject_into_skip(skip, "enc0", 1, Injection([a, b]), _tokens())
    assert as_pairs(aug.coordinate_tensor) == {(1, 1), (5, 5), (6, 6), (7, 7)}
    assert aug.coordinate_tensor.shape[0] == 4
    assert len(injected) == 1


def test_two_roles_at_one_tap_get_distinct_tokens():
    skip = skip_of([[1, 1]], feats=torch.zeros(1, 4))
    m = InjectionGroup("enc1", [torch.tensor([[4, 4]])], "masked", 2)
    c = InjectionGroup("enc1", [torch.tensor([[9, 9]])], "candidate", 2)
    aug, injected = inject_into_skip(skip, "enc1", 2, Injection([m, c]), _tokens())
    rows = {
        tuple(int(v) for v in xy): float(f[0])
        for xy, f in zip(aug.coordinate_tensor, aug.feature_tensor, strict=True)
    }
    assert rows[(4, 4)] == 7.0 and rows[(9, 9)] == -3.0
    assert {g.role for g in injected} == {"masked", "candidate"}


def test_the_same_coordinate_under_two_roles_is_refused():
    """Two questions, one token: the design hole role typing exists to close. Stage 4 decides
    what the union does; until then it is an error, not a silent choice."""
    skip = skip_of([[1, 1]])
    m = InjectionGroup("enc1", [torch.tensor([[4, 4]])], "masked", 2)
    c = InjectionGroup("enc1", [torch.tensor([[4, 4]])], "candidate", 2)
    with pytest.raises(ValueError, match="two roles"):
        inject_into_skip(skip, "enc1", 2, Injection([m, c]), _tokens())


def test_full_res_masked_coordinates_project_onto_the_half_res_skip():
    """The DINO/charge request is at stride 1; the enc1 skip is at stride 2."""
    skip = skip_of([[0, 0]], stride=(2, 2))
    m = InjectionGroup("enc1", [torch.tensor([[10, 10], [11, 11], [20, 21]])], "masked", 1)
    aug, injected = inject_into_skip(skip, "enc1", 2, Injection([m]), _tokens())
    assert as_pairs(injected[0].coords[0]) == {(5, 5), (10, 10)}, "floor-divided and deduped"
    assert injected[0].stride == 2, "reported in the tap's units"
    assert aug.batched_coordinates.tensor_stride == (2, 2), "the skip's stride is preserved"


def test_a_stride_that_does_not_divide_the_tap_is_refused():
    skip = skip_of([[0, 0]])
    g = InjectionGroup("enc1", [torch.tensor([[4, 4]])], "masked", 4)
    with pytest.raises(ValueError, match="not divisible"):
        inject_into_skip(skip, "enc1", 2, Injection([g]), _tokens())


def test_nothing_requested_at_this_tap_returns_the_skip_itself():
    skip = skip_of([[1, 1]])
    inj = Injection([InjectionGroup("enc1", [torch.tensor([[4, 4]])], "masked", 1)])
    aug, injected = inject_into_skip(skip, "enc0", 1, inj, _tokens())
    assert aug is skip and injected == []


def test_injection_group_rejects_an_unknown_role():
    with pytest.raises(ValueError, match="unknown injection role"):
        InjectionGroup("enc0", [], "occupied", 1)


def test_merge_of_nothing_is_none():
    assert Injection.merge([None, None]) is None
    g = InjectionGroup("enc0", [torch.zeros(0, 2)], "masked", 1)
    merged = Injection.merge([None, Injection([g]), Injection([g])])
    assert merged is not None and len(merged.groups) == 2
