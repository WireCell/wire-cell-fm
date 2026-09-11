"""The cropper: crops are subsets of the source with their original coordinates, and the index
each crop reports selects exactly its rows -- which is what carries the truth along."""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402

from wcfm.model.augment import Cropper  # noqa: E402

from .fake_backbone import as_pairs, make_batch, rows_of  # noqa: E402

pytestmark = pytest.mark.stack

W, H = 64, 48


def cropper(**kw) -> Cropper:
    random.seed(0)
    torch.manual_seed(0)
    return Cropper(
        image_w=W, image_h=H, n_global=1, n_local=2, min_active_pixels=5, blur_sigma_px=2.0, **kw
    )


def test_crop_count_and_global_flags():
    crops = cropper()(make_batch((60, 50), width=W, height=H, blob=True).voxels)
    assert len(crops) == 3
    assert [c.is_global for c in crops] == [True, False, False]


def test_every_crop_is_a_subset_of_the_source_with_original_coordinates():
    vox = make_batch((60, 50), width=W, height=H, blob=True).voxels
    for crop in cropper()(vox):
        for b in range(2):
            assert as_pairs(rows_of(crop.voxels, b)) <= as_pairs(rows_of(vox, b))


def test_the_index_selects_exactly_the_crops_rows_in_order():
    """This is the alignment truth depends on."""
    vox = make_batch((60, 50), width=W, height=H, blob=True).voxels
    for crop in cropper()(vox):
        for b in range(2):
            s = int(vox.offsets[b])
            picked = vox.coordinate_tensor[s + crop.index[b]]
            assert torch.equal(picked, rows_of(crop.voxels, b))
            assert torch.equal(
                vox.feature_tensor[s + crop.index[b]],
                crop.voxels.feature_tensor[
                    int(crop.voxels.offsets[b]) : int(crop.voxels.offsets[b + 1])
                ],
            )


def test_an_empty_image_gives_empty_crop_entries_and_intact_offsets():
    vox = make_batch((40, 0, 30), width=W, height=H, blob=True).voxels
    for crop in cropper()(vox):
        assert len(crop.voxels.offsets) == 4
        assert rows_of(crop.voxels, 1).shape[0] == 0 and crop.index[1].numel() == 0


def test_local_crops_are_smaller_than_global_on_average():
    vox = make_batch((200, 200), width=W, height=H, blob=True).voxels
    crops = cropper()(vox)
    n_global = crops[0].voxels.coordinate_tensor.shape[0]
    n_local = sum(c.voxels.coordinate_tensor.shape[0] for c in crops[1:]) / 2
    assert n_local < n_global


def test_n_global_must_be_at_least_one():
    with pytest.raises(ValueError):
        Cropper(image_w=W, image_h=H, n_global=0)
