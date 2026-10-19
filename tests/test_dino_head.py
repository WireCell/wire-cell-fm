"""`DINOProjectionHead` scores a feature against unit-norm prototypes: the output is a cosine,
and rescaling a prototype's stored row changes nothing."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402
from warpconvnet.geometry.types.voxels import Voxels  # noqa: E402

from wcfm.data.voxels import voxels_from  # noqa: E402
from wcfm.model.terms import DINOProjectionHead  # noqa: E402

pytestmark = pytest.mark.stack


def voxels(n=30, dim=8) -> Voxels:
    torch.manual_seed(0)
    coords = torch.randint(0, 100, (n, 2), dtype=torch.int32)
    offsets = torch.tensor([0, n // 2, n])
    return voxels_from(coords, torch.randn(n, dim), offsets)


def test_outputs_are_cosines_and_ignore_the_prototype_norm():
    torch.manual_seed(0)
    head = DINOProjectionHead(8, 16, 12, n_layers=2)
    x = voxels()
    before = head(x).feature_tensor
    assert before.shape == (30, 12)
    assert before.abs().max() <= 1.0 + 1e-6

    with torch.no_grad():
        head.last.block.weight[3].mul_(50.0)
    after = head(x).feature_tensor
    torch.testing.assert_close(after, before)


def test_the_prototype_weight_receives_a_gradient():
    head = DINOProjectionHead(8, 16, 12, n_layers=2)
    head(voxels()).feature_tensor.sum().backward()
    assert head.last.block.weight.grad is not None
    assert head.last.block.weight.grad.abs().sum() > 0
