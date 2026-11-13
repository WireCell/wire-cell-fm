"""The bottleneck attention's `qk_norm`, on the dense path, which runs without CUDA."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from wcfm.data.voxels import voxels_from  # noqa: E402
from wcfm.model.backbones.attention2d import SpatialFeatureAttention2D  # noqa: E402
from wcfm.model.config import MinkUNetConfig  # noqa: E402

pytestmark = pytest.mark.stack


def two_images(dim: int):
    torch.manual_seed(0)
    coords = torch.unique(torch.randint(0, 32, (60, 2), dtype=torch.int32), dim=0)
    return voxels_from(
        coords, torch.randn(coords.shape[0], dim), torch.tensor([0, 25, len(coords)])
    )


@pytest.mark.parametrize("qk_norm", [False, True])
def test_qk_norm_makes_the_attention_blind_to_the_scale_of_the_q_and_k_weights(qk_norm):
    """The runaway it guards against: q and k weights growing until a head is one-hot."""
    a = SpatialFeatureAttention2D(
        16, num_heads=4, qkv_bias=True, use_encoding=False, enable_flash=False, qk_norm=qk_norm
    ).eval()
    xs = two_images(16)
    with torch.no_grad():
        before = a(xs).feature_tensor.clone()
        a.qkv.weight[:2] *= 50
        a.qkv.bias[:2] *= 50
        after = a(xs).feature_tensor
    moved = float((after - before).abs().max())
    assert (moved < 1e-3) if qk_norm else (moved > 1e-1), moved


def test_a_stored_config_without_qk_norm_rebuilds_without_the_norms():
    """A checkpoint is rebuilt by calling the constructor with its stored config. A config
    written before `qk_norm` existed must rebuild without the norms, or its strict load fails
    on keys it never had. `test_the_old_backbone_registry_is_reachable_as_flags` pins that the
    shipped backbone turns them on."""
    on = OmegaConf.structured(MinkUNetConfig(encoding_range=64.0, qk_norm=True))
    keys = instantiate(on).state_dict()
    stored = OmegaConf.to_container(on)
    del stored["qk_norm"]
    old = instantiate(OmegaConf.create(stored)).state_dict()
    assert set(keys) - set(old) == {
        f"bottleneck.attn.{n}_norm.{p}" for n in "qk" for p in ("weight", "bias")
    }
