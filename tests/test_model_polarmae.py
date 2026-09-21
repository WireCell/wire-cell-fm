"""`PolarMAEBackbone` honours the `Backbone` contract on the CPU, and a checkpoint carrying it
goes through `load_module` and `extract` with its `local` tap.

The backbone here is a few thousand parameters wide; the architecture is the real one.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402

from wcfm.engine.checkpoint import Checkpoint, save_checkpoint  # noqa: E402
from wcfm.eval.extract import OUT_TAP, extract  # noqa: E402
from wcfm.eval.format import FeatureStore  # noqa: E402
from wcfm.eval.loading import inference_step, load_module  # noqa: E402
from wcfm.model.backbones import PolarMAEBackbone  # noqa: E402
from wcfm.model.backbones.polarmae import LOCAL_DIM, random_token_mask  # noqa: E402

from .fake_backbone import make_batch

TINY = {
    "center": [32.0, 24.0, 0.0],
    "scale": 1.0 / 40.0,
    "group_radius_px": 3.0,
    "num_init_groups": 16,
    "context_length": 12,
    "group_max_points": 6,
    "group_upscale_points": 16,
    "overlap_factor": 0.5,
    "embed_dim": 24,
    "depth": 2,
    "num_heads": 2,
    "decoder_depth": 1,
    "upsample_k": 3,
}


def tiny(**over) -> PolarMAEBackbone:
    torch.manual_seed(0)
    bb = PolarMAEBackbone(**{**TINY, **over})
    return bb.eval()


def batch(counts=(40, 30, 0, 12), seed=1):
    return make_batch(counts, width=64, height=48, seed=seed)


# ------------------------------------------------------------------------ the contract


def test_out_and_local_sit_on_the_input_coordinates():
    bb = tiny()
    xs = batch().voxels
    with torch.no_grad():
        fb = bb(xs, None, ("local",))
    assert torch.equal(fb.out.offsets, xs.offsets)
    assert torch.equal(fb.out.coordinate_tensor, xs.coordinate_tensor)
    assert fb.out.feature_tensor.shape == (xs.coordinate_tensor.shape[0], bb.out_dim)
    assert fb.taps["local"].feature_tensor.shape[1] == LOCAL_DIM == bb.tap_dim("local")
    assert torch.isfinite(fb.out.feature_tensor).all()
    assert fb.injected is None
    assert not bb.supports_injection


def test_an_unknown_tap_and_an_injection_are_refused():
    from wcfm.model.backbones import Injection, InjectionGroup

    bb = tiny()
    xs = batch().voxels
    with pytest.raises(ValueError, match="no tap"):
        bb(xs, None, ("enc0",))
    inj = Injection([InjectionGroup("local", [torch.zeros(1, 2)], "masked")])
    with pytest.raises(ValueError, match="cannot inject"):
        bb(xs, inj, ())


def test_every_point_maps_back_to_its_row():
    """`points_from` and the `(event, slot)` index are inverse to each other."""
    bb = tiny()
    xs = batch().voxels
    points, lengths, (event, slot) = bb.points_from(xs)
    assert lengths.tolist() == [40, 30, 0, 12]
    back = points[event, slot]
    xy = (back[:, :2] / bb.scale) + bb.center[:2]
    assert torch.allclose(xy, xs.coordinate_tensor[:, :2].float(), atol=1e-4)
    assert torch.allclose(back[:, 3], xs.feature_tensor[:, 0])


def test_every_retained_centre_is_a_token():
    bb = tiny()
    points, lengths, _ = bb.points_from(batch().voxels)
    g = bb.grouping(points, lengths)
    assert torch.equal(g.emb_mask, g.point_mask.sum(2) > 0)
    assert torch.equal(g.point_mask, g.idx.ge(0))
    assert (g.groups[~g.emb_mask] == 0).all()
    # a retained centre is a point of the cloud, so it is a member of its own ball
    assert g.emb_mask.sum(1).tolist() == [int(n) for n in (g.point_mask.sum(2) > 0).sum(1)]


def test_layer_norm_makes_an_event_independent_of_its_batch():
    """With `norm="layer"` attention and normalisation stay inside an event, so its features
    do not move when other events share the batch; with `norm="global"` they do."""
    a = batch((40,), seed=3).voxels
    ab = batch((40, 25), seed=3).voxels  # the same first event, one more beside it
    for norm, same in (("layer", True), ("global", False)):
        bb = tiny(norm=norm)
        with torch.no_grad():
            alone = bb(a).out.feature_tensor
            together = bb(ab).out.feature_tensor[:40]
        assert torch.allclose(alone, together, atol=1e-4) is same, norm


def test_encode_decode_and_the_token_mask():
    bb = tiny().train()
    points, lengths, _ = bb.points_from(batch().voxels)
    tb = bb.tokenize(points, lengths)
    masked, visible = random_token_mask(tb.lengths, tb.tokens.shape[1], 0.6)
    assert not (masked & visible).any()
    assert torch.equal(masked | visible, tb.emb_mask)
    assert masked.sum(1).tolist() == (0.6 * tb.lengths).to(torch.int64).tolist()
    enc = bb.encode(tb, visible)
    dec = bb.decode(tb, enc, visible, masked)
    assert dec.shape == tb.tokens.shape
    dec[masked].pow(2).mean().backward()
    assert bb.mask_token.grad is not None
    assert all(p.grad is not None for p in bb.encoder.parameters())


def test_local_features_cover_exactly_the_grouped_pixels():
    bb = tiny()
    points, lengths, (event, slot) = bb.points_from(batch().voxels)
    tb = bb.tokenize(points, lengths)
    local, covered = bb.local_features(tb, points.shape[1])
    g = tb.groups
    rows = g.idx[g.point_mask & g.emb_mask.unsqueeze(-1)]
    ev = torch.arange(g.idx.shape[0]).view(-1, 1, 1).expand_as(g.idx)[
        g.point_mask & g.emb_mask.unsqueeze(-1)
    ]
    want = torch.zeros_like(covered)
    want[ev, rows] = True
    assert torch.equal(covered, want)
    assert (local[~covered] == 0).all()
    assert (local[covered].abs().sum(-1) > 0).all()


# ------------------------------------------------------------------- extraction


def cfg_model() -> dict:
    """An `SslModule` over the tiny backbone, the shape `wcfm train model=dino
    model/backbone=polarmae` would write: a masked view, an EMA teacher, no injection."""
    return {
        "_target_": "wcfm.model.modules.SslModule",
        "backbone": {"_target_": "wcfm.model.backbones.PolarMAEBackbone", **TINY},
        "terms": {
            "dino": {
                "_target_": "wcfm.model.terms.DinoTerm",
                "score_injected": False,
                "proj_head": {"hidden_dim": 16, "output_dim": 8, "n_layers": 2},
            }
        },
        "augment": {
            "_target_": "wcfm.model.augment.Augment",
            "masker": {"_target_": "wcfm.model.augment.PixelMasker", "ratio": 0.5},
        },
        "teacher": {"_target_": "wcfm.model.modules.EmaTeacher", "momentum_start": 0.9},
        "normalize": {
            "_target_": "wcfm.model.augment.FeatureLogTransform",
            "min_val": 1.0,
            "max_val": 200.0,
        },
    }


def write_checkpoint(path):
    from hydra.utils import instantiate

    cfg = {"model": cfg_model()}
    module = instantiate(cfg["model"])
    save_checkpoint(
        path, Checkpoint(epoch=1, step=10, cfg=cfg, model=module.state_dict(), optimizer={}, rng={})
    )
    return module


def test_a_checkpoint_round_trips_through_load_module(tmp_path):
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    module, ck = load_module(ckpt)
    assert isinstance(module.backbone, PolarMAEBackbone) and ck.step == 10
    out = inference_step(module, batch().voxels, ("student", "teacher"), ("local",))
    assert set(out) == {"student", "teacher"}
    assert out["student"].out.feature_tensor.shape[1] == TINY["embed_dim"]


def test_extract_reads_the_backbone_with_its_local_tap(tmp_path):
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    data = []
    for i in range(2):
        b = batch((30, 20, 25), seed=10 + i)
        b.meta["event_key"] = [f"b{i}e{j}" for j in range(3)]
        data.append(b)
    extract(
        ckpt,
        store_root=tmp_path / "f",
        eval_set_root=tmp_path / "e",
        loader=data,
        sources=("student",),
        taps=("local",),
    )
    store = FeatureStore(tmp_path / "f")
    assert store.available() == {("student", OUT_TAP), ("student", "local")}
    prov = store.provenance()
    assert prov.tap_strides == {OUT_TAP: 1, "local": 1}
    feats = np.asarray(store.features("student", OUT_TAP))
    assert feats.shape == (150, TINY["embed_dim"])
    assert np.isfinite(feats).all()
