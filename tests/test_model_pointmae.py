"""`PointMaeModule` with the chamfer and energy terms: the contract, one step, thinning, a
short run through the real `Trainer` on the CPU, extraction, and the `polarmae` preset's dry
run. The backbone is the tiny one from `test_model_polarmae`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402

from wcfm.cli.train import main  # noqa: E402
from wcfm.engine.checkpoint import Checkpoint, save_checkpoint  # noqa: E402
from wcfm.engine.protocol import TrainingModule  # noqa: E402
from wcfm.engine.trainer import Trainer  # noqa: E402
from wcfm.eval.extract import OUT_TAP, extract  # noqa: E402
from wcfm.eval.format import FeatureStore  # noqa: E402
from wcfm.eval.loading import inference_step, load_module  # noqa: E402
from wcfm.model.augment.transforms import FeatureLogTransform  # noqa: E402
from wcfm.model.backbones import MinkUNetAttention, PolarMAEBackbone  # noqa: E402
from wcfm.model.modules import PointMaeModule  # noqa: E402
from wcfm.model.modules.ssl import _inference_context  # noqa: E402
from wcfm.model.terms import ChamferTerm, ChargeTerm, EnergyTerm  # noqa: E402

from .fake_backbone import batch_loader, make_batch
from .test_engine import cfg as engine_cfg
from .test_engine import cpu_fabric
from .test_model_polarmae import TINY


def module(**over) -> PointMaeModule:
    torch.manual_seed(0)
    return PointMaeModule(
        backbone=PolarMAEBackbone(**TINY),
        terms={"chamfer": ChamferTerm(), "energy": EnergyTerm()},
        normalize=FeatureLogTransform(1.0, 200.0),
        **over,
    )


def batch(counts=(40, 30, 0, 12), seed=1):
    return make_batch(counts, width=64, height=48, seed=seed)


def ctx(m: PointMaeModule):
    return _inference_context(m, 0, torch.device("cpu"))


# ------------------------------------------------------------------------ the contract


def test_the_module_satisfies_the_protocol_and_refuses_the_wrong_pieces():
    assert isinstance(module(), TrainingModule)
    with pytest.raises(ValueError, match="at least one term"):
        PointMaeModule(backbone=PolarMAEBackbone(**TINY), terms={})
    with pytest.raises(TypeError, match="not a GroupTerm"):
        PointMaeModule(backbone=PolarMAEBackbone(**TINY), terms={"charge": ChargeTerm()})
    with pytest.raises(TypeError, match="PolarMAEBackbone"):
        PointMaeModule(
            backbone=MinkUNetAttention.__new__(MinkUNetAttention), terms={"chamfer": ChamferTerm()}
        )
    with pytest.raises(ValueError, match="mask_ratio"):
        module(mask_ratio=1.0)


def test_one_step_differentiates_every_parameter():
    m = module()
    out = m.training_step(batch(), ctx(m))
    assert out.n_samples == 4
    assert torch.isfinite(out.loss)
    for key in ("loss", "loss_chamfer", "loss_energy", "n_voxels", "n_tokens", "n_masked"):
        assert key in out.scalars, key
    assert out.scalars["n_voxels"] == 82 and out.scalars["n_points"] == 82
    assert 0 < out.scalars["n_masked"] < out.scalars["n_tokens"]
    assert out.scalars["loss"] == pytest.approx(
        out.scalars["loss_chamfer"] + out.scalars["loss_energy"], rel=1e-5
    )
    out.loss.backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert missing == [], "a head outside the forward, or an unused parameter"
    obs = m.observables()
    assert obs["student/tokens"].shape[1] == TINY["embed_dim"]
    assert set(m.grad_taxonomy()) >= {"encoder", "decoder", "chamfer_head", "energy_head"}


def test_the_weights_scale_the_terms():
    torch.manual_seed(0)
    m = PointMaeModule(
        backbone=PolarMAEBackbone(**TINY),
        terms={"chamfer": ChamferTerm(weight=2.0), "energy": EnergyTerm(weight=0.5)},
    )
    out = m.training_step(batch(), ctx(m))
    assert out.scalars["loss"] == pytest.approx(
        2.0 * out.scalars["loss_chamfer"] + 0.5 * out.scalars["loss_energy"], rel=1e-5
    )


def test_thinning_drops_low_charge_and_caps_the_count_in_order():
    m = module(max_points=20, charge_threshold=50.0)
    b = batch()
    raw = b.voxels.feature_tensor[:, 0].clone()
    points, lengths, (event, slot) = m.backbone.points_from(b.voxels)
    keep = torch.ones(points.shape[:2], dtype=torch.bool)
    keep[event, slot] = raw > 50.0
    thinned, new_lengths = m.thin(points.clone(), lengths, keep)
    assert (new_lengths <= 20).all()
    assert (new_lengths <= keep.sum(1)).all()
    for bi in range(4):
        kept = thinned[bi, : new_lengths[bi]]
        orig = points[bi, : lengths[bi]]
        # every kept point is above threshold, and they keep their original relative order
        above = orig[keep[bi, : lengths[bi]]]
        assert all((above == row).all(1).any() for row in kept)
        pos = [int((orig == row).all(1).nonzero()[0]) for row in kept]
        assert pos == sorted(pos)
    out = m.training_step(b, ctx(m))
    assert out.scalars["n_points"] <= 4 * 20 and out.scalars["n_voxels"] == 82


def test_an_empty_batch_element_and_a_tiny_one_do_not_break_the_step():
    m = module()
    out = m.training_step(batch((0, 3)), ctx(m))
    assert torch.isfinite(out.loss)


# ---------------------------------------------------------------------- the loop


def test_a_short_run_through_the_trainer_moves_the_parameters(tmp_path):
    m = module()
    before = m.backbone.mask_token.detach().clone()
    trainer = Trainer(
        engine_cfg(tmp_path, optim__lr=1e-3),
        m,
        fabric=cpu_fabric(),
        loader=batch_loader(steps=3, counts=(30, 20), width=64, height=48),
        run_dir=tmp_path / "run",
    )
    result = trainer.fit()
    assert result["steps"] == 6, "2 epochs x 3 batches"
    assert not torch.equal(m.backbone.mask_token.detach(), before)


# -------------------------------------------------------------- extraction and config


def cfg_model() -> dict:
    return {
        "_target_": "wcfm.model.modules.PointMaeModule",
        "backbone": {"_target_": "wcfm.model.backbones.PolarMAEBackbone", **TINY},
        "terms": {
            "chamfer": {"_target_": "wcfm.model.terms.ChamferTerm"},
            "energy": {"_target_": "wcfm.model.terms.EnergyTerm"},
        },
        "normalize": {
            "_target_": "wcfm.model.augment.FeatureLogTransform",
            "min_val": 1.0,
            "max_val": 200.0,
        },
    }


def test_a_checkpoint_round_trips_and_extracts(tmp_path):
    from hydra.utils import instantiate

    cfg = {"model": cfg_model()}
    m = instantiate(cfg["model"])
    ckpt = tmp_path / "c.pt"
    save_checkpoint(
        ckpt, Checkpoint(epoch=1, step=10, cfg=cfg, model=m.state_dict(), optimizer={}, rng={})
    )
    loaded, ck = load_module(ckpt)
    assert isinstance(loaded, PointMaeModule) and set(loaded.terms) == {"chamfer", "energy"}
    out = inference_step(loaded, batch().voxels, ("student",), ("local",))
    assert out["student"].out.feature_tensor.shape == (82, TINY["embed_dim"])

    data = []
    for i in range(2):
        b = batch((30, 20, 25), seed=10 + i)
        b.meta["event_key"] = [f"b{i}e{j}" for j in range(3)]
        data.append(b)
    extract(ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e", loader=data,
            taps=("local",))
    store = FeatureStore(tmp_path / "f")
    assert store.available() == {("student", OUT_TAP), ("student", "local")}
    assert np.isfinite(np.asarray(store.features("student", OUT_TAP))).all()


def test_the_polarmae_preset_dry_runs(tmp_path, capsys):
    code = main(["--dry-run", "model=polarmae", "run.name=x", f"run.output_root={tmp_path}"])
    assert code == 0
    out = capsys.readouterr().out
    assert "PointMaeModule constructed" in out
