"""The real backbone on a device: the one thing no CPU test can reach.

Sparse convolutions raise without CUDA, so injection through the actual U-Net -- tokens at
the skips changing where the decoder emits -- and ``SslModule`` over ``MinkUNetAttention``
are asserted here. ``wcfm test --gpu`` runs this suite.
"""

from __future__ import annotations

import random

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402

from wcfm.model.augment import Augment, BlockMasker, Cropper  # noqa: E402
from wcfm.model.backbones import Injection, InjectionGroup, MinkUNetAttention  # noqa: E402
from wcfm.model.modules import EmaTeacher, SslModule  # noqa: E402
from wcfm.model.terms import ChargeTerm, DinoTerm  # noqa: E402

from .fake_backbone import as_pairs, batch_loader, make_batch, rows_of  # noqa: E402

pytestmark = pytest.mark.gpu

W, H = 64, 64


@pytest.fixture
def device():
    return torch.device("cuda")


def scene(device):
    """Active structure, a hole punched in it, the hole as masked coordinates."""
    full = torch.tensor(
        [[c, t] for c in range(0, 64, 2) for t in range(0, 64, 2)], dtype=torch.int32
    )
    inside = (full[:, 0] >= 20) & (full[:, 0] < 40) & (full[:, 1] >= 20) & (full[:, 1] < 40)
    kept, removed = full[~inside], full[inside]
    from wcfm.data.voxels import voxels_from

    xs = voxels_from(
        kept.to(device),
        torch.ones(kept.shape[0], 1, device=device),
        torch.tensor([0, kept.shape[0]]),
    )
    return xs, removed.to(device)


def test_injected_coordinates_appear_in_the_output_and_are_reported(device):
    torch.manual_seed(0)
    m = MinkUNetAttention(encoding_range=64).to(device).eval()
    xs, removed = scene(device)
    inject = Injection([InjectionGroup(t, [removed], "masked", 1) for t in ("enc0", "enc1")])
    with torch.no_grad():
        plain = m(xs, None, ("enc0", "enc1", "dec_half"))
        bundle = m(xs, inject, ("enc0", "enc1", "dec_half"))
    out = as_pairs(bundle.out.coordinate_tensor)
    assert as_pairs(removed) <= out, "every masked coordinate has a feature in the output"
    assert as_pairs(plain.out.coordinate_tensor) == as_pairs(xs.coordinate_tensor)
    assert not (as_pairs(removed) & as_pairs(plain.out.coordinate_tensor))
    # taps report the encoder's own outputs, not the augmented skips
    assert as_pairs(bundle.taps["enc0"].coordinate_tensor) == as_pairs(
        plain.taps["enc0"].coordinate_tensor
    )
    # and the decoder's half-res stage grew by the injected half-res cells
    assert (
        bundle.taps["dec_half"].coordinate_tensor.shape[0]
        > plain.taps["dec_half"].coordinate_tensor.shape[0]
    )
    roles = {(g.tap, g.role, g.stride) for g in bundle.injected.groups}
    assert roles == {("enc0", "masked", 1), ("enc1", "masked", 2)}
    enc1 = next(g for g in bundle.injected.groups if g.tap == "enc1")
    assert as_pairs(enc1.coords[0]) == as_pairs(
        torch.div(removed, 2, rounding_mode="floor")
    ) - as_pairs(plain.taps["enc1"].coordinate_tensor)


def test_the_backbone_refuses_taps_and_roles_it_does_not_have(device):
    m = MinkUNetAttention(encoding_range=64, inject_roles=()).to(device)
    xs, removed = scene(device)
    with pytest.raises(ValueError, match="no tap"):
        m(xs, None, ("nowhere",))
    with pytest.raises(ValueError, match="cannot inject"):
        m(xs, Injection([InjectionGroup("enc0", [removed], "masked", 1)]))


def test_out_dim_and_width_flags_are_honoured(device):
    m = MinkUNetAttention(
        encoding_range=64, widths=(16, 16, 32, 32), out_dim=24, attention=False
    ).to(device)
    xs, _ = scene(device)
    with torch.no_grad():
        b = m(xs, None, ("bottleneck",))
    assert (
        b.out.feature_tensor.shape[1] == 24 and b.taps["bottleneck"].feature_tensor.shape[1] == 32
    )


def test_tokens_injected_at_the_input_pass_through_the_encoder(device):
    torch.manual_seed(0)
    m = MinkUNetAttention(encoding_range=64, inject_at="input").to(device).eval()
    xs, removed = scene(device)
    inject = Injection([InjectionGroup("enc0", [removed], "masked", 1)])
    with torch.no_grad():
        plain = m(xs, None, ("bottleneck",))
        bundle = m(xs, inject, ("enc0", "bottleneck"))
    assert as_pairs(removed) <= as_pairs(bundle.out.coordinate_tensor)
    assert as_pairs(removed) <= as_pairs(bundle.taps["enc0"].coordinate_tensor)
    # the hole holds whole stride-4 cells, which only the tokens can have created
    assert (
        bundle.taps["bottleneck"].coordinate_tensor.shape[0]
        > plain.taps["bottleneck"].coordinate_tensor.shape[0]
    )
    assert {(g.tap, g.role) for g in bundle.injected.groups} == {("enc0", "masked")}


@pytest.mark.parametrize("stage3, side, depth", [(False, 4, 1), (True, 8, 2)])
def test_without_a_decoder_each_pixel_carries_its_cell_on_the_input_rows(
    device, stage3, side, depth
):
    torch.manual_seed(0)
    m = (
        MinkUNetAttention(encoding_range=64, decoder="none", stage3=stage3, bottleneck_depth=depth)
        .to(device)
        .eval()
    )
    xs, removed = scene(device)
    with torch.no_grad():
        out = m(xs).out
    assert torch.equal(out.coordinate_tensor, xs.coordinate_tensor)
    assert torch.equal(out.offsets.cpu(), xs.offsets.cpu())
    _, cell = torch.unique(
        torch.div(xs.coordinate_tensor, side, rounding_mode="floor"), dim=0, return_inverse=True
    )
    first = torch.full((int(cell.max()) + 1,), -1, dtype=torch.long, device=device)
    first.scatter_reduce_(
        0, cell, torch.arange(cell.shape[0], device=device), "amin", include_self=False
    )
    feats = out.feature_tensor
    assert torch.equal(feats, feats[first[cell]]), "pixels of one cell must agree"
    assert not torch.equal(feats, feats[:1].expand_as(feats)), "every cell is the same"
    with pytest.raises(ValueError, match="no tap"):
        m(xs, None, ("dec_full",))
    with pytest.raises(ValueError, match="inject_at='input'"):
        m(xs, Injection([InjectionGroup("enc0", [removed], "masked", 1)]))


def test_stage3_with_the_unet_decoder_emits_per_pixel_on_the_input_rows(device):
    torch.manual_seed(0)
    m = MinkUNetAttention(encoding_range=64, stage3=True, bottleneck_depth=2).to(device).eval()
    xs, removed = scene(device)
    inject = Injection([InjectionGroup("enc0", [removed], "masked", 1)])
    with torch.no_grad():
        out = m(xs).out
        masked = m(xs, inject)
    assert as_pairs(out.coordinate_tensor) == as_pairs(xs.coordinate_tensor)
    assert as_pairs(removed) <= as_pairs(masked.out.coordinate_tensor)
    # unlike decoder='none', the pixels of one stride-8 cell do not all share a vector
    _, cell = torch.unique(
        torch.div(out.coordinate_tensor, 8, rounding_mode="floor"), dim=0, return_inverse=True
    )
    first = torch.full((int(cell.max()) + 1,), -1, dtype=torch.long, device=device)
    first.scatter_reduce_(
        0, cell, torch.arange(cell.shape[0], device=device), "amin", include_self=False
    )
    feats = out.feature_tensor
    assert not torch.equal(feats, feats[first[cell]])


def test_stage3_quarter_taps_share_the_encoders_rows(device):
    m = MinkUNetAttention(encoding_range=64, stage3=True).to(device).eval()
    xs, _ = scene(device)
    with torch.no_grad():
        taps = m(xs, None, ("enc2", "dec_quarter")).taps
    assert as_pairs(taps["dec_quarter"].coordinate_tensor) == as_pairs(
        taps["enc2"].coordinate_tensor
    )
    assert {t.feature_tensor.shape[1] for t in taps.values()} == {m.tap_dim("enc2")}


def test_the_plain_decoder_emits_on_the_input_rows(device):
    m = MinkUNetAttention(encoding_range=64, decoder="plain").to(device).eval()
    xs, removed = scene(device)
    with torch.no_grad():
        b = m(xs, None, ("dec_half",))
    assert as_pairs(b.out.coordinate_tensor) == as_pairs(xs.coordinate_tensor)
    with pytest.raises(ValueError, match="inject_at='input'"):
        m(xs, Injection([InjectionGroup("enc0", [removed], "masked", 1)]))


def _module(**backbone):
    random.seed(0)
    torch.manual_seed(0)
    return SslModule(
        backbone=MinkUNetAttention(encoding_range=64, **backbone),
        terms={
            "dino": DinoTerm(
                score_injected=True, proj_head={"hidden_dim": 32, "output_dim": 16, "n_layers": 2}
            ),
            "charge": ChargeTerm(weight=0.1),
        },
        augment=Augment(
            cropper=Cropper(
                image_w=W, image_h=H, n_global=1, n_local=1, min_active_pixels=5, blur_sigma_px=2.0
            ),
            masker=BlockMasker(ratio=0.4, win_ch=2, win_tick=2),
        ),
        teacher=EmaTeacher(0.9, 1.0),
    )


@pytest.mark.parametrize(
    "inject_at, decoder",
    [("skips", "unet"), ("input", "unet"), ("input", "plain"), ("input", "none")],
)
def test_hybrid_plus_charge_trains_on_the_real_backbone(shared_tmp, inject_at, decoder):
    """Three steps of the real model through the Trainer at `32-true`, the one precision
    `MinkUNetAttention` runs at, for every place a token can enter. The masked token moving off
    its initial value is the gradient having reached it."""
    from lightning_fabric import Fabric

    from wcfm.engine.trainer import Trainer

    from .test_engine import cfg

    module = _module(decoder=decoder, inject_at=inject_at)
    token = module.backbone.tokens["enc0__masked"]
    before = token.detach().clone()
    trainer = Trainer(
        cfg(shared_tmp, optim__epochs=1, metrics__step_cadence=1, run__precision="32-true"),
        module,
        fabric=Fabric(accelerator="cuda", devices=1, precision="32-true"),
        loader=batch_loader(steps=3, counts=(300, 250), width=W, height=H, blob=True),
        run_dir=shared_tmp / f"gpu_{inject_at}_{decoder}",
    )
    result = trainer.fit()
    assert result["steps"] == 3 and result["skipped_nonfinite"] == 0
    assert not torch.equal(token.detach().cpu(), before.cpu()), "the token received no gradient"


def test_region_scene_charge_target_alignment(device):
    """The charge head is read at exactly the masked coordinates; a miss would raise."""
    torch.manual_seed(0)
    b = make_batch((200,), width=W, height=H, blob=True)
    b = b.to(device)
    m = SslModule(
        backbone=MinkUNetAttention(encoding_range=64),
        terms={"charge": ChargeTerm(weight=1.0)},
        augment=Augment(masker=BlockMasker(ratio=0.4, win_ch=2, win_tick=2)),
    ).to(device)

    from wcfm.engine.protocol import StepContext

    ctx = StepContext(
        step=0,
        epoch=1,
        device=device,
        module=m,
        all_reduce=lambda x, **k: x,
    )
    out = m.training_step(b, ctx)
    assert torch.isfinite(torch.tensor(out.scalars["loss_charge"]))
    assert rows_of(b.voxels, 0).shape[0] == 200


def test_term_gradients_decomposes_the_objective_on_the_real_backbone(device):
    """Stage 5's offline gradient probe, against the real sparse-conv forward.

    The framework-side arithmetic -- norms, cosines, accumulation -- is covered on CPU in
    `tests/test_eval_gradients.py` against a fake module. What only a GPU can check is that the
    hook itself runs: a real augment, a real forward per view, and `torch.autograd.grad` over
    the backbone with `retain_graph` while two terms share one view's graph.
    """
    from wcfm.eval.gradients import gradient_report

    module = _module().to(device)
    batch = make_batch(counts=(300, 250), width=W, height=H, blob=True).to(device)

    grads = module.term_gradients(batch)
    assert set(grads) == {"dino", "charge"}, "every active term must contribute a gradient"

    n_backbone = sum(p.numel() for p in module.backbone.parameters())
    for name, g in grads.items():
        assert g.shape == (n_backbone,), f"{name} is not a flat vector over the backbone"
        assert torch.isfinite(g).all(), f"{name} produced a non-finite gradient"
        assert float(g.norm()) > 0, f"{name} produced an all-zero gradient"

    # It must not have left gradients on the parameters: `autograd.grad` returns them rather
    # than accumulating, and a probe that dirtied `.grad` would corrupt a resumed run.
    assert all(p.grad is None for p in module.backbone.parameters())

    # And the whole report runs end to end through the same loader shape extraction uses.
    report = gradient_report(
        module,
        batch_loader(steps=2, counts=(300, 250), width=W, height=H, blob=True),
        device=device,
        max_batches=2,
    )
    assert report["terms"] == ["charge", "dino"]
    assert report["scope"] == "backbone" and report["n_parameters"] == n_backbone
    assert -1.0 <= report["cosine"]["charge|dino"] <= 1.0
