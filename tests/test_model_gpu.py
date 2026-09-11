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


def _module(precision_flag: list):
    random.seed(0)
    torch.manual_seed(0)

    class Recording(MinkUNetAttention):
        def forward(self, xs, inject=None, taps=()):
            precision_flag.append(torch.is_autocast_enabled("cuda"))
            return super().forward(xs, inject, taps)

    return SslModule(
        backbone=Recording(encoding_range=64),
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


@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
def test_hybrid_plus_charge_trains_on_the_real_backbone_and_teacher_shares_precision(
    shared_tmp, precision
):
    """One epoch of the real model through the Trainer, at both precisions. The teacher runs
    through ``ctx.module(teacher=True)`` so under bf16-mixed it autocasts exactly as the
    student does -- recorded inside the backbone's forward, since the output dtype cannot
    tell you (README, "The GPU suites")."""
    from lightning_fabric import Fabric

    from wcfm.engine.trainer import Trainer

    from .test_engine import cfg

    flags: list[bool] = []
    module = _module(flags)
    fabric = Fabric(accelerator="cuda", devices=1, precision=precision)
    trainer = Trainer(
        cfg(shared_tmp, optim__epochs=1, metrics__step_cadence=1, run__precision=precision),
        module,
        fabric=fabric,
        loader=batch_loader(steps=3, counts=(300, 250), width=W, height=H, blob=True),
        run_dir=shared_tmp / f"gpu_{precision}",
    )
    result = trainer.fit()
    assert result["steps"] == 3 and result["skipped_nonfinite"] == 0
    assert flags, "the backbone never ran"
    expect = precision == "bf16-mixed"
    assert all(f == expect for f in flags), (
        f"autocast seen inside the backbone: {sorted(set(flags))}; every forward, student and "
        f"teacher, must agree with run.precision={precision}"
    )


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

    module = _module([]).to(device)
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
