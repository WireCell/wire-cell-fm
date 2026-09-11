"""``SslModule`` end to end on a CPU, over the real ``Trainer``, with ``LinearBackbone`` where the
sparse convolutions would be. Every property below is about the module's *orchestration* --
which pairs reach the loss, what gets injected, how weights compose, what the checkpoint
holds, the order of normalisation and masking -- none of which needs a sparse convolution.

The pair-plan checks port ``test_masked_view_pairing.py`` from the old repo, which needed a
GPU only because the old model could not be built without one.
"""

from __future__ import annotations

import dataclasses
import inspect
import random

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")
pytest.importorskip("lightning_fabric")

import torch  # noqa: E402

from wcfm.engine.protocol import StepContext  # noqa: E402
from wcfm.engine.trainer import Trainer  # noqa: E402
from wcfm.model.augment import (  # noqa: E402
    Augment,
    BlockMasker,
    Cropper,
    FeatureLogTransform,
    RegionMasker,
)
from wcfm.model.modules import EmaTeacher, NoTeacher, SslModule  # noqa: E402
from wcfm.model.terms import ChargeTerm, DinoTerm  # noqa: E402

from .fake_backbone import (  # noqa: E402
    LinearBackbone,
    batch_loader,
    coord_label,
    make_batch,
    rows_of,
)
from .test_engine import cfg, cpu_fabric  # noqa: E402

pytestmark = pytest.mark.stack

W, H = 64, 48


def augment(*, crop=True, mask=True, n_global=1, n_local=2) -> Augment:
    random.seed(0)
    torch.manual_seed(0)
    cropper = (
        Cropper(
            image_w=W,
            image_h=H,
            n_global=n_global,
            n_local=n_local,
            min_active_pixels=5,
            blur_sigma_px=2.0,
        )
        if crop
        else None
    )
    masker = BlockMasker(ratio=0.5, win_ch=2, win_tick=2) if mask else None
    return Augment(cropper=cropper, masker=masker)


def build(preset: str, *, normalize=None, teacher=True, **aug) -> SslModule:
    torch.manual_seed(0)
    terms = {}
    if preset in ("dino", "hybrid", "hybrid_charge"):
        terms["dino"] = DinoTerm(
            score_injected=preset != "dino",
            proj_head={"hidden_dim": 16, "output_dim": 8, "n_layers": 2},
        )
    if preset in ("hybrid_charge", "charge"):
        terms["charge"] = ChargeTerm(weight=0.1)
    return SslModule(
        backbone=LinearBackbone(in_dim=1, hidden=8, out_dim=8),
        terms=terms,
        augment=augment(**aug),
        teacher=EmaTeacher(0.9, 1.0) if teacher else NoTeacher(),
        normalize=normalize,
    )


def ctx_for(module, step=0, total=10) -> StepContext:
    return StepContext(
        step=step,
        epoch=1,
        device=torch.device("cpu"),
        module=module,
        all_reduce=lambda x, **k: x,
        is_last_microstep=True,
        extra={"total_iters": total, "epoch_len": total, "world_size": 1},
    )


def batch(**kw):
    return make_batch((60, 50), width=W, height=H, blob=True, **kw)


# ------------------------------------------------------------------------ the pair plan


def test_hybrid_keeps_one_same_index_pair_per_global():
    """attn_mae at n_global=1: 2 local pairs + (0, 0); at n_global=2: 4 views x 2 globals."""
    m = build("hybrid", n_global=1, n_local=2)
    plan = m.augment(batch())
    dino = m.terms["dino"]
    assert dino.total_contrib(plan) == 3
    assert all(dino.runs_on(plan, k) for k in range(3))
    m2 = build("hybrid", n_global=2, n_local=2)
    assert m2.terms["dino"].total_contrib(m2.augment(batch())) == 8


def test_plain_dino_skips_the_same_index_pair_and_the_lone_global_view():
    """With one global and no injection, view 0 pairs with nothing and is not encoded at all."""
    m = build("dino", n_global=1, n_local=2)
    plan = m.augment(batch())
    dino = m.terms["dino"]
    assert dino.total_contrib(plan) == 2
    assert not dino.runs_on(plan, 0) and dino.runs_on(plan, 1) and dino.runs_on(plan, 2)


def test_masking_off_keeps_skipping_even_with_score_injected():
    """The same-index pair is a prediction task only when tokens were actually placed, so a
    plan with no masker skips it whatever the term was told. (The module refuses this
    configuration outright -- tested below -- so the term's own logic is asserted here.)"""
    term = DinoTerm(score_injected=True)
    plan = augment(mask=False, n_global=1, n_local=2)(batch())
    assert not plan.masked and term.total_contrib(plan) == 2 and not term.runs_on(plan, 0)


def test_a_single_uncropped_view_keeps_its_only_pair():
    m = build("hybrid", crop=False)
    assert m.terms["dino"].total_contrib(m.augment(batch())) == 1


# ------------------------------------------------------------------- one training step


def test_hybrid_scores_masked_positions_and_reports_the_split():
    m = build("hybrid")
    out = m.training_step(batch(), ctx_for(m))
    assert out.scalars["n_pairs"] == 3
    assert "loss_masked" in out.scalars and "loss_unmasked" in out.scalars
    assert out.scalars["loss_masked"] > 0 and torch.isfinite(torch.tensor(out.scalars["loss"]))


def test_plain_dino_reports_no_masked_split():
    m = build("dino")
    out = m.training_step(batch(), ctx_for(m))
    assert "loss_masked" not in out.scalars and out.scalars["n_pairs"] == 2


def test_hybrid_plus_charge_composes_both_terms_and_the_weights():
    """``loss == w_dino * mean_dino + w_charge * mean_charge``, exactly as the plan states."""
    m = build("hybrid_charge")
    out = m.training_step(batch(), ctx_for(m))
    s = out.scalars
    assert "loss_charge" in s and "loss_dino" in s
    assert s["loss"] == pytest.approx(1.0 * s["loss_dino"] + 0.1 * s["loss_charge"], rel=1e-5)


def test_dino_excludes_positions_another_term_injected_when_not_scoring_them(monkeypatch):
    """hybrid+charge with score_injected=false: charge injects the masked coordinates, so they
    ARE in the student output; the dino term must filter them out rather than rely on their
    absence. Observed on the rows the loss actually receives."""
    import wcfm.model.terms.dino as dino_mod

    seen: list[int] = []
    orig = dino_mod.PixelDINOLoss.forward

    def spy(self, s, s_bb, t, counts, is_masked=None):
        seen.append((int(s.shape[0]), is_masked))
        return orig(self, s, s_bb, t, counts, is_masked=is_masked)

    monkeypatch.setattr(dino_mod.PixelDINOLoss, "forward", spy)
    m = SslModule(
        backbone=LinearBackbone(in_dim=1, hidden=8, out_dim=8),
        terms={
            "dino": DinoTerm(
                score_injected=False, proj_head={"hidden_dim": 16, "output_dim": 8, "n_layers": 2}
            ),
            "charge": ChargeTerm(weight=0.1),
        },
        augment=augment(crop=False),
        teacher=EmaTeacher(0.9, 1.0),
    )
    b = batch()
    n_total = b.voxels.coordinate_tensor.shape[0]
    out = m.training_step(b, ctx_for(m))
    assert "loss_charge" in out.scalars, "charge ran and therefore injected"
    assert "loss_masked" not in out.scalars, "dino did not score the injected rows"
    ((n_rows, tag),) = seen
    assert tag is None, "the tag is consumed by the filter, not passed to the loss"
    assert n_rows < n_total, "the injected (masked) rows were removed before the loss"


def test_normalisation_runs_before_the_masker_so_the_charge_target_is_in_log_space():
    """ADR 0003, asserted: the masker's target equals the transform of the raw charge at the
    masked coordinates. Move the transform into the loader and this fails."""
    tf = FeatureLogTransform(min_val=3.75, max_val=83861.2)
    m = build("hybrid_charge", crop=False, normalize=tf)
    b = batch()
    raw: dict[tuple[int, int, int], float] = {}
    for bb in range(2):
        s_, e_ = int(b.voxels.offsets[bb]), int(b.voxels.offsets[bb + 1])
        for (x, y), v in zip(
            b.voxels.coordinate_tensor[s_:e_].tolist(),
            b.voxels.feature_tensor[s_:e_, 0].tolist(),
            strict=True,
        ):
            raw[(bb, int(x), int(y))] = float(v)  # keyed per image: the two blobs overlap
    captured = {}
    orig = m.augment

    class Recorder:
        cropper, masker = orig.cropper, orig.masker
        n_global = orig.n_global

        def __call__(self, batch_):
            plan = orig(batch_)
            captured["plan"] = plan
            return plan

    m.augment = Recorder()
    m.training_step(b, ctx_for(m))
    view = captured["plan"].views[0]
    for bb in range(2):
        for (x, y), got in zip(
                view.mask.masked_coords[bb].tolist(),
                view.mask.masked_feats[bb][:, 0].tolist(),
                strict=True,
            ):
            want = float(tf.value(torch.tensor(raw[(bb, int(x), int(y))])))
            assert got == pytest.approx(want, rel=1e-5)


def test_truth_rides_along_with_crop_and_mask():
    """``pixel_labels`` encode their coordinate; after crop + mask they must still name the
    coordinate of the row they sit on."""
    m = build("hybrid")
    plan = m.augment(batch())
    for view in plan.views:
        for bb in range(2):
            labels = torch.as_tensor(view.meta["pixel_labels"][bb])
            assert torch.equal(labels, coord_label(rows_of(view.voxels, bb)))
        assert view.meta["event_key"] == ["ev0", "ev1"], "event-level truth passes through"


# ------------------------------------------------------------------ through the Trainer


def run(tmp_path, preset="hybrid_charge", epochs=2, **over):
    module = build(preset)
    trainer = Trainer(
        cfg(tmp_path, optim__epochs=epochs, metrics__step_cadence=1, **over),
        module,
        fabric=cpu_fabric(),
        loader=batch_loader(steps=4, counts=(60, 50), width=W, height=H, blob=True),
        run_dir=tmp_path / "ssl_run",
    )
    result = trainer.fit()
    return module, trainer, result


def test_the_trainer_runs_hybrid_plus_charge_end_to_end(tmp_path):
    module, trainer, result = run(tmp_path)
    assert result["steps"] == 8 and result["skipped_nonfinite"] == 0
    rows = [
        __import__("json").loads(line)
        for line in (trainer.run_dir / "metrics" / "step.jsonl").read_text().splitlines()
        if line.strip()
    ]
    keys = set().union(*rows)
    assert {
        "loss",
        "loss_dino",
        "loss_charge",
        "kl",
        "teacher_entropy",
        "teacher_momentum",
        "n_pairs",
        "n_voxels",
        "lr",
    } <= keys
    assert all(torch.isfinite(torch.tensor(r["loss"])) for r in rows)


def test_the_centring_buffer_is_in_the_state_dict_and_was_updated(tmp_path):
    module, _, _ = run(tmp_path)
    sd = module.state_dict()
    assert "terms.dino.loss.center" in sd and "terms.dino.loss.center_initialized" in sd
    assert (
        bool(sd["terms.dino.loss.center_initialized"])
        and sd["terms.dino.loss.center"].abs().sum() > 0
    )
    # And a fresh module loads it strictly: nothing lazy, nothing missing.
    fresh = build("hybrid_charge")
    fresh.load_state_dict(sd)
    assert torch.equal(fresh.terms["dino"].loss.center, sd["terms.dino.loss.center"])


def test_the_teacher_is_an_ema_of_the_student_and_takes_no_gradient(tmp_path):
    module, _, _ = run(tmp_path)
    tb, sb = module.teacher_backbone, module.backbone
    assert not any(p.requires_grad for p in tb.parameters())
    assert not any(p.requires_grad for p in module.terms["dino"].teacher_head.parameters())
    s = torch.cat([p.detach().reshape(-1) for p in sb.parameters()])
    t = torch.cat([p.detach().reshape(-1) for p in tb.parameters()])
    assert not torch.equal(s, t), "the teacher lags the student"
    init = build("hybrid_charge")
    t0 = torch.cat([p.detach().reshape(-1) for p in init.teacher_backbone.parameters()])
    assert not torch.equal(t, t0), "the teacher moved"


def test_param_groups_and_taxonomy_name_every_trainable_part():
    m = build("hybrid_charge")
    names = [g["name"] for g in m.param_groups()]
    assert names == ["backbone", "dino_head", "charge_head"]
    n_grouped = sum(sum(p.numel() for p in g["params"]) for g in m.param_groups())
    n_trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n_grouped == n_trainable, "every trainable parameter is in exactly one group"
    tax = m.grad_taxonomy()
    assert set(tax) == {"net", "head", "tokens", "dino_head", "charge_head"}
    assert tax["dino_head"] == ("terms.dino.",)


def test_observables_are_detached_feature_matrices(tmp_path):
    module, _, _ = run(tmp_path, epochs=1)
    obs = module.observables()
    for key in ("student/out", "teacher/out", "student/head", "teacher/head", "dino/center"):
        assert key in obs and not obs[key].requires_grad, key
    assert obs["student/out"].dim() == 2 and obs["student/out"].shape[1] == 8


def test_provenance_names_the_parts():
    p = build("hybrid_charge").provenance()
    assert p["module"] == "SslModule" and p["terms"] == {"dino": "DinoTerm", "charge": "ChargeTerm"}
    assert p["backbone"] == "LinearBackbone" and p["teacher"]


def test_train_mode_keeps_the_teacher_in_eval():
    m = build("hybrid")
    m.train()
    assert m.training and not m.teacher_backbone.training
    assert not m.terms["dino"].teacher_head.training


# --------------------------------------------------------------------------- validation


def test_dino_with_neither_masking_nor_cropping_is_refused():
    with pytest.raises(ValueError, match="compares each view against itself"):
        build("dino", crop=False, mask=False)


def test_score_injected_without_a_masker_is_refused():
    with pytest.raises(ValueError, match="nothing to reinject"):
        build("hybrid", mask=False)


def test_score_injected_without_a_token_is_refused():
    with pytest.raises(ValueError, match="holds tokens only for"):
        SslModule(
            backbone=LinearBackbone(inject_roles=()),
            terms={"dino": DinoTerm(score_injected=True)},
            augment=augment(),
            teacher=EmaTeacher(),
        )


def test_charge_without_a_masker_is_refused():
    with pytest.raises(ValueError, match="needs a masker"):
        SslModule(
            backbone=LinearBackbone(),
            terms={"charge": ChargeTerm()},
            augment=augment(mask=False),
            teacher=NoTeacher(),
        )


def test_a_dino_term_without_a_teacher_is_refused_and_so_is_an_unused_teacher():
    with pytest.raises(ValueError, match="distil from a teacher"):
        build("hybrid", teacher=False)
    with pytest.raises(ValueError, match="no term uses a teacher"):
        SslModule(
            backbone=LinearBackbone(),
            terms={"charge": ChargeTerm()},
            augment=augment(),
            teacher=EmaTeacher(),
        )


def test_region_masking_with_cropping_is_refused():
    """ADR 0004: the grid is defined on the canvas; under cropping the canvas is the crop."""
    random.seed(0)
    torch.manual_seed(0)
    aug = Augment(
        cropper=Cropper(image_w=W, image_h=H, n_global=1, n_local=1),
        masker=RegionMasker(image_w=W, image_h=H, cell_w=8, cell_h=8),
    )
    with pytest.raises(ValueError, match="cannot run on a crop"):
        SslModule(
            backbone=LinearBackbone(),
            terms={"dino": DinoTerm(score_injected=True)},
            augment=aug,
            teacher=EmaTeacher(),
        )


def test_a_charge_only_module_runs_without_a_teacher():
    m = build("charge", crop=False, teacher=False)
    out = m.training_step(batch(), ctx_for(m))
    assert set(out.scalars) >= {"loss", "loss_charge"} and "kl" not in out.scalars


def test_no_terms_is_refused():
    with pytest.raises(ValueError, match="at least one term"):
        SslModule(backbone=LinearBackbone(), terms={}, augment=augment(), teacher=NoTeacher())


# ------------------------------------------------- the schema and the constructors agree


@pytest.mark.parametrize(
    "config_cls,impl",
    [
        ("MinkUNetConfig", "wcfm.model.backbones:MinkUNetAttention"),
        ("PixelMaskerConfig", "wcfm.model.augment:PixelMasker"),
        ("BlockMaskerConfig", "wcfm.model.augment:BlockMasker"),
        ("RegionMaskerConfig", "wcfm.model.augment:RegionMasker"),
        ("CropperConfig", "wcfm.model.augment:Cropper"),
        ("LogTransformConfig", "wcfm.model.augment:FeatureLogTransform"),
        ("EmaTeacherConfig", "wcfm.model.modules:EmaTeacher"),
        ("ChargeTermConfig", "wcfm.model.terms:ChargeTerm"),
        ("DinoTermConfig", "wcfm.model.terms:DinoTerm"),
    ],
)
def test_schema_defaults_equal_constructor_defaults(config_cls, impl):
    """The old repo's eight divergent defaults, prevented: the dataclass in ``config.py`` and
    the class it instantiates must agree on every default they both state."""
    import dataclasses
    import importlib

    from omegaconf import MISSING

    import wcfm.model.config as C

    schema = getattr(C, config_cls)
    mod, name = impl.split(":")
    cls = getattr(importlib.import_module(mod), name)
    sig = inspect.signature(cls.__init__).parameters
    nested = {"proj_head", "cov_penalty", "var_penalty"}
    for f in dataclasses.fields(schema):
        if f.name.startswith("_") or f.name in nested or f.name not in sig:
            continue
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        if default is MISSING or sig[f.name].default is inspect.Parameter.empty:
            continue
        got = sig[f.name].default
        norm = lambda v: list(v) if isinstance(v, (list, tuple)) else v  # noqa: E731
        assert norm(default) == norm(got), (
            f"{config_cls}.{f.name}: schema {default!r} vs constructor {got!r}"
        )
    if config_cls == "DinoTermConfig":
        from wcfm.model.terms.dino import DEFAULT_HEAD

        head = dataclasses.asdict(C.ProjHeadConfig())
        assert head == DEFAULT_HEAD, "the default projection head is stated twice; keep them equal"


def test_dry_run_composes_and_constructs_the_real_model_on_cpu(tmp_path, capsys):
    """``wcfm train --dry-run model=hybrid``: the real backbone, terms and augment stage
    build from the shipped config without a device -- which is what fails on a login node
    instead of after a queue wait."""
    from wcfm.cli.train import main

    code = main(["--dry-run", "model=hybrid", "run.name=dry", f"run.output_root={tmp_path}"])
    assert code == 0, capsys.readouterr().err
    out = capsys.readouterr().out
    assert "SslModule constructed" in out and "parameters" in out


def test_the_step_returns_one_loss_for_every_view_and_backwards_nothing():
    """One forward carrying all views, one loss handed back -- the DINOv1 shape.

    ADR 0001's DDP mechanics are unchanged (several forwards into one backward really does
    over-reduce); what changed is where the view loop lives, and who differentiates. Upstream
    keeps the loop inside the wrapped forward and takes one backward; this module keeps the
    loop inside `forward` and lets the ENGINE take the one backward (ADR 0006).

    A regression to a loss per view is invisible in every scalar, hence the count. That the
    engine then backwards exactly once is asserted in `tests/test_engine.py`.
    """
    m = build("hybrid_charge", n_global=1, n_local=2)
    encodes: list[int] = []
    real_encode = m._encode

    def counting_encode(*a, **kw):
        if not kw.get("teacher"):
            encodes.append(1)
        return real_encode(*a, **kw)

    m._encode = counting_encode  # type: ignore[method-assign]
    out = m.training_step(batch(), ctx_for(m))
    # Without this the test passes vacuously the moment the plan scores a single view.
    assert len(encodes) >= 2, f"only {len(encodes)} student views scored; nothing to prove"
    assert out.loss is not None and out.loss.ndim == 0, (
        f"{len(encodes)} views encoded but the step returned {out.loss!r}: one scalar loss "
        "for the whole step is the contract"
    )
    assert out.loss.requires_grad, "the engine cannot backward a detached loss"
    assert all(p.grad is None for p in m.backbone.parameters()), (
        "the module took a backward of its own; the engine owns that now"
    )
    assert torch.isfinite(torch.tensor(out.scalars["loss"]))


def test_the_forward_is_entered_once_no_matter_how_many_views():
    """DDP arms its reducer once per ``forward()``. The step must enter it once for the
    student -- plus once per GLOBAL view for the teacher, which runs under ``no_grad`` and so
    never arms anything (``DistributedDataParallel.forward`` skips ``prepare_for_backward``
    when grad is disabled). Counting them apart is the only way to see the arming pattern."""
    m = build("hybrid_charge", n_global=1, n_local=3)
    student, teacher_calls = [], []
    real = m.forward

    def counting(voxels, *a, **kw):
        (teacher_calls if kw.get("teacher") else student).append(1)
        return real(voxels, *a, **kw)

    m.forward = counting  # type: ignore[method-assign]
    m.training_step(batch(), ctx_for(m))
    assert len(student) == 1, f"the student forward ran {len(student)} times, must be 1"
    assert len(teacher_calls) == 1, "one global view, so one teacher forward"


def test_per_term_gradients_still_sum_to_the_gradient_the_optimizer_applies():
    """The decomposition `TermGrad` rests on, now that one ``autograd.grad`` per TERM
    replaces one per term per view.

    Summing each term's contribution across views before differentiating is the same number
    -- the gradient of a sum is the sum of the gradients -- but "the same number" is exactly
    the kind of claim that is worth an assertion rather than an argument. The distributed
    suite checks these vectors are identical ACROSS RANKS; nothing checked their value.
    """
    m = build("hybrid_charge", n_global=1, n_local=2)
    ctx = ctx_for(m)
    ctx = dataclasses.replace(ctx, extra={**ctx.extra, "collect_term_gradients": True})
    out = m.training_step(batch(), ctx)

    # The engine differentiates the returned loss; here the test stands in for it. Without
    # this every `.grad` is None and the comparison below is zeros against zeros -- which is
    # exactly how this test failed the first time it met the new contract.
    assert out.loss is not None
    out.loss.backward()

    grads = m.last_term_gradients
    assert grads is not None and set(grads) == {"dino", "charge"}

    shared = [p for p in m.backbone.parameters() if p.requires_grad]
    applied = torch.cat(
        [(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in shared]
    )
    summed = grads["dino"] + grads["charge"]
    assert torch.allclose(summed, applied, rtol=1e-4, atol=1e-6), (
        "the per-term gradients no longer decompose the gradient the optimizer applies: "
        f"max |sum - applied| = {float((summed - applied).abs().max()):.3e}"
    )


def test_the_multi_view_forward_survives_fabrics_mixed_precision_cast(tmp_path):
    """``bf16-mixed`` on the CPU, purely to exercise Fabric's argument cast.

    Under a mixed precision Fabric runs ``_apply_to_collection`` over everything passed to
    ``forward``, to cast floating-point tensors to the compute dtype. That walk treats
    dataclasses specially -- a frozen one raises (``allow_frozen=False``) and a mutable one is
    deepcopied -- so wrapping the per-view arguments in a dataclass broke every mixed-precision
    run while leaving all 638 fp32 tests green. Cluster 2309 found it on a GPU; this finds it
    in seconds, because the cast has nothing to do with the device.

    ``ViewRequest`` is therefore a NamedTuple, and this test is what keeps it one.
    """
    from lightning_fabric import Fabric

    module = build("hybrid_charge", n_global=1, n_local=2)
    trainer = Trainer(
        cfg(tmp_path, optim__epochs=1, metrics__step_cadence=1, run__precision="bf16-mixed"),
        module,
        fabric=Fabric(accelerator="cpu", devices=1, precision="bf16-mixed"),
        loader=batch_loader(steps=2, counts=(60, 50), width=W, height=H, blob=True),
        run_dir=tmp_path / "bf16_run",
    )
    result = trainer.fit()
    assert result["steps"] == 2 and result["skipped_nonfinite"] == 0
