"""`DistillTerm`: the teacher comes out of a checkpoint, and the loss is zero against itself.

The controlling test here is `test_distilling_from_itself_scores_zero_over_every_pixel`. Every
other assertion is about plumbing; that one is the only check that the teacher forward, the
charge transform and the coordinate join all agree, and each of those fails silently on its
own -- a wrong one produces a loss that descends and means nothing.

The checkpoints are written by `test_eval_extract.write_checkpoint`, which instantiates a real
module from a `cfg.model` of `_target_`s and saves that module's own `state_dict`, so a strict
load has to pass.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402

from wcfm.model.augment import Augment, FeatureLogTransform, PixelMasker  # noqa: E402
from wcfm.model.modules import SslModule  # noqa: E402
from wcfm.model.terms import DistillTerm  # noqa: E402

from .fake_backbone import LinearBackbone, make_batch
from .test_eval_extract import write_checkpoint


def module_with(
    term: DistillTerm,
    *,
    min_val: float = 1.0,
    max_val: float = 200.0,
    masker: bool = True,
) -> SslModule:
    """An `SslModule` around one distill term, with the same backbone shape the checkpoints
    carry, so `build()` sizes the projector against a matching width."""
    return SslModule(
        backbone=LinearBackbone(in_dim=1, hidden=8, out_dim=8),
        terms={"distill": term},
        augment=Augment(masker=PixelMasker(ratio=0.5) if masker else None),
        normalize=FeatureLogTransform(min_val=min_val, max_val=max_val),
    )


def teacher_at(tmp_path, name="teacher.pt", *, teacher: bool = False):
    path = tmp_path / name
    write_checkpoint(path, teacher=teacher)
    return path


# ------------------------------------------------------------------------ construction


def test_a_missing_branch_is_refused_and_the_message_names_what_there_is(tmp_path):
    path = teacher_at(tmp_path, teacher=False)
    with pytest.raises(ValueError, match=r"no 'teacher' branch"):
        DistillTerm(checkpoint=str(path), source="teacher")
    # and the branch that does exist loads
    assert DistillTerm(checkpoint=str(path), source="student") is not None


def test_both_branches_load_from_a_run_that_trained_a_teacher(tmp_path):
    path = teacher_at(tmp_path, teacher=True)
    for source in ("student", "teacher"):
        term = DistillTerm(checkpoint=str(path), source=source)
        assert term.teacher.out_dim == 8


def test_no_checkpoint_is_refused_at_construction(tmp_path):
    with pytest.raises(ValueError, match="checkpoint"):
        DistillTerm()


def test_the_projector_is_built_eagerly_against_both_widths(tmp_path):
    term = DistillTerm(checkpoint=str(teacher_at(tmp_path)))
    assert term.head is None
    module_with(term)
    assert term.head is not None


# --------------------------------------------------------------------------- freezing


def test_the_teacher_is_frozen_and_absent_from_the_optimizer_groups(tmp_path):
    from wcfm.config.schema import OptimConfig
    from wcfm.engine.optim import build_optimizer

    term = DistillTerm(checkpoint=str(teacher_at(tmp_path)))
    module = module_with(term)
    assert all(not p.requires_grad for p in term.teacher.parameters())

    optimizer = build_optimizer(module, OptimConfig())
    in_groups = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert not (in_groups & {id(p) for p in term.teacher.parameters()}), (
        "compose_param_groups writes requires_grad back onto the tensors in a declared group, "
        "so a frozen teacher that reaches one is silently unfrozen and trained"
    )


def test_train_leaves_the_teacher_in_eval(tmp_path):
    term = DistillTerm(checkpoint=str(teacher_at(tmp_path)))
    module = module_with(term)
    module.train()
    assert module.training is True
    assert term.teacher.training is False


# ---------------------------------------------------------------------- the charge scale


def test_a_teacher_trained_on_another_charge_scale_is_refused(tmp_path):
    """`cfg_model` pins the teacher's transform at 1.0/200.0, so a run normalising to anything
    else hands it inputs it never saw. `SslModule.validate` finds the hook with getattr."""
    term = DistillTerm(checkpoint=str(teacher_at(tmp_path)))
    assert term.teacher_cfg["model"]["normalize"]["max_val"] == 200.0
    with pytest.raises(ValueError, match=r"normalize\.max_val=200"):
        module_with(term, max_val=999.0)


def test_a_matching_charge_scale_constructs(tmp_path):
    term = DistillTerm(checkpoint=str(teacher_at(tmp_path)))
    assert module_with(term, min_val=1.0, max_val=200.0) is not None


# ------------------------------------------------------------------- the controlling test


def test_distilling_from_itself_scores_zero_over_every_pixel(tmp_path):
    """Teacher and student are the same weights, so the projector is the only thing between
    them. Initialise it to the identity and the cosine distance must be 0 at every pixel, and
    the join must match all of them.

    A non-zero loss here means the teacher forward, the normalisation or the coordinate join
    disagrees with the student's, and nothing else in this file would notice.
    """
    path = teacher_at(tmp_path)
    term = DistillTerm(checkpoint=str(path))
    # No masker: the student encodes every pixel the teacher does, so the join is the identity
    # and the matched count is the whole image.
    module = module_with(term, masker=False)
    # The student is the teacher: same cfg.model, same saved weights.
    module.backbone.load_state_dict(term.teacher.state_dict())
    with torch.no_grad():
        # warpconvnet's Linear wraps an nn.Linear as `.block` and applies it through
        # `x.replace(batched_features=...)`, which is what keeps the coordinates the join needs.
        term.head.block.weight.copy_(torch.eye(8))
        term.head.block.bias.zero_()

    batch = make_batch((40, 35))
    module.normalize(batch.voxels)
    plan = module.augment(batch)
    term.begin_step(plan, None)
    n_voxels = int(plan.views[0].clean.coordinate_tensor.shape[0])

    bundle = module.backbone(plan.views[0].voxels, None, ())
    out = term.compute(bundle, term.head_forward(bundle), 0, plan, None, None)

    assert out.scalars["distill_matched"] == pytest.approx(float(n_voxels)), (
        "the join lost pixels: teacher and student ran on the same coordinates"
    )
    assert out.counts["distill_matched"] == 0, "a matched count is a total, not a mean"
    assert float(out.loss.detach()) == pytest.approx(0.0, abs=1e-5)


def test_the_stash_is_dropped_at_the_end_of_the_step(tmp_path):
    term = DistillTerm(checkpoint=str(teacher_at(tmp_path)))
    module = module_with(term)
    batch = make_batch((20, 18))
    module.normalize(batch.voxels)
    term.begin_step(module.augment(batch), None)
    assert term._teacher_out
    term.on_step_end(None)
    assert term._teacher_out == []


def test_the_checkpoint_carries_the_teacher_and_reloads_strictly(tmp_path):
    """`Trainer._save` writes `self.module.state_dict()` whole, with no exclusion mechanism, so
    the teacher rides along and a strict reload has to find it."""
    term = DistillTerm(checkpoint=str(teacher_at(tmp_path)))
    module = module_with(term)
    state = module.state_dict()
    assert any(k.startswith("terms.distill.teacher.") for k in state)
    module_with(DistillTerm(checkpoint=str(teacher_at(tmp_path)))).load_state_dict(state)


def test_the_config_axis_composes_a_distill_only_run(tmp_path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from wcfm.config.store import register_all

    conf = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "conf")
    GlobalHydra.instance().clear()
    register_all()
    try:
        with initialize_config_dir(config_dir=conf, version_base="1.3"):
            cfg = compose(
                config_name="config",
                overrides=[
                    "model=kd",
                    "model.terms.distill.checkpoint=/gpfs01/nowhere/epoch_40.pt",
                    "run.name=t",
                ],
            )
    finally:
        GlobalHydra.instance().clear()
    assert list(cfg.model.terms) == ["distill"]
    assert cfg.model.terms.distill.checkpoint == "/gpfs01/nowhere/epoch_40.pt"
    assert cfg.model.terms.distill.source == "student"
    assert cfg.model.terms.distill.weight == pytest.approx(1.0)


def test_the_teacher_checkpoint_is_required_by_the_schema(tmp_path):
    """`checkpoint` is MISSING, which is why there is no `conf/model/distill.yaml` preset: a
    preset carrying it could not compose, and every preset is composed by the config suite."""
    import dataclasses

    from omegaconf import MISSING

    from wcfm.model.config import DistillTermConfig

    fields = {f.name: f.default for f in dataclasses.fields(DistillTermConfig)}
    assert fields["checkpoint"] is MISSING
