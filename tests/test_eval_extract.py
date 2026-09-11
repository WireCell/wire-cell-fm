"""`wcfm eval extract`: one pass, both branches, and the invariants that make truth joinable.

The module is rebuilt from the checkpoint's own `cfg.model` by `wcfm.eval.loading`, so these
tests write a real v2 checkpoint and read it back the way the CLI does -- there is no path
where a test hands `extract` a module it built itself, because "can the config in the file be
instantiated" is half of what the loader is for.

The control this file turns on is `test_a_reordering_backbone_is_refused`. Every other
assertion about positional truth alignment is uninformative unless a backbone that breaks it
actually fails, and the failure the old repo's check guards against is silent by construction:
the labels stay correct, they just describe different pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402
from warpconvnet.geometry.types.voxels import Voxels  # noqa: E402

from wcfm.engine.checkpoint import Checkpoint, save_checkpoint  # noqa: E402
from wcfm.eval.extract import OUT_TAP, extract  # noqa: E402
from wcfm.eval.format import EvalSet, FeatureStore  # noqa: E402
from wcfm.eval.loading import load_module  # noqa: E402

from .fake_backbone import make_batch

BACKBONE = {
    "_target_": "tests.fake_backbone.LinearBackbone",
    "in_dim": 1,
    "hidden": 8,
    "out_dim": 8,
}


def cfg_model(teacher: bool = True) -> dict:
    """The `cfg.model` a checkpoint carries: `_target_`s all the way down, as hydra wrote it."""
    return {
        "_target_": "wcfm.model.modules.SslModule",
        "backbone": dict(BACKBONE),
        "terms": {
            "dino": {
                "_target_": "wcfm.model.terms.DinoTerm",
                "score_injected": True,
                "proj_head": {"hidden_dim": 16, "output_dim": 8, "n_layers": 2},
            }
        }
        if teacher
        else {"charge": {"_target_": "wcfm.model.terms.ChargeTerm", "weight": 0.1}},
        "augment": {
            "_target_": "wcfm.model.augment.Augment",
            "masker": {"_target_": "wcfm.model.augment.PixelMasker", "ratio": 0.5},
        },
        "teacher": {"_target_": "wcfm.model.modules.EmaTeacher", "momentum_start": 0.9}
        if teacher
        else {"_target_": "wcfm.model.modules.NoTeacher"},
        # min/max interpolate from `data` in a real config; pinned here so the transform
        # is real and the raw-charge ordering test has something that actually changes values.
        "normalize": {
            "_target_": "wcfm.model.augment.FeatureLogTransform",
            "min_val": 1.0,
            "max_val": 200.0,
        },
    }


def write_checkpoint(path, *, teacher: bool = True):
    """Instantiate, then save that very module's `state_dict` -- so a strict load must pass."""
    from hydra.utils import instantiate

    cfg = {"model": cfg_model(teacher)}
    module = instantiate(cfg["model"])
    save_checkpoint(
        path,
        Checkpoint(epoch=3, step=30, cfg=cfg, model=module.state_dict(), optimizer={}, rng={}),
    )
    return module


def batches(n: int = 3, seed: int = 0, per_batch: int = 3):
    """`make_batch`'s tuple is pixels PER IMAGE, so its length is the event count.

    Keys are made unique across batches: `make_batch` names every batch's events `ev0, ev1,
    ...`, which is fine for a single batch and is a duplicated event once several are
    concatenated -- and extraction refuses that, because a repeated event lands on both sides
    of the train/val split.
    """
    counts = (6,) * per_batch
    out = []
    for i in range(n):
        b = make_batch(counts, seed=seed + i, width=16, height=16)
        b.meta["event_key"] = [f"s{seed}b{i}e{j}" for j in range(per_batch)]
        out.append(b)
    return out


def labelled(n: int = 4, seed: int = 0):
    """Batches whose `pixel_labels` are in the PID taxonomy.

    `make_batch` encodes the coordinate in the label (`x * 1000 + y`) so a misaligned gather is
    detectable, which is what its own tests need -- but no such value is in `PID_CLASSES`, so a
    pool drawn over them would be empty and a test of pooling would assert nothing.
    """
    out = batches(n, seed=seed)
    rng = np.random.RandomState(seed)
    for b in out:
        b.meta["pixel_labels"] = [
            rng.randint(0, 7, size=len(a)).astype(np.int8) for a in b.meta["pixel_labels"]
        ]
    return out


# ------------------------------------------------------------------------------ the pass


def test_one_pass_writes_both_branches_and_truth_once(tmp_path):
    ckpt = tmp_path / "checkpoint_epoch3.pt"
    write_checkpoint(ckpt)
    data = batches()

    res = extract(
        ckpt,
        store_root=tmp_path / "feat" / "epoch3",
        eval_set_root=tmp_path / "evalset",
        loader=data,
        eval_set_id="unit",
    )

    assert res.n_events == 9  # 3 batches x 3 events
    store = FeatureStore(tmp_path / "feat" / "epoch3")
    assert store.available() == {("student", OUT_TAP), ("teacher", OUT_TAP)}

    prov = store.provenance()
    assert prov.sources == ["student", "teacher"]
    assert prov.rows == "all"
    assert prov.tap_strides == {OUT_TAP: 1}
    assert prov.extra["epoch"] == 3
    assert prov.extract_seconds > 0

    # Truth lives in the eval set, NOT in the checkpoint's directory: that is the split.
    assert not (tmp_path / "feat" / "epoch3" / "labels.npy").exists()
    es = EvalSet.load(tmp_path / "evalset")
    es.verify(tmp_path / "evalset")
    assert es.n_events == 9
    assert "pixel_labels" in es.truth_arrays

    feats = store.features("student", OUT_TAP)
    assert feats.dtype == np.float16
    assert feats.shape[0] == res.n_pixels
    assert feats.shape[0] == len(es.read(tmp_path / "evalset", "pixel_labels"))


def test_the_student_and_teacher_blocks_differ(tmp_path):
    """An EMA teacher initialised as a copy still diverges once terms build their own heads;
    identical blocks would mean one branch was silently extracted twice."""
    ckpt = tmp_path / "c.pt"
    module = write_checkpoint(ckpt)
    with torch.no_grad():  # move the teacher off the student, as training would
        for p in module.teacher_backbone.parameters():
            p.add_(0.5)
    save_checkpoint(
        ckpt,
        Checkpoint(
            epoch=1, step=1, cfg={"model": cfg_model()}, model=module.state_dict(),
            optimizer={}, rng={},
        ),
    )
    extract(
        ckpt,
        store_root=tmp_path / "f",
        eval_set_root=tmp_path / "e",
        loader=batches(1),
    )
    store = FeatureStore(tmp_path / "f")
    s = np.asarray(store.features("student", OUT_TAP))
    t = np.asarray(store.features("teacher", OUT_TAP))
    assert s.shape == t.shape
    assert not np.allclose(s, t)


def test_raw_charge_is_captured_before_the_transform(tmp_path):
    """`inference_step` normalises in place, so reading charge after it stores log(q) under a
    column named for the ADC. The input here is >= 1.0, so the two are never equal."""
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    data = batches(1)
    raw = data[0].voxels.feature_tensor[:, 0].clone().numpy()

    extract(ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e", loader=data)

    es = EvalSet.load(tmp_path / "e")
    stored = np.asarray(es.read(tmp_path / "e", "charges"))
    assert np.allclose(stored, raw, atol=1e-4)
    assert stored.max() > 2.0  # a log transform would have crushed this


def test_max_images_caps_events_not_batches(tmp_path):
    """v1 took `max_images // batch_size` batches, so batch_size changed the scored set.
    Here the cap lands on events, and a batch is truncated to reach it exactly."""
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)

    a = extract(
        ckpt, store_root=tmp_path / "a", eval_set_root=tmp_path / "ea",
        loader=batches(3, seed=1), max_images=7,
    )
    b = extract(
        ckpt, store_root=tmp_path / "b", eval_set_root=tmp_path / "eb",
        loader=batches(3, seed=1), max_images=100,
    )
    assert a.n_events == 7
    assert b.n_events == 9


# --------------------------------------------------------------- the alignment invariant


class _Reordering(torch.nn.Module):
    """A backbone that returns the input's voxels in a different order -- the failure the
    invariant exists to catch, and one that changes no label and raises no error on its own."""

    TAPS = ()
    TAP_STRIDE: dict = {}

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, xs, inject=None, taps=()):
        bundle = self.inner(xs, inject, taps)
        out = bundle.out
        perm = torch.arange(out.coordinate_tensor.shape[0]).flip(0)
        bundle.out = Voxels(
            batched_coordinates=type(out.batched_coordinates)(
                out.coordinate_tensor[perm], offsets=out.offsets
            ),
            batched_features=type(out.batched_features)(
                out.feature_tensor[perm], offsets=out.offsets
            ),
            offsets=out.offsets,
        )
        return bundle


def test_a_reordering_backbone_is_refused(tmp_path):
    """The control. Without this, every other alignment assertion here is uninformative."""
    ckpt = tmp_path / "c.pt"
    module = write_checkpoint(ckpt)
    module.backbone = _Reordering(module.backbone)

    import wcfm.eval.extract as ex

    real = ex.load_module
    ex.load_module = lambda *a, **k: (module, real(*a, **k)[1])
    try:
        with pytest.raises(RuntimeError, match="reordered or moved"):
            extract(
                ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
                loader=batches(1), sources=("student",),
            )
    finally:
        ex.load_module = real


# ----------------------------------------------------------------------- branches, pools


def test_teacher_is_refused_on_a_run_that_trained_without_one(tmp_path):
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt, teacher=False)
    with pytest.raises(ValueError, match="none of the requested sources"):
        extract(
            ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
            loader=batches(1), sources=("teacher",),
        )


def test_a_missing_branch_is_dropped_rather_than_raising(tmp_path):
    """A sweep asks for both; the runs without a teacher still extract their student."""
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt, teacher=False)
    extract(
        ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
        loader=batches(1), sources=("student", "teacher"),
    )
    assert FeatureStore(tmp_path / "f").provenance().sources == ["student"]


def test_row_index_joins_features_to_truth_in_both_row_spaces(tmp_path):
    """The property a probe relies on: `feat[pool]` and `truth[row_index[pool]]` line up,
    without the probe knowing whether the block was pooled."""
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    data = labelled(4)

    full = extract(
        ckpt, store_root=tmp_path / "full", eval_set_root=tmp_path / "e",
        loader=data, rows="all", pool_per_class=2, sources=("student",),
    )
    pooled = extract(
        ckpt, store_root=tmp_path / "pool", eval_set_root=tmp_path / "e",
        loader=data, rows="pooled", pool_per_class=2, sources=("student",),
    )
    assert pooled.n_rows < full.n_pixels

    es = EvalSet.load(tmp_path / "e")
    labels = np.asarray(es.read(tmp_path / "e", "pixel_labels"))

    for res, root in ((full, "full"), (pooled, "pool")):
        store = FeatureStore(tmp_path / root)
        pools = store.pools()
        feat = np.asarray(store.features("student", OUT_TAP))
        assert feat.shape[0] == res.n_rows == len(pools["row_index"])
        idx = pools["pid_train"]
        # the join, done exactly as a probe would do it
        assert labels[pools["row_index"][idx]].shape == (len(idx),)
        assert feat[idx].shape == (len(idx), 8)

    # and the two row spaces select the SAME pixels, which is what makes the scores comparable
    fp, pp = FeatureStore(tmp_path / "full").pools(), FeatureStore(tmp_path / "pool").pools()
    assert (fp["row_index"][fp["pid_train"]] == pp["row_index"][pp["pid_train"]]).all()
    ff = np.asarray(FeatureStore(tmp_path / "full").features("student", OUT_TAP))
    pf = np.asarray(FeatureStore(tmp_path / "pool").features("student", OUT_TAP))
    assert np.array_equal(ff[fp["pid_train"]], pf[pp["pid_train"]])


def test_pooled_rows_need_pixel_truth(tmp_path):
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    data = [make_batch((4, 4), seed=0, width=16, height=16, pixel_truth=False)]
    with pytest.raises(ValueError, match="needs `pixel_labels`"):
        extract(
            ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
            loader=data, rows="pooled",
        )


# ------------------------------------------------------------------------- the eval set


def test_a_second_checkpoint_reuses_the_eval_set_and_writes_no_truth(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    write_checkpoint(a)
    write_checkpoint(b)
    data = batches(2)

    first = extract(a, store_root=tmp_path / "fa", eval_set_root=tmp_path / "e", loader=data)
    before = sorted(p.name for p in (tmp_path / "e").iterdir())
    second = extract(b, store_root=tmp_path / "fb", eval_set_root=tmp_path / "e", loader=data)

    assert first.eval_set.key_hash == second.eval_set.key_hash
    assert sorted(p.name for p in (tmp_path / "e").iterdir()) == before


def test_a_changed_event_set_is_refused_rather_than_overwritten(tmp_path):
    """Results already written beside a set were scored on its events; silently replacing
    them is the failure the key hash exists to make visible."""
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    extract(ckpt, store_root=tmp_path / "f1", eval_set_root=tmp_path / "e", loader=batches(2))

    other = batches(2, seed=50)  # a different seed gives different keys as well as pixels
    with pytest.raises(ValueError, match="built from different events"):
        extract(ckpt, store_root=tmp_path / "f2", eval_set_root=tmp_path / "e", loader=other)


def test_an_empty_loader_says_so(tmp_path):
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    with pytest.raises(ValueError, match="yielded no events"):
        extract(ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e", loader=[])


# ---------------------------------------------------------------------------- the loader


def test_a_strict_load_refuses_weights_that_do_not_match_the_config(tmp_path):
    """A tolerant load gives a part-initialised backbone whose features look normal."""
    ckpt = tmp_path / "c.pt"
    module = write_checkpoint(ckpt)
    full = module.state_dict()
    dropped = next(iter(full))
    state = {k: v for k, v in full.items() if k != dropped}
    save_checkpoint(
        ckpt,
        Checkpoint(epoch=1, step=1, cfg={"model": cfg_model()}, model=state, optimizer={}, rng={}),
    )
    with pytest.raises(RuntimeError, match="do not match the model"):
        load_module(ckpt)


def test_a_checkpoint_without_a_model_config_cannot_be_rebuilt(tmp_path):
    ckpt = tmp_path / "c.pt"
    save_checkpoint(
        ckpt, Checkpoint(epoch=1, step=1, cfg={}, model={}, optimizer={}, rng={})
    )
    with pytest.raises(ValueError, match="carries no `cfg.model`"):
        load_module(ckpt)


# ------------------------------------------------------------------------- strided taps


class _WithHalf(torch.nn.Module):
    """Adds a stride-2 tap that behaves: one row per distinct coordinate at that stride.

    A strided tap cannot satisfy the stride-1 invariant -- it has fewer rows, on a coarser
    grid -- so copying that check onto it would fire on a correct backbone. What extraction
    checks instead is the per-event count the stride implies, and this is the backbone that
    proves the check passes when it should.
    """

    TAPS = ("half",)
    TAP_STRIDE = {"half": 2}
    prune = 0

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, xs, inject=None, taps=()):
        bundle = self.inner(xs, inject, ())
        if "half" in tuple(taps):
            bundle.taps["half"] = self._half(xs, bundle.out)
        return bundle

    def _half(self, xs, out):
        from wcfm.data.voxels import voxels_from

        coords, feats, counts = [], [], []
        for b in range(len(xs.offsets) - 1):
            s, e = int(xs.offsets[b]), int(xs.offsets[b + 1])
            low = torch.unique(
                torch.div(xs.coordinate_tensor[s:e], 2, rounding_mode="floor"), dim=0
            )
            if self.prune and low.shape[0] > self.prune:
                low = low[: -self.prune]  # the control: drop rows the stride says are there
            coords.append(low)
            feats.append(out.feature_tensor[s : s + low.shape[0]])
            counts.append(low.shape[0])
        offsets = torch.tensor([0, *np.cumsum(counts).tolist()], dtype=torch.int64)
        return voxels_from(torch.cat(coords), torch.cat(feats), offsets)


def _with_half(tmp_path, prune: int = 0):
    ckpt = tmp_path / "c.pt"
    module = write_checkpoint(ckpt)
    wrapper = _WithHalf(module.backbone)
    wrapper.prune = prune
    module.backbone = wrapper

    import wcfm.eval.extract as ex

    real = ex.load_module
    ex.load_module = lambda *a, **k: (module, real(*a, **k)[1])
    return ckpt, real


def test_a_strided_tap_writes_its_own_coordinates(tmp_path):
    ckpt, real = _with_half(tmp_path)
    import wcfm.eval.extract as ex

    try:
        res = extract(
            ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
            loader=batches(2), sources=("student",), taps=("half",),
        )
    finally:
        ex.load_module = real

    store = FeatureStore(tmp_path / "f")
    assert store.available() == {("student", OUT_TAP), ("student", "half")}
    assert store.provenance().tap_strides == {OUT_TAP: 1, "half": 2}

    half = store.features("student", "half")
    coords = store.coords("half")
    assert half.shape[0] == coords.shape[0]
    # fewer rows than pixels, because the grid is coarser -- and NOT joinable to truth
    # positionally, which is why the coordinates are written at all
    assert half.shape[0] <= res.n_pixels
    assert store.coords("half").shape[1] == 2


def test_a_strided_tap_that_prunes_voxels_is_refused(tmp_path):
    """The control for the strided branch of the invariant. A tap whose rows are not the
    strided input has no reconstructable coordinates, and nothing else would notice."""
    ckpt, real = _with_half(tmp_path, prune=1)
    import wcfm.eval.extract as ex

    try:
        with pytest.raises(RuntimeError, match="pruned or invented voxels"):
            extract(
                ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
                loader=batches(1), sources=("student",), taps=("half",),
            )
    finally:
        ex.load_module = real


def test_a_tap_with_no_published_stride_is_refused(tmp_path):
    """Assuming stride 1 is exactly the assumption that makes a truth join silently wrong."""
    ckpt, real = _with_half(tmp_path)
    import wcfm.eval.extract as ex

    try:
        with pytest.raises((ValueError, KeyError)):
            extract(
                ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
                loader=batches(1), sources=("student",), taps=("nosuchtap",),
            )
    finally:
        ex.load_module = real


def test_out_may_not_be_requested_as_a_tap(tmp_path):
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    with pytest.raises(ValueError, match="is the name of the final feature map"):
        extract(
            ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
            loader=batches(1), taps=(OUT_TAP,),
        )


# ------------------------------------------------------------------- the cap, and its limit


def test_the_provenance_records_whether_the_cap_bound(tmp_path):
    """While `max_images` binds, the scored set is the first N of a fixed stream at any batch
    size. When it does not, the reader's own short-tail drop is back in play and the set is
    batch-size dependent again -- so which of the two happened is recorded rather than assumed.
    """
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)

    bound = extract(
        ckpt, store_root=tmp_path / "a", eval_set_root=tmp_path / "ea",
        loader=batches(3), max_images=5, sources=("student",),
    )
    assert bound.provenance.extra["cap_bound"] is True
    assert bound.n_events == 5

    ran_out = extract(
        ckpt, store_root=tmp_path / "b", eval_set_root=tmp_path / "eb",
        loader=batches(3), max_images=10_000, sources=("student",),
    )
    assert ran_out.provenance.extra["cap_bound"] is False


def test_a_truth_column_missing_from_some_batches_is_refused(tmp_path):
    """A short column joins to the wrong events at every row past the gap, with no error."""
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    data = batches(2)
    del data[1].meta["pixel_labels"]  # present in the first batch, absent in the second
    with pytest.raises(ValueError, match=r"truth column 'pixel_labels' has \d+ rows"):
        extract(
            ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
            loader=data, sources=("student",),
        )


def test_a_repeated_event_key_is_refused(tmp_path):
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    data = batches(2)
    data[1].meta["event_key"] = list(data[0].meta["event_key"])
    with pytest.raises(ValueError, match="repeats 3 event key"):
        extract(
            ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
            loader=data, sources=("student",),
        )


def test_pools_are_drawn_under_the_default_row_space_too(tmp_path):
    """`rows='all'` still writes the pools: the row space says what was written, not what was
    scored, and a probe reads `pools.npz` either way."""
    ckpt = tmp_path / "c.pt"
    write_checkpoint(ckpt)
    extract(
        ckpt, store_root=tmp_path / "f", eval_set_root=tmp_path / "e",
        loader=labelled(3), rows="all", pool_per_class=2, sources=("student",),
    )
    pools = FeatureStore(tmp_path / "f").pools()
    # The whole suite's pools, not just probe_pid's -- that is what makes every probe score
    # the same population. `knn_pool` is here because this loader carries `pixel_labels`;
    # overlap/instance/vertex need truth tiers it does not, so they are absent rather than
    # empty. `event_sample` is never here: it is pooled into per-event vectors instead.
    assert set(pools) == {"row_index", "pid_train", "pid_val", "knn_pool"}
    assert len(pools["pid_train"]) > 0
