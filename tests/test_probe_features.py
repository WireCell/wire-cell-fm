"""`load_features`: the shield that lets the probe bodies stay the old repo's arithmetic.

The property that matters, and the only one a probe cannot check for itself: **a probe sees one
row space**. Under `rows="pooled"` the store holds a subset of the eval set's pixels while
truth on disk is still full length, so every per-pixel column has to be subset on the way
through and the CSR offsets rebuilt. If that is wrong, nothing has the wrong shape -- the probe
scores different pixels than it thinks and reports a plausible number.

So the tests below build the same extraction twice, once under each row space, and require the
two to agree pixel for pixel wherever they overlap.
"""

from __future__ import annotations

import numpy as np
import pytest

from wcfm.eval.format import EvalSet, FeatureStore, Provenance
from wcfm.eval.probes.features import load_features, raw_charge, raw_charge_kind

N_EVENTS, PER_EVENT, DIM = 6, 20, 8
N_PIXELS = N_EVENTS * PER_EVENT


def _truth(seed: int = 0):
    rng = np.random.RandomState(seed)
    offsets = np.arange(N_EVENTS + 1, dtype=np.int64) * PER_EVENT
    return {
        "pixel_labels": rng.randint(0, 7, N_PIXELS).astype(np.int8),
        "pixel_energyfrac": rng.uniform(0.3, 1.0, N_PIXELS).astype(np.float32),
        "pixel_trackid": rng.randint(0, 5, N_PIXELS).astype(np.int32),
        "labels": rng.randint(0, 4, N_EVENTS).astype(np.int64),
        "vertex_xyz": rng.uniform(-100, 100, (N_EVENTS, 3)).astype(np.float32),
    }, {
        "positions": np.stack(
            [rng.randint(0, 800, N_PIXELS), rng.randint(0, 2000, N_PIXELS)], axis=1
        ).astype(np.int32),
        "charges": rng.uniform(1, 500, N_PIXELS).astype(np.float32),
        "offsets": offsets,
    }


def _build(root, *, rows: str, row_index: np.ndarray, charge_params: dict | None = None):
    """One extraction on disk, in whichever row space, with a pool over the written rows."""
    truth, geometry = _truth()
    es_root = root / "eval_set"
    EvalSet.create(
        es_root,
        id="set-a",
        event_keys=[f"ev{i}" for i in range(N_EVENTS)],
        truth=truth,
        geometry=geometry,
    )
    store_root = root / "epoch3"
    store = FeatureStore(store_root)
    # Row i of the block is eval-set pixel row_index[i]; make that checkable by construction.
    block = np.stack([row_index.astype(np.float32)] * DIM, axis=1)
    store.write_features("student", "out", block)
    store.write_pools(
        row_index=row_index,
        pid_train=np.arange(0, len(row_index), 2, dtype=np.int64),
        pid_val=np.arange(1, len(row_index), 2, dtype=np.int64),
    )
    store.write_provenance(
        Provenance(
            eval_set_id="set-a",
            event_key_hash="h",
            checkpoint="c.pt",
            checkpoint_sha256="s",
            sources=["student"],
            taps=["out"],
            rows=rows,
            apa=0,
            view="W",
            charge_transform_params=charge_params or {},
            extra={"epoch": 3, "module": {"backbone": "MinkUNetAttention"}},
        )
    )
    return store_root, es_root, truth, geometry


def test_all_rows_is_the_identity(tmp_path):
    row_index = np.arange(N_PIXELS, dtype=np.int64)
    store_root, _, truth, geometry = _build(tmp_path, rows="all", row_index=row_index)
    fx = load_features(store_root, "student", verbose=False)

    assert fx.n_pixels == N_PIXELS and fx.n_events == N_EVENTS
    np.testing.assert_array_equal(fx.offsets, geometry["offsets"])
    np.testing.assert_array_equal(fx.truth["pixel_labels"], truth["pixel_labels"])
    np.testing.assert_array_equal(fx.positions, geometry["positions"])
    np.testing.assert_array_equal(fx.labels, truth["labels"])
    # Row i really is pixel i.
    np.testing.assert_array_equal(fx.feat[:, 0], np.arange(N_PIXELS))


def test_pooled_rows_present_one_consistent_row_space(tmp_path):
    """The property. Every column is subset the same way, and the CSR still bounds events."""
    row_index = np.sort(
        np.random.RandomState(1).choice(N_PIXELS, N_PIXELS // 3, replace=False)
    ).astype(np.int64)
    store_root, _, truth, geometry = _build(tmp_path, rows="pooled", row_index=row_index)
    fx = load_features(store_root, "student", verbose=False)

    assert fx.n_pixels == len(row_index)
    np.testing.assert_array_equal(fx.feat[:, 0], row_index)
    # Truth and geometry were subset to exactly those pixels.
    np.testing.assert_array_equal(fx.truth["pixel_labels"], truth["pixel_labels"][row_index])
    np.testing.assert_array_equal(fx.positions, geometry["positions"][row_index])
    np.testing.assert_array_equal(fx.charges, geometry["charges"][row_index])
    # And the rebuilt CSR still slices one event at a time.
    assert fx.offsets[0] == 0 and fx.offsets[-1] == len(row_index)
    for ev in range(N_EVENTS):
        lo, hi = int(fx.offsets[ev]), int(fx.offsets[ev + 1])
        rows = row_index[lo:hi]
        assert np.all(rows // PER_EVENT == ev), f"event {ev} slice leaked into a neighbour"
    np.testing.assert_array_equal(fx.pixel_event, np.asarray(row_index) // PER_EVENT)


def test_the_two_row_spaces_name_the_same_pixels(tmp_path):
    """Score-equivalence: whatever a probe reads for a row, both layouts agree on it."""
    keep = np.sort(
        np.random.RandomState(2).choice(N_PIXELS, N_PIXELS // 2, replace=False)
    ).astype(np.int64)
    a_root, _, _, _ = _build(
        tmp_path / "a", rows="all", row_index=np.arange(N_PIXELS, dtype=np.int64)
    )
    b_root, _, _, _ = _build(tmp_path / "b", rows="pooled", row_index=keep)
    a = load_features(a_root, "student", verbose=False)
    b = load_features(b_root, "student", verbose=False)

    for col in ("pixel_labels", "pixel_energyfrac", "pixel_trackid"):
        np.testing.assert_array_equal(a.truth[col][keep], b.truth[col], err_msg=col)
    np.testing.assert_array_equal(a.positions[keep], b.positions)
    np.testing.assert_array_equal(a.feat[keep, 0], b.feat[:, 0])


def test_an_unsorted_row_index_is_refused(tmp_path):
    """It would leave rows no longer grouped by event, and every per-event probe would mix
    two events' pixels while every array kept its shape."""
    row_index = np.array([5, 3, 40, 7], dtype=np.int64)
    store_root, _, _, _ = _build(tmp_path, rows="pooled", row_index=row_index)
    with pytest.raises(SystemExit, match="not strictly increasing"):
        load_features(store_root, "student", verbose=False)


def test_pools_come_back_in_row_space_and_a_missing_one_raises(tmp_path):
    row_index = np.arange(N_PIXELS, dtype=np.int64)
    store_root, _, _, _ = _build(tmp_path, rows="all", row_index=row_index)
    fx = load_features(store_root, "student", verbose=False)

    assert fx.has_pool("pid_train") and not fx.has_pool("overlap_val")
    assert fx.pool("pid_train").max() < fx.n_pixels
    # Raises rather than redrawing: a fallback draw would be right under `rows="all"` and
    # silently wrong under `rows="pooled"`, and the shapes cannot tell the two apart.
    with pytest.raises(KeyError, match="does not draw its own"):
        fx.pool("overlap_val")


def test_a_missing_branch_or_tap_says_what_is_there(tmp_path):
    store_root, _, _, _ = _build(
        tmp_path, rows="all", row_index=np.arange(N_PIXELS, dtype=np.int64)
    )
    with pytest.raises(SystemExit, match="holds no 'teacher'"):
        load_features(store_root, "teacher", verbose=False)
    with pytest.raises(SystemExit, match="no tap 'enc0'"):
        load_features(store_root, "student", tap="enc0", verbose=False)


def test_raw_charge_uses_the_trained_transform_when_its_parameters_were_recorded(tmp_path):
    row_index = np.arange(N_PIXELS, dtype=np.int64)
    store_root, _, _, geometry = _build(
        tmp_path,
        rows="all",
        row_index=row_index,
        charge_params={"kind": "log", "min_val": 1.0, "max_val": 4000.0},
    )
    fx = load_features(store_root, "student", verbose=False)
    assert raw_charge_kind(fx) == "trained"

    rc = raw_charge(fx)
    assert rc.shape == (N_PIXELS, 3)
    np.testing.assert_array_equal(rc[:, 0], geometry["positions"][:, 0])
    np.testing.assert_array_equal(rc[:, 1], geometry["positions"][:, 1])
    # The charge column is the backbone's own input: FeatureLogTransform, not log10(1+q).
    q = geometry["charges"].astype(np.float64)
    y0, y1 = np.log10(1.0), np.log10(4000.0 + 1.0)
    want = 2.0 * (np.log10(q + 1.0) - y0) / (y1 - y0) - 1.0
    np.testing.assert_allclose(rc[:, 2], want, rtol=1e-6)


def test_raw_charge_falls_back_and_says_so_when_no_parameters_were_recorded(tmp_path, capsys):
    store_root, _, _, _ = _build(
        tmp_path, rows="all", row_index=np.arange(N_PIXELS, dtype=np.int64)
    )
    fx = load_features(store_root, "student", verbose=False)
    assert raw_charge_kind(fx) == "log10_1p"
    raw_charge(fx)
    assert "NOT comparable" in capsys.readouterr().out


def test_provenance_carries_what_the_probes_read(tmp_path):
    store_root, _, _, _ = _build(
        tmp_path,
        rows="all",
        row_index=np.arange(N_PIXELS, dtype=np.int64),
        charge_params={"kind": "log", "min_val": 1.0, "max_val": 4000.0},
    )
    fx = load_features(store_root, "student", verbose=False)
    assert fx.provenance["epoch"] == 3
    assert fx.provenance["backbone_name"] == "MinkUNetAttention"
    # probe_vertex refuses to run without these two.
    assert fx.provenance["apa"] == 0 and fx.provenance["view"] == "W"
    assert fx.provenance["rows"] == "all"
    assert fx.provenance["feat_min_val"] == 1.0


def test_require_names_the_missing_tier(tmp_path):
    store_root, _, _, _ = _build(
        tmp_path, rows="all", row_index=np.arange(N_PIXELS, dtype=np.int64)
    )
    fx = load_features(store_root, "student", verbose=False)
    fx.require("pixel_labels", "pixel_trackid")
    assert not fx.has("pixel_truth_q")
    with pytest.raises(SystemExit, match="pixel_truth_q"):
        fx.require("pixel_truth_q")
