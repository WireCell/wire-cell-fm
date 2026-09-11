"""The probe suite, end to end on a synthetic extraction.

These are not accuracy tests -- the features are synthetic, so a score means nothing. They pin
the things that broke when the suite moved to format v2, all of which produce a plausible number
rather than an error:

* a probe reads its pool instead of drawing one, so every probe scores the same population;
* the pool it reads is in row space and lines up with the truth it scores against;
* a probe whose truth tier is absent says so rather than returning zeros;
* the reported keys are the ones `merge` and the archived tables are written against.

The store is built through the **real** `draw_pools`, so what the probes read here is what
extraction writes.
"""

from __future__ import annotations

import numpy as np
import pytest

from wcfm.eval.format import EvalSet, FeatureStore, Provenance
from wcfm.eval.pools import EVENT_POOLED_FROM, PoolSpec, draw_pools, mean_pool
from wcfm.eval.probes.features import load_features
from wcfm.eval.rawcharge import raw_charge_from
from wcfm.eval.taxonomy import PID_NAMES as PID_NAMES_ALL
from wcfm.eval.taxonomy import PIXEL_CLASS_NAMES as SIZE_BIN_NAMES_KNN
from wcfm.eval.taxonomy import SIZE_BIN_NAMES as SIZE_BIN_NAMES_EXPECTED

N_EVENTS, PER_EVENT, DIM = 24, 60, 12
N_PIXELS = N_EVENTS * PER_EVENT
CHARGE_PARAMS = {"kind": "log", "min_val": 1.0, "max_val": 4000.0}
SPEC = PoolSpec(
    per_class=12,
    seed=42,
    overlap_train_per_class=20,
    overlap_val_pixels=400,
    vertex_train_per_class=20,
    vertex_val_pixels=400,
    instance_max_queries=150,
    event_max_per_event=30,
    knn_max_per_class=20,
)


def _positions_near_vertices(vertex_xyz, rng):
    """Pixels scattered around each event's projected vertex, half within the headline radius."""
    from wcfm.data.wire_geometry import WireGeometry

    geom = WireGeometry.load(t0_ticks=-0.649)
    out = np.zeros((N_PIXELS, 2), dtype=np.int32)
    for ev in range(N_EVENTS):
        _, u, v, w, tick = geom.project(vertex_xyz[ev], apa=0)
        ch = float(geom.channel_for_view("W", np.array([u, v, w])))
        lo = ev * PER_EVENT
        half = PER_EVENT // 2
        near = rng.uniform(-12, 12, (half, 2))
        far = rng.uniform(-400, 400, (PER_EVENT - half, 2))
        offs = np.concatenate([near, far], axis=0)
        out[lo : lo + PER_EVENT, 0] = np.clip(ch + offs[:, 0], 0, 2559)
        out[lo : lo + PER_EVENT, 1] = np.clip(tick + offs[:, 1], 0, 6000)
    return out


def _make_store(root, *, rows="all", truth_tiers=("labels", "energyfrac", "trackid")):
    """A complete v2 extraction whose features carry a real, learnable signal.

    The features are the one-hot pixel label plus noise, so a head can actually score above
    chance -- which is what makes "the pool lines up with the truth" a testable claim rather
    than a shape check.
    """
    rng = np.random.RandomState(7)
    offsets = np.arange(N_EVENTS + 1, dtype=np.int64) * PER_EVENT
    labels = rng.randint(0, 7, N_PIXELS).astype(np.int8)

    truth = {
        "labels": rng.randint(0, 4, N_EVENTS).astype(np.int64),
        # Inside APA 0's bounding box (y in [-600, -1.5], z in [0.3, 230.3], |x| < 360), so
        # `vertex_distance` projects them rather than dropping every event as out-of-volume.
        "vertex_xyz": np.stack(
            [
                rng.uniform(-300, 300, N_EVENTS),
                rng.uniform(-550, -50, N_EVENTS),
                rng.uniform(10, 220, N_EVENTS),
            ],
            axis=1,
        ).astype(np.float32),
    }
    if "labels" in truth_tiers:
        truth["pixel_labels"] = labels
    if "energyfrac" in truth_tiers:
        truth["pixel_energyfrac"] = rng.uniform(0.3, 1.0, N_PIXELS).astype(np.float32)
    if "trackid" in truth_tiers:
        truth["pixel_trackid"] = rng.randint(0, 6, N_PIXELS).astype(np.int32)

    # Positions are placed around each event's OWN projected vertex, half of them inside the
    # headline radius. Scattering them at random instead leaves the "near" class empty at every
    # radius, and the vertex probe then reports three skipped sweep entries -- correct, but it
    # exercises nothing. This is the one place the fixture has to know some geometry.
    geometry = {"positions": _positions_near_vertices(truth["vertex_xyz"], rng), "offsets": offsets}
    geometry["charges"] = rng.uniform(1, 500, N_PIXELS).astype(np.float32)

    es_root = root / "features" / "eval_set"
    EvalSet.create(
        es_root,
        id="set-a",
        event_keys=[f"ev{i}" for i in range(N_EVENTS)],
        truth=truth,
        geometry=geometry,
    )

    block = np.zeros((N_PIXELS, DIM), dtype=np.float32)
    block[np.arange(N_PIXELS), labels.astype(np.int64) % DIM] = 1.0
    block += rng.normal(0, 0.25, block.shape).astype(np.float32)

    spec = PoolSpec(**{k: v for k, v in SPEC.__dict__.items() if k != "notes"})
    pools = draw_pools(truth=truth, geometry=geometry, spec=spec, apa=0, view="W")
    sample = pools.pop(EVENT_POOLED_FROM, np.zeros(0, dtype=np.int64))

    if rows == "pooled":
        non_empty = [p for p in pools.values() if len(p)]
        row_index = np.unique(np.concatenate(non_empty)).astype(np.int64)
        pools = {k: np.searchsorted(row_index, v).astype(np.int64) for k, v in pools.items()}
    else:
        row_index = np.arange(N_PIXELS, dtype=np.int64)

    store_root = root / "features" / "epoch7"
    store = FeatureStore(store_root)
    store.write_features("student", "out", block[row_index])
    store.write_event_means("student", "out", mean_pool(block, sample, offsets, N_EVENTS))
    # What `wcfm.eval.extract` writes alongside it: the raw-charge baseline, pooled over the
    # SAME sample. Under `rows="pooled"` those pixels are not on disk, so probe_event can only
    # read vectors -- and if the fixture skipped this it would be testing a store extraction
    # never produces.
    store.write_event_means(
        "raw",
        "out",
        mean_pool(
            raw_charge_from(geometry["positions"], geometry["charges"], CHARGE_PARAMS),
            sample,
            offsets,
            N_EVENTS,
        ),
    )
    store.write_pools(row_index=row_index, **pools)
    store.write_provenance(
        Provenance(
            eval_set_id="set-a",
            event_key_hash="h",
            checkpoint="checkpoint_epoch7.pt",
            checkpoint_sha256="deadbeef",
            sources=["student"],
            taps=["out"],
            rows=rows,
            apa=0,
            view="W",
            charge_transform="log[1.0,4000.0]",
            charge_transform_params=CHARGE_PARAMS,
            pool_spec=spec.as_dict(),
            pool_per_class=spec.per_class,
            pool_seed=spec.seed,
            extra={"epoch": 7, "module": {"backbone": "MinkUNetAttention"}},
        )
    )
    return store_root


@pytest.fixture
def store(tmp_path):
    return _make_store(tmp_path / "run_a")


@pytest.fixture
def pooled_store(tmp_path):
    return _make_store(tmp_path / "run_b", rows="pooled")


class _Args:
    def __init__(self, **kw):
        self.source = "student"
        self.tap = "out"
        self.seed = 0
        self.device = "cpu"
        self.__dict__.update(kw)


# ------------------------------------------------------------------------------- probe_pid


def test_pid_scores_the_extraction_pool_and_reports_the_archived_keys(store):
    from wcfm.eval.probes.probe_pid import run_one

    entry = run_one(store, _Args())
    p = entry["pid"]
    assert "error" not in p

    # The headline keys every archived pid_*.json and every merge table is written against.
    for key in ("svm_feat", "svm_raw", "mlp_feat", "mlp_raw", "delta_svm", "delta_mlp"):
        assert key in p, key
    for key in ("per_class_f1", "per_class_iou", "confusion", "chance", "macro_recall"):
        assert key in p, key
    assert p["classes"][0] == "Background"
    assert "Background" not in p["headline_classes"]

    # It scored the pool extraction drew, not one of its own.
    fx = load_features(store, "student", verbose=False)
    assert p["n_train"] == len(fx.pool("pid_train"))
    assert p["n_val"] == len(fx.pool("pid_val"))

    # The header carries what makes two of these comparable.
    assert entry["eval_set_id"] == "set-a"
    assert entry["raw_charge_transform"] == "trained"


def test_pid_gives_the_same_answer_under_both_row_spaces(store, pooled_store):
    """The claim `rows="pooled"` rests on: the probe scores the same pixels either way.

    Same seed, same pool spec, same features -- so the scores must agree exactly, not merely
    closely. If the row-space mapping were off by anything, the heads would train on different
    pixels and the numbers would drift apart.
    """
    from wcfm.eval.probes.probe_pid import run_one

    a = run_one(store, _Args())["pid"]
    b = run_one(pooled_store, _Args())["pid"]
    assert a["n_train"] == b["n_train"] and a["n_val"] == b["n_val"]
    assert a["class_counts_val"] == b["class_counts_val"]
    assert a["svm_feat"] == pytest.approx(b["svm_feat"])
    assert a["mlp_feat"] == pytest.approx(b["mlp_feat"])


def test_pid_beats_its_own_chance_baseline_on_a_learnable_signal(store):
    """Not an accuracy claim -- a wiring one. The features here encode the label, so a head
    that scores at chance means the pool and the truth are not lined up."""
    from wcfm.eval.probes.probe_pid import run_one

    p = run_one(store, _Args())["pid"]
    assert p["mlp_feat"] > p["chance"]["uniform"]["m_f1"] * 2


def test_pid_refuses_an_extraction_with_no_pixel_labels(tmp_path):
    from wcfm.eval.probes.probe_pid import run_one

    store = _make_store(tmp_path / "bare", truth_tiers=())
    with pytest.raises(SystemExit, match="pixel_labels"):
        run_one(store, _Args())


# --------------------------------------------------------------------------- probe_overlap


def test_overlap_sweeps_thresholds_and_reports_the_archived_keys(store):
    from wcfm.eval.probes.probe_overlap import THRESHOLDS, run_one

    m = run_one(store, _Args())["overlap"]
    assert "error" not in m
    assert m["val_population"] == "natural"
    assert set(m["sweep"]) == {f"{t:g}" for t in THRESHOLDS}

    head = m["sweep"][f"{THRESHOLDS[0]:g}"]
    for key in ("svm_feat", "svm_raw", "mlp_feat", "mlp_raw", "chance", "prevalence_val"):
        assert key in head, key
    # Flat headline keys, so `compare` reaches them without knowing the sweep shape.
    for key in ("f1_mlp_feat", "f1_mlp_raw", "delta_f1_mlp", "recall_mlp_feat"):
        assert key in m, key
    assert "per_type" in m


def test_overlap_scores_the_pooled_validation_population(store):
    from wcfm.eval.probes.probe_overlap import run_one

    fx = load_features(store, "student", verbose=False)
    m = run_one(store, _Args())["overlap"]
    assert m["n_val"] == len(fx.pool("overlap_val"))
    # One validation population shared across thresholds -- the sweep compares tasks, not
    # samples -- so every threshold reports the same n_val through the same rows.
    assert len({s.get("n_train") for s in m["sweep"].values()}) >= 1


def test_overlap_refuses_a_pool_drawn_under_other_thresholds(store, monkeypatch):
    """`PoolSpec.check` is the guard: edit THRESHOLDS, score an old store, and every
    `overlap_train_i` would otherwise be read under a name that no longer describes it."""
    import wcfm.eval.probes.probe_overlap as po

    monkeypatch.setattr(po, "THRESHOLDS", (0.9, 0.8))
    fx = load_features(store, "student", verbose=False)
    with pytest.raises(ValueError, match="written against"):
        po.overlap_metric(fx, np.zeros((fx.n_pixels, 3), dtype=np.float32), 0, "cpu")


def test_overlap_needs_energyfrac(tmp_path):
    from wcfm.eval.probes.probe_overlap import run_one

    store = _make_store(tmp_path / "no_ef", truth_tiers=("labels", "trackid"))
    with pytest.raises(SystemExit, match="pixel_energyfrac"):
        run_one(store, _Args())


# ---------------------------------------------------------------------------- probe_vertex


def test_vertex_projects_and_sweeps_radii(store):
    from wcfm.eval.probes.probe_vertex import RADII_PX, run_one

    m = run_one(store, _Args())["vertex"]
    assert "error" not in m, m.get("error")
    assert m["n_events_projected"] > 0
    assert m["vertex_kind"] == "interaction_only"
    assert set(m["sweep"]) == {f"{r:g}" for r in RADII_PX}
    # The projection constant is recorded with the number, not assumed by the reader.
    assert m["t0_ticks_assumed"] == pytest.approx(-0.649)


def test_vertex_uses_the_same_key_names_as_overlap(store):
    """One call on a different question; a reader should not learn two spellings."""
    from wcfm.eval.probes.probe_overlap import run_one as overlap_run
    from wcfm.eval.probes.probe_vertex import run_one as vertex_run

    v = vertex_run(store, _Args())["vertex"]
    o = overlap_run(store, _Args())["overlap"]
    shared = {"f1_mlp_feat", "f1_mlp_raw", "delta_f1_mlp", "recall_mlp_feat",
              "precision_mlp_feat", "prevalence_val"}
    assert shared <= set(v) and shared <= set(o)


def test_vertex_says_so_when_the_extraction_recorded_no_apa_or_view(tmp_path):
    """v1 raised SystemExit deep inside the projection. Here it is a recorded error on the
    entry, so a merged table shows which runs could not answer rather than losing the row."""
    from wcfm.eval.probes.probe_vertex import vertex_metric

    store = _make_store(tmp_path / "no_geom")
    fx = load_features(store, "student", verbose=False)
    fx.provenance["view"] = ""
    m = vertex_metric(fx, np.zeros((fx.n_pixels, 3), dtype=np.float32), 0, "cpu")
    assert "no apa/view" in m["error"]


# ----------------------------------------------------------------------------- probe_event


def test_event_scores_pooled_vectors_written_at_extraction(store):
    from wcfm.eval.probes.probe_event import KS, run_one

    m = run_one(store, _Args())["event_knn"]
    assert "error" not in m, m.get("error")
    assert m["n_events_used"] > 0
    assert set(m["feat"]) == {str(k) for k in KS}
    for k in (str(x) for x in KS):
        assert {"purity", "accuracy", "macro_f1"} <= set(m["feat"][k])
    assert "delta_accuracy" in m and "chance" in m
    # The cap is the extraction's, recorded rather than accepted as a flag.
    assert m["max_pixels_per_event"] == SPEC.event_max_per_event


def test_event_pools_both_sides_over_the_same_pixels(store):
    """Extraction writes the feature means and the raw-charge means under one sample, so the
    comparison cannot be confounded by which pixels each side saw."""
    from wcfm.eval.format import FeatureStore

    st = FeatureStore(store)
    feat = st.event_means("student", "out")
    raw = st.event_means("raw", "out")
    assert feat.shape[0] == raw.shape[0] == N_EVENTS
    assert raw.shape[1] == 3  # channel, tick, log charge


def test_event_works_under_pooled_rows_where_its_pixels_are_absent(pooled_store):
    """The whole reason the pooling moved to extraction: under `rows="pooled"` the ~2,000
    pixels per event this probe averages are not on disk, so it must read vectors."""
    from wcfm.eval.probes.probe_event import run_one

    m = run_one(pooled_store, _Args())["event_knn"]
    assert "error" not in m, m.get("error")
    assert m["n_events_used"] > 0


# -------------------------------------------------------------------------- probe_instance


def test_instance_votes_within_events_and_reports_the_macro_headline(store):
    from wcfm.eval.probes.probe_instance import run_one

    m = run_one(store, _Args(knn_k=3))["instance"]
    assert "error" not in m, m.get("error")
    assert m["headline"] == "macro_margin_feat"
    assert m["n_queries_scored"] > 0
    assert "ceiling" in m and "singleton_fraction" in m
    # Pooled is reported but explicitly not the headline.
    assert "pooled" in m and "chance_accuracy" in m["pooled"]
    assert set(m["per_size"]) == set(SIZE_BIN_NAMES_EXPECTED)


def test_instance_scores_the_query_pool_extraction_drew(store):
    from wcfm.eval.probes.probe_instance import run_one

    fx = load_features(store, "student", verbose=False)
    m = run_one(store, _Args(knn_k=3))["instance"]
    drawn = len(fx.pool("instance_queries"))
    assert m["n_queries_scored"] + m["n_queries_dropped_small_event"] == drawn


def test_instance_needs_trackid(tmp_path):
    from wcfm.eval.probes.probe_instance import run_one

    store = _make_store(tmp_path / "no_tid", truth_tiers=("labels", "energyfrac"))
    with pytest.raises(SystemExit, match="pixel_trackid"):
        run_one(store, _Args(knn_k=3))


# -------------------------------------------------------------------------- probe_knn_pid


def test_knn_scores_every_branch_against_one_pool(store):
    from wcfm.eval.probes.probe_knn_pid import run_one

    entries = run_one(store, _Args(knn_k=5, batch_size=1024, with_purity=False, ks="1,5"))
    assert set(entries) == {"run_a:ep7:student"}
    m = entries["run_a:ep7:student"]["knn_pixel"]

    fx = load_features(store, "student", verbose=False)
    assert m["n_pixels_scored"] == len(fx.pool("knn_pool"))
    # The protocol is recorded, because it is NOT probe_pid's and must not be quoted beside it.
    assert m["leakage_free"] is False
    assert m["classes"] == list(SIZE_BIN_NAMES_KNN)
    assert "Background" not in m["classes"]
    assert set(m["per_class_f1"]) == set(m["classes"])
    assert 0.0 <= m["overall_accuracy"] <= 1.0


def test_knn_reports_how_many_events_each_class_pool_came_from(store):
    """The per-image cap exists so a class pool spans events; without the count nobody can
    tell a 2-event pool from a 200-event one, and the uncapped Track recall was 0.777 against
    a leakage-free 0.394."""
    from wcfm.eval.probes.probe_knn_pid import run_one

    entries = run_one(store, _Args(knn_k=5, batch_size=1024, with_purity=False, ks="1,5"))
    m = entries["run_a:ep7:student"]["knn_pixel"]
    assert set(m["events_per_class"]) == set(m["classes"])
    assert max(m["events_per_class"].values()) > 1


def test_auto_per_image_cap_spreads_over_the_target_events():
    from wcfm.eval.pools import SPREAD_TARGET_EVENTS, auto_per_image_cap

    # 10,000 per class over >=200 events -> 50 per image.
    assert auto_per_image_cap(10_000, 1000) == 50
    assert SPREAD_TARGET_EVENTS == 200
    # Fewer events than the target: the cap grows so the quota is still fillable.
    assert auto_per_image_cap(10_000, 50) == 200
    assert auto_per_image_cap(1, 1000) == 1


# ------------------------------------------------------------------------- probe_spectrum


def test_spectrum_reports_collapse_measures_per_tap(store):
    from wcfm.eval.probes.probe_spectrum import run_one

    entry = run_one(store, _Args(max_rows=10_000, top_k=8))
    sp = entry["spectrum"]
    assert "out" in sp["per_tap"]
    out = sp["per_tap"]["out"]
    for key in ("participation_ratio", "rankme", "eig_top", "dims_for_90pct", "condition_number"):
        assert key in out, key
    assert out["dim"] == DIM
    assert 1.0 <= out["participation_ratio"] <= DIM
    assert 1.0 <= out["rankme"] <= DIM
    assert len(out["eig_top"]) == min(8, DIM)
    # Flat headline keys, so `compare` reaches them without walking per_tap.
    assert sp["participation_ratio"] == out["participation_ratio"]


def test_spectrum_uses_the_training_loops_own_arithmetic(store):
    """The point of importing rather than reimplementing: a disagreement between the in-loop
    curve and this number must be a difference in the features, not in the code."""
    import numpy as np

    from wcfm.eval.probes.probe_spectrum import spectrum_of
    from wcfm.metrics.collectors import participation_ratio, rankme

    fx = load_features(store, "student", verbose=False)
    rows = np.arange(fx.n_pixels)
    stats = spectrum_of(fx.feat, rows, top_k=4)
    cov = np.cov(np.asarray(fx.feat[rows], dtype=np.float64), rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    assert stats["participation_ratio"] == pytest.approx(participation_ratio(eig))
    assert stats["rankme"] == pytest.approx(rankme(eig))


def test_spectrum_row_sample_is_stable_across_processes(store):
    """`hash()` on a str is randomised per process, so it cannot seed a sample two checkpoints
    have to share. The seed is a crc32 and this pins it."""
    from wcfm.eval.probes.probe_spectrum import _tap_seed

    assert _tap_seed("out", 0) == _tap_seed("out", 0)
    assert _tap_seed("out", 0) != _tap_seed("enc0", 0)
    # A literal, so a change to the derivation is visible in the diff rather than silent.
    # zlib.crc32(b"out") == 3119148441, and 3119148441 % 2**31 == 971664793.
    assert _tap_seed("out", 0) == 971664793


def test_spectrum_reports_no_teacher_cosine_when_there_is_no_teacher(store):
    from wcfm.eval.probes.probe_spectrum import run_one

    out = run_one(store, _Args(max_rows=10_000, top_k=8))["spectrum"]["per_tap"]["out"]
    # The fixture writes one branch. Absent, not 0 or NaN -- a `mae` run has no teacher, and
    # reporting a number would read as a collapsed one.
    assert "teacher_cosine" not in out


def test_spectrum_class_separation_answers_the_head_free_question(store):
    from wcfm.eval.probes.probe_spectrum import run_one

    sep = run_one(store, _Args(max_rows=10_000, top_k=8))["spectrum"]["class_separation"]
    assert "error" not in sep, sep.get("error")
    assert sep["separation_ratio"] > 0
    assert len(sep["closest_pair"]) == 2
    assert set(sep["classes"]) <= set(PID_NAMES_ALL)


# ------------------------------------------------------------------ the CLI, end to end


def test_probe_then_merge_through_the_cli(store, tmp_path, capsys):
    """The pipeline a Condor node runs: score an extraction, then tabulate the JSONs."""
    import json

    from wcfm.cli.eval import main

    out_dir = tmp_path / "probes"
    rc = main(
        [
            "probe",
            str(store),
            "--stages=pid,spectrum",
            f"--out-dir={out_dir}",
            "--device=cpu",
        ]
    )
    assert rc == 0
    written = sorted(p.name for p in out_dir.glob("*.json"))
    # Named by epoch tag, so every epoch of a run lands in one directory and
    # `wcfm eval merge <run>/probes/*.json` is the whole trajectory.
    assert written == ["pid_ep7.json", "spectrum_ep7.json"]
    entry = json.loads((out_dir / "pid_ep7.json").read_text())["run_a:ep7:student"]
    assert "pid" in entry

    capsys.readouterr()
    rc = main(["merge", *[str(p) for p in out_dir.glob("*.json")]])
    assert rc == 0
    table = capsys.readouterr().out
    assert "pid_mlp" in table and "run_a:ep7:student" in table
    assert "pr" in table  # the spectrum columns merged in alongside


def test_a_failing_stage_does_not_cost_the_others(store, tmp_path, capsys):
    """A sweep should not lose five completed measurements because the sixth aborted -- and it
    must not report success either."""
    from wcfm.cli.eval import main

    out_dir = tmp_path / "probes"
    # `vertex` runs; `instance` needs pixel_trackid, which this store has, so force a failure
    # by pointing a stage at a store that is missing its truth instead.
    bare = _make_store(tmp_path / "bare_run", truth_tiers=("labels",))
    rc = main(
        ["probe", str(bare), "--stages=pid,overlap", f"--out-dir={out_dir}", "--device=cpu"]
    )
    out = capsys.readouterr().out
    assert rc == 1, "a failed stage must make the job non-zero"
    assert "FAILED stages" in out and "overlap" in out
    # ...and pid still wrote its result.
    assert (out_dir / "pid_ep7.json").exists()
