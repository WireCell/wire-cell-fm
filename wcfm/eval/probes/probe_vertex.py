"""Vertex: can a readout tell that a pixel sits close to the interaction point?

The 3-D true vertex is projected into this view and every pixel gets its distance to it in pixel
space. The probe asks a yes/no question about that number -- is this pixel within `r` pixels of
the vertex? -- and scores it exactly like the overlap probe: a fixed head, the raw-charge input
as the number to beat, and a random guess measured on the same pixels.

The radius is swept rather than chosen. `r = 20 px` is the headline, but 10 and 30 are scored
alongside it, so a reader can see whether the answer depends on where the line was drawn. Each
radius is its own task with its own prevalence and its own chance level, all of which are
reported.

Scored on the natural validation population rather than a balanced one: near-vertex pixels are
~5% of the image, so precision means something. Training is balanced, because a head must not
learn the prior, and scoring is not, because the score must face it.

The split is by event. A per-pixel split would leak badly here: distance to the vertex is a
smooth function of position within an event, so pixels of the same event on both sides would
hand either head that event's position-to-distance map.

Every pixel takes part, truthed or not: distance to the vertex is geometry, and a noise pixel
beside the vertex is still beside the vertex.

The projection is not done here. It is truth and geometry only, so it lives in
`wcfm.eval.geometry` and runs at extraction time, where it decides which pixels can be pooled at
all; this module reads the resulting pools and recovers the distances for the labels. `apa` and
`view` are still required, and they come from `Provenance` rather than from a kwarg the caller
repeats per epoch.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..geometry import vertex_distance
from .features import Features, load_features, raw_charge
from .linear_heads import fit_mlp
from .probe_overlap import binary_scores
from .results import run_header, run_label, write_json

__all__ = ["RADII_PX", "main", "run_one", "vertex_metric"]

# Swept radii in pixels; the first is the headline. Channel pitch and tick spacing are different
# physical scales (~0.5 cm and 0.321 cm), so a disc of constant pixel radius is an ellipse in cm:
# 20 px reaches ~10 cm across channels and ~6.4 cm along drift. The sweep is what makes the
# choice of 20 auditable -- if the answer moves between 10 and 30, no single radius should be
# quoted.
RADII_PX = (20.0, 10.0, 30.0)


def _fit_and_score(Xtr, ytr, Xva, yva, seed: int, device: str):
    """The fixed MLP head on one (train, val) pair, scored on the near class."""
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    pred, _ = fit_mlp(
        sc.transform(Xtr), ytr, sc.transform(Xva), n_classes=2, seed=seed, device=device
    )
    return binary_scores(yva, pred)


def vertex_metric(fx: Features, raw: np.ndarray, seed: int, device: str) -> dict:
    """Score the near/far call at every swept radius."""
    fx.pool_spec.check("probe_vertex", vertex_radii_px=RADII_PX)
    per_class = fx.pool_spec.vertex_train_per_class
    t0_ticks = fx.pool_spec.vertex_t0_ticks

    apa, view = fx.provenance.get("apa", -1), fx.provenance.get("view", "")
    if not view or apa < 0:
        return {
            "error": "the extraction recorded no apa/view, so the true vertex cannot be "
            "projected into this view. Re-extract: `wcfm eval extract` reads both off the "
            "run's own data config."
        }

    # Recomputed rather than stored, and it agrees with the draw row for row. The distance
    # of a pixel depends only on its own position and its event's vertex, so evaluating it over
    # a SUBSET of rows gives each surviving row the same value it had over the full eval set --
    # which is what makes this safe under `rows="pooled"`, where `fx.positions`/`fx.offsets`
    # are the subset. The per-event loop still groups correctly because `load_features` rebuilt
    # the CSR, and a row's label therefore cannot disagree with the pool it is in.
    dist, valid, info = vertex_distance(
        positions=fx.positions,
        offsets=fx.offsets,
        vertex_xyz=fx.vertex_xyz,
        apa=apa,
        view=view,
        t0_ticks=t0_ticks,
    )
    if info["n_events_projected"] == 0:
        return {"error": "no event vertex projected inside the wire volume", **info}

    va_all = fx.pool("vertex_val")
    if len(va_all) == 0:
        return {"error": "no projected pixels on one side of the split", **info}

    res = {
        "radii_px": [float(r) for r in RADII_PX],
        "headline_radius_px": float(RADII_PX[0]),
        "t0_ticks_assumed": float(t0_ticks),
        # Interaction vertex only, which is what the containers label. Recorded because a
        # production labelling more vertices per event would be a denser task with a different
        # chance level, and the two must not be read as one number.
        "vertex_kind": "interaction_only",
        "train_per_class": int(per_class),
        "n_val": int(len(va_all)),
        "val_population": "natural",
        "seed": int(seed),
        **info,
    }

    Xva = {
        "feat": np.asarray(fx.feat[va_all], dtype=np.float32),
        "raw": np.asarray(raw[va_all], dtype=np.float32),
    }

    sweep = {}
    for i, r in enumerate(RADII_PX):
        y_all = np.zeros(len(dist), dtype=np.int64)
        y_all[valid & (dist <= r)] = 1
        yva = y_all[va_all]
        tr = fx.pool(f"vertex_train_{i}")
        if len(tr) == 0:
            sweep[f"{r:g}"] = {"error": "empty training pool at this radius"}
            continue
        ytr = y_all[tr]
        # A tighter radius has fewer near pixels to draw on, on both sides of the split. Both
        # counts are recorded because the sweep is only comparable while every radius trains on
        # a full pool and scores on enough positives for precision to mean anything -- at 20 px
        # the near class is ~5% of the image, and 10 px is ~4x thinner than that.
        entry = {
            "prevalence_val": float(yva.mean()),
            "n_val_near": int(yva.sum()),
            "n_train": int(len(tr)),
            "train_counts": {"far": int((ytr == 0).sum()), "near": int((ytr == 1).sum())},
            "train_pool_short": bool(min((ytr == 0).sum(), (ytr == 1).sum()) < per_class),
            # A seeded coin flip on the same pixels. Not something the head is asked to beat --
            # the raw-charge input is what it is measured against -- but a head that has
            # collapsed to a constant cannot clear it, so it is what separates a weak score from
            # a broken one. Same arithmetic as the real scores.
            "chance": binary_scores(
                yva, np.random.RandomState(seed + 7).randint(0, 2, len(yva))
            ),
        }
        if ytr.min() == ytr.max() or yva.min() == yva.max():
            entry["error"] = "one class empty at this radius"
            sweep[f"{r:g}"] = entry
            continue
        for src in ("feat", "raw"):
            X = fx.feat if src == "feat" else raw
            entry[f"mlp_{src}"] = _fit_and_score(
                np.asarray(X[tr], dtype=np.float32), ytr, Xva[src], yva, seed, device
            )
        entry["delta_f1_mlp"] = entry["mlp_feat"]["f1"] - entry["mlp_raw"]["f1"]
        sweep[f"{r:g}"] = entry
    res["sweep"] = sweep

    # Headline, flat, so `compare` can reach it without knowing the sweep shape. Same key names
    # as probe_overlap, because it is the same call on a different question and a reader should
    # not have to learn two spellings.
    h = sweep[f"{RADII_PX[0]:g}"]
    if "error" not in h:
        res["f1_mlp_feat"] = h["mlp_feat"]["f1"]
        res["f1_mlp_raw"] = h["mlp_raw"]["f1"]
        res["delta_f1_mlp"] = h["delta_f1_mlp"]
        res["recall_mlp_feat"] = h["mlp_feat"]["recall"]
        res["precision_mlp_feat"] = h["mlp_feat"]["precision"]
        res["prevalence_val"] = h["prevalence_val"]
    return res


def run_one(store_root: Path, args) -> dict:
    fx = load_features(store_root, source=args.source, tap=args.tap)
    print(f"\n=== {run_label(store_root, args.source)} ===")

    entry = run_header(fx, fx.pool_spec.seed, fx.pool_spec.vertex_train_per_class)
    entry["metrics_run"] = ["vertex"]

    t0 = time.time()
    m = vertex_metric(fx, raw_charge(fx), args.seed, args.device)
    entry["vertex"] = m
    if "error" in m:
        print(f"  [error] {m['error']}")
        return entry

    print(
        f"  scored={m['n_val']} pixels (natural)  "
        f"{m['n_events_projected']} events projected, "
        f"{m['n_events_vertex_outside_volume']} vertices outside volume"
    )
    for r in (f"{x:g}" for x in RADII_PX):
        s = m["sweep"][r]
        if "error" in s:
            print(f"  r={r:>4} px  skipped: {s['error']}")
            continue
        f, w = s["mlp_feat"], s["mlp_raw"]
        print(
            f"  r={r:>4} px  near-vertex rate {s['prevalence_val']:.4f}  "
            f"F1 feat {f['f1']:.4f} (raw {w['f1']:.4f}, "
            f"delta {s['delta_f1_mlp']:+.4f})  "
            f"of those it called near, {100 * f['precision']:.1f}% were; "
            f"of those that were, it found {100 * f['recall']:.1f}%"
        )
        short = (
            f"  [training pool short: {s['train_counts']['near']} near vs "
            f"{m['train_per_class']} asked]"
            if s["train_pool_short"]
            else ""
        )
        print(
            f"            {s['n_val_near']} near pixels scored;  "
            f"coin-flip F1 {s['chance']['f1']:.4f}{short}"
        )
    print(f"  [t0={m['t0_ticks_assumed']} ticks, {time.time() - t0:.0f}s]")
    return entry


def main(argv: list[str] | None = None) -> int:
    from .runner import probe_argv

    args = probe_argv("vertex", argv, default_out="vertex.json")
    results = {}
    for store in args.features:
        results[run_label(store, args.source)] = run_one(Path(store), args)
        write_json(results, args.out)
    write_json(results, args.out)
    print(f"\nwrote {args.out}")
    return 0
