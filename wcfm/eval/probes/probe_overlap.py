"""Overlap: can a readout tell that a pixel's charge is shared?

A pixel's `pixel_energyfrac` is its leading contributor's share of the deposited energy, so
`overlap = 1 - pixel_energyfrac` is contamination: 0 when one particle owns the pixel outright,
0.5 when two contribute equally. The probe asks a yes/no question about that number -- is this
pixel more contaminated than `t`? -- and scores it exactly like PID: two fixed heads, the
raw-charge input as the number to beat, and the degenerate answers measured on the same pixels.

The threshold is swept rather than chosen. `t = 0.2` is the headline, but 0.1 and 0.3 are scored
alongside it, so a reader can see whether the answer depends on where the line was drawn. Each
threshold is its own task with its own prevalence and its own chance level, all of which are
reported.

Scored on the natural validation population rather than a balanced one, which is available here
and is not for PID: the contaminated class is 14% of truth pixels rather than 0.6%, so precision
is not a prior artefact. It also closes a confound -- contamination rates differ 4x across types
(Blip 8.7%, DeltaRay 35.4%), but every type is minority-contaminated, so on the natural
population a head that knows only the particle type scores F1 = 0. On a balanced pool it would
not, and some of the score would be PID leaking in.

Only pixels carrying truth take part; contamination is undefined without it.

The pools are read rather than drawn. `overlap_val` is the natural validation population and
`overlap_train_<i>` is the balanced training pool for `THRESHOLDS[i]` -- both drawn at
extraction by `wcfm.eval.pools` with this module's constants, which `PoolSpec.check` verifies
before anything is scored.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..pools import overlap_contamination, truth_mask
from ..taxonomy import PID_HEADLINE, pid_name
from .features import Features, load_features, raw_charge
from .linear_heads import fit_mlp, fit_svm
from .results import run_header, run_label, write_json

__all__ = ["THRESHOLDS", "binary_scores", "main", "overlap_metric", "run_one"]

# Swept thresholds on `overlap`; the first is the headline. Contamination above ~0.5 is rare
# (1.6% of truth pixels), so the sweep stays in the range where both classes are well populated.
THRESHOLDS = (0.2, 0.1, 0.3)


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision, recall and F1 for the positive (contaminated) class.

    Degenerate cases score 0.0 rather than NaN: a head that never predicts the positive class
    has no precision to speak of, and 0 is the honest reading of it -- the same convention
    `probe_pid` uses per class.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(np.mean(y_pred == y_true)),
        "predicted_positive_rate": float(np.mean(y_pred == 1)),
    }


def trivial_scores(y_true: np.ndarray, seed: int) -> dict:
    """What the degenerate answers score on this exact population.

    `majority` is always "pure" here, since contamination is the minority at every swept
    threshold -- so it scores F1 = 0 by construction, and its accuracy is just the prevalence
    restated. `uniform` guesses at random. Both are the same arithmetic as the real scores.
    """
    rng = np.random.RandomState(seed + 7)
    maj = int(np.mean(y_true) > 0.5)
    return {
        "majority": {
            "predicts": "contaminated" if maj else "pure",
            **binary_scores(y_true, np.full(len(y_true), maj)),
        },
        "uniform": binary_scores(y_true, rng.randint(0, 2, len(y_true))),
    }


def _fit_and_score(Xtr, ytr, Xva, yva, seed: int, device: str):
    """Both fixed heads on one (train, val) pair. Returns `(scores, mlp predictions)`.

    The predictions come back so the per-type breakdown can slice the very predictions the
    headline scored, instead of refitting an identical head.
    """
    from sklearn.preprocessing import StandardScaler

    # Fitted on the training rows only -- the same step every other head gets. Without it the
    # raw baseline's channel/tick columns of magnitude 1e3 go straight into Adam at lr 5e-3 and
    # the head collapses to a constant.
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    out = {"svm": binary_scores(yva, fit_svm(Xtr_s, ytr, Xva_s, seed))}
    pred, _ = fit_mlp(Xtr_s, ytr, Xva_s, n_classes=2, seed=seed, device=device)
    out["mlp"] = binary_scores(yva, pred)
    return out, pred


def overlap_metric(fx: Features, raw: np.ndarray, seed: int, device: str) -> dict:
    """Score the contaminated/pure call at every swept threshold."""
    fx.pool_spec.check("probe_overlap", overlap_thresholds=THRESHOLDS)
    per_class = fx.pool_spec.overlap_train_per_class

    ov = overlap_contamination(fx.truth["pixel_labels"], fx.truth["pixel_energyfrac"])
    tm = truth_mask(fx.truth["pixel_labels"])
    pid = fx.truth["pixel_labels"].astype(np.int64)

    # One validation population for every threshold, drawn uniformly at extraction time so it
    # carries the natural prevalence, and shared across thresholds so the sweep compares tasks
    # rather than samples.
    va_all = fx.pool("overlap_val")
    if len(va_all) == 0:
        return {"error": "no truth pixels on one side of the split"}

    res = {
        "thresholds": [float(t) for t in THRESHOLDS],
        "headline_threshold": float(THRESHOLDS[0]),
        "n_truth_pixels": int(tm.sum()),
        "train_per_class": int(per_class),
        "n_val": int(len(va_all)),
        "val_population": "natural",
    }

    Xva = {
        "feat": np.asarray(fx.feat[va_all], dtype=np.float32),
        "raw": np.asarray(raw[va_all], dtype=np.float32),
    }

    sweep = {}
    headline_pred = headline_y = None
    for i, t in enumerate(THRESHOLDS):
        yva = (ov[va_all] > t).astype(np.int64)
        y_all = (ov > t).astype(np.int64)
        # The TRAINING pool is balanced (rule 3): the head must not learn the prior, but the
        # score must face the real one. Drawn at extraction against this same threshold.
        tr = fx.pool(f"overlap_train_{i}")
        if len(tr) == 0:
            sweep[f"{t:g}"] = {"error": "empty training pool at this threshold"}
            continue
        ytr = y_all[tr]
        entry = {
            "prevalence_val": float(yva.mean()),
            "n_train": int(len(tr)),
            "train_counts": {
                "pure": int((ytr == 0).sum()),
                "contaminated": int((ytr == 1).sum()),
            },
            "chance": trivial_scores(yva, seed),
        }
        if ytr.min() == ytr.max() or yva.min() == yva.max():
            entry["error"] = "one class empty at this threshold"
            sweep[f"{t:g}"] = entry
            continue
        for src in ("feat", "raw"):
            X = fx.feat if src == "feat" else raw
            heads, pred = _fit_and_score(
                np.asarray(X[tr], dtype=np.float32), ytr, Xva[src], yva, seed, device
            )
            for head, vals in heads.items():
                entry[f"{head}_{src}"] = vals
            if t == THRESHOLDS[0] and src == "feat":
                headline_pred, headline_y = pred, yva
        for head in ("svm", "mlp"):
            entry[f"delta_f1_{head}"] = entry[f"{head}_feat"]["f1"] - entry[f"{head}_raw"]["f1"]
        sweep[f"{t:g}"] = entry
    res["sweep"] = sweep

    # Headline, flat, so `compare` can reach it without knowing the sweep shape.
    h = sweep[f"{THRESHOLDS[0]:g}"]
    if "error" not in h:
        for head in ("svm", "mlp"):
            res[f"f1_{head}_feat"] = h[f"{head}_feat"]["f1"]
            res[f"f1_{head}_raw"] = h[f"{head}_raw"]["f1"]
            res[f"delta_f1_{head}"] = h[f"delta_f1_{head}"]
        res["recall_mlp_feat"] = h["mlp_feat"]["recall"]
        res["precision_mlp_feat"] = h["mlp_feat"]["precision"]
        res["prevalence_val"] = h["prevalence_val"]

    # Per particle type, at the headline threshold. The base rate is what makes these readable:
    # recall at a 35% prior (DeltaRay) means something different from recall at 9% (Blip).
    if headline_pred is not None:
        res["per_type"] = _per_type(pid[va_all], headline_y, headline_pred)
    return res


def _per_type(pid_va: np.ndarray, yva: np.ndarray, pred: np.ndarray) -> dict:
    """Recall and F1 within each particle type, at the headline threshold.

    Refits nothing: this slices the very predictions the headline scored, so it cannot disagree
    with it. Types with too few validation pixels report only their count, rather than a number
    resting on a handful of rows.
    """
    out = {}
    for c in PID_HEADLINE:
        m = pid_va == c
        entry = {"n": int(m.sum())}
        if m.sum() >= 500:
            entry["base_rate"] = float(yva[m].mean())
            entry.update(binary_scores(yva[m], pred[m]))
        out[pid_name(c)] = entry
    return out


def run_one(store_root: Path, args) -> dict:
    fx = load_features(store_root, source=args.source, tap=args.tap)
    fx.require("pixel_labels", "pixel_energyfrac")
    print(f"\n=== {run_label(store_root, args.source)} ===")

    entry = run_header(fx, fx.pool_spec.seed, fx.pool_spec.overlap_train_per_class)
    entry["metrics_run"] = ["overlap"]

    t0 = time.time()
    m = overlap_metric(fx, raw_charge(fx), args.seed, args.device)
    m["seed"] = int(args.seed)
    entry["overlap"] = m
    if "error" in m:
        print(f"  [error] {m['error']}")
        return entry

    print(f"  truth pixels={m['n_truth_pixels']}  scored={m['n_val']} (natural)")
    for t in (f"{x:g}" for x in THRESHOLDS):
        s = m["sweep"][t]
        if "error" in s:
            print(f"  t={t:>4}  skipped: {s['error']}")
            continue
        print(
            f"  t={t:>4}  prevalence {s['prevalence_val']:.4f}  "
            f"F1 mlp {s['mlp_feat']['f1']:.4f} (raw {s['mlp_raw']['f1']:.4f}, "
            f"delta {s['delta_f1_mlp']:+.4f})  svm {s['svm_feat']['f1']:.4f}  "
            f"recall {s['mlp_feat']['recall']:.4f}  "
            f"precision {s['mlp_feat']['precision']:.4f}"
        )
    print(f"  [{time.time() - t0:.0f}s]")
    return entry


def main(argv: list[str] | None = None) -> int:
    from .runner import probe_argv

    args = probe_argv("overlap", argv, default_out="overlap.json")
    results = {}
    for store in args.features:
        results[run_label(store, args.source)] = run_one(Path(store), args)
        write_json(results, args.out)
    write_json(results, args.out)
    print(f"\nwrote {args.out}")
    return 0
