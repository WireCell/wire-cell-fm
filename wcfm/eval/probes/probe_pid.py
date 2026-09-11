"""PID probe: can a small head read particle type off the frozen features?

Headline is macro-F1 over the six PIDs (Background is pooled and predicted, but not scored --
see `PID_HEADLINE`), reported for two fixed heads (linear SVM, small MLP) and for the raw-charge
input. The difference is what the backbone added.

Both sides of the split are class-balanced, up to `pool_per_class` pixels per PID per side. That
is deliberate: this metric exists to compare checkpoints, and an unbalanced macro-F1 here is
dominated by sampling noise on the rare PIDs.

Everything reported comes from one confusion matrix per (head, source): purity (precision),
efficiency (recall), F1 and IoU, per PID and macro-averaged.

Hyperparameters are fixed and never tuned per checkpoint (see `linear_heads.py`).

The pools are read rather than drawn. `pid_train` and `pid_val` come from `pools.npz`, where
`wcfm.eval.pools` drew them at extraction time, so the population is identical to what every
other probe scores.

Usage:
    wcfm eval probe pid <run>/features/epoch100 --out pid.json
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..taxonomy import PID_CLASSES, PID_HEADLINE, PID_NAMES
from .features import Features, load_features, raw_charge
from .linear_heads import fit_mlp, fit_svm
from .results import run_header, run_label, write_json

__all__ = ["main", "pid_metric", "run_one"]


def _metrics_from_confusion(conf: np.ndarray, headline_idx) -> dict:
    """Per-PID and macro F1 / IoU / purity / efficiency from a confusion matrix.

    `conf` is [true, pred]. Macro averages run over `headline_idx` only. Every quantity
    comes from the same matrix, so every head and every feature source is scored by identical
    arithmetic.
    """
    conf = np.asarray(conf, dtype=np.float64)
    tp = np.diag(conf)
    support = conf.sum(axis=1)  # true count per class
    predicted = conf.sum(axis=0)  # predicted count per class
    fp = predicted - tp
    fn = support - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
        )
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)

    h = list(headline_idx)
    total = conf.sum()
    head_support = support[h].sum()
    return {
        "per_class": {
            "f1": f1,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "support": support,
        },
        "m_f1": float(np.mean(f1[h])),
        "m_iou": float(np.mean(iou[h])),
        "m_precision": float(np.mean(precision[h])),
        "m_recall": float(np.mean(recall[h])),
        # Prevalence-weighted. Over all classes, so Background counts.
        "all_acc": float(tp.sum() / total) if total > 0 else 0.0,
        # Micro-efficiency over the headline PIDs: of pixels truly of one of the six, the
        # fraction labelled correctly. Not `all_acc` restricted.
        "all_acc_headline": (float(tp[h].sum() / head_support) if head_support > 0 else 0.0),
    }


def _named(values: np.ndarray, classes, names) -> dict:
    """Label a per-class vector by class name."""
    classes = list(classes)
    return {names[classes.index(c)]: float(values[classes.index(c)]) for c in classes}


def _trivial_scores(y_true: np.ndarray, headline_idx, k: int, seed: int) -> dict:
    """What the degenerate answers score on this exact population.

    Two of them, because they differ and neither is `1/k` in general:

    `uniform`
        predict a class uniformly at random. On a perfectly balanced pool this gives macro-F1 =
        1/k; on an unbalanced one it does not, which is why it is measured rather than assumed.
    `majority`
        always predict the most common class. Scores 0 on every class but one, so its macro-F1
        depends entirely on *which* class that is: if the majority class is outside the headline
        the whole macro average is 0, and if it is inside, one sixth of a large F1 can clear a
        weak readout. Which class it lands on is recorded rather than assumed.

    Both go through `_metrics_from_confusion`, so they are the same arithmetic as the real
    scores and directly comparable with them.

    On a balanced pool every class has the same count, so the majority is a tie broken
    arbitrarily by `argmax`; `majority_tied` flags that, because the number is then an
    artefact of the tie-break and not worth reading.

    `y_true` must be the contiguous 0..k-1 remapping rather than raw `pixel_labels`, because
    the `bincount(minlength=k)` below assumes it.
    """
    from sklearn.metrics import confusion_matrix

    all_idx = list(range(k))
    if len(y_true) == 0:
        return {}
    counts = np.bincount(y_true, minlength=k)
    maj = int(counts.argmax())
    out = {
        "majority_class_idx": maj,
        "majority_tied": bool((counts == counts[maj]).sum() > 1),
    }
    rng = np.random.RandomState(seed + 7)
    for name, pred in (
        ("uniform", rng.randint(0, k, len(y_true))),
        ("majority", np.full(len(y_true), maj)),
    ):
        m = _metrics_from_confusion(confusion_matrix(y_true, pred, labels=all_idx), headline_idx)
        out[name] = {
            "m_f1": m["m_f1"],
            "m_iou": m["m_iou"],
            "all_acc": m["all_acc"],
            "all_acc_headline": m["all_acc_headline"],
        }
    return out


def pid_metric(fx: Features, raw: np.ndarray, seed: int, device: str) -> dict:
    """Train SVM + MLP heads on `feat` and on `raw`; score the balanced pool."""
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler

    y = fx.truth["pixel_labels"].astype(np.int64)
    classes, names, headline_classes = list(PID_CLASSES), list(PID_NAMES), list(PID_HEADLINE)
    per_class = fx.pool_spec.per_class

    # Read, not drawn -- see the module docstring. Same recipe, same seeds, one population for
    # the whole suite.
    tr, va = fx.pool("pid_train"), fx.pool("pid_val")
    if len(tr) == 0 or len(va) == 0:
        return {"error": "empty train or val pool"}

    # Contiguous 0..k-1 labels for the heads; `classes` may be non-contiguous.
    lut = {c: i for i, c in enumerate(classes)}
    ytr = np.array([lut[v] for v in y[tr]], dtype=np.int64)
    yva = np.array([lut[v] for v in y[va]], dtype=np.int64)
    k = len(classes)
    headline_idx = [lut[c] for c in headline_classes]
    all_idx = list(range(k))

    res = {
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "classes": [names[classes.index(c)] for c in classes],
        "headline_classes": [names[classes.index(c)] for c in headline_classes],
        "class_counts_train": {
            names[classes.index(c)]: int((y[tr] == c).sum()) for c in classes
        },
        "class_counts_val": {names[classes.index(c)]: int((y[va] == c).sum()) for c in classes},
        # How many distinct events each PID's validation pixels came from. A PID whose 10000
        # pixels come from 113 events is not measured on 10000 independent things, and macro-F1
        # weights it equally with Track regardless. The pixel count alone hides that.
        "event_counts_val": {
            names[classes.index(c)]: int(np.unique(fx.pixel_event[va[y[va] == c]]).size)
            for c in classes
        },
        # Which classes actually entered the macro average. `headline_classes` is already our
        # macro-ignore mechanism; this records the outcome so a collapsed pool is visible in the
        # result file and not only in a warning on stdout.
        "macro_classes_used": [names[classes.index(c)] for c in headline_classes],
    }
    # Rare motifs can starve a pool; macro-F1 then weights a class measured on a handful of
    # pixels equally with Track. Surface it rather than hide it.
    short = {n: c for n, c in res["class_counts_val"].items() if c < per_class}
    if short:
        res["short_pools_val"] = short
        for n, c in sorted(short.items(), key=lambda kv: kv[1]):
            print(f"  [warn] val pool for {n!r} has {c} pixels (< {per_class}) -- its F1 is noisy")

    # What a useless answer scores on the balanced pool. Reported next to the real numbers so
    # neither is read against an assumed 0.
    res["chance"] = _trivial_scores(yva, headline_idx, k, seed)
    if res["chance"]:
        # `majority_class_idx` is a position in `classes`, and `names` is parallel to `classes`,
        # so it indexes `names` directly.
        res["chance"]["majority_class"] = names[res["chance"]["majority_class_idx"]]

    per_class_f1, per_class_iou, confusion = {}, {}, {}
    per_class_precision, per_class_recall = {}, {}
    macro = {}
    for src, X in (("feat", fx.feat), ("raw", raw)):
        # Cast the pooled slice, not the whole array (features are float16).
        Xtr_r = np.asarray(X[tr], dtype=np.float32)
        Xva_r = np.asarray(X[va], dtype=np.float32)
        scaler = StandardScaler().fit(Xtr_r)
        Xtr, Xva = scaler.transform(Xtr_r), scaler.transform(Xva_r)

        heads = {
            "svm": fit_svm(Xtr, ytr, Xva, seed),
            "mlp": fit_mlp(Xtr, ytr, Xva, k, seed, device=device)[0],
        }

        for head, pred in heads.items():
            tag = f"{head}_{src}"
            conf = confusion_matrix(yva, pred, labels=all_idx)
            m = _metrics_from_confusion(conf, headline_idx)
            # Headline key, unchanged: macro-F1 over the six types.
            res[tag] = m["m_f1"]
            res[f"miou_{tag}"] = m["m_iou"]
            macro[tag] = m
            per_class_f1[tag] = _named(m["per_class"]["f1"], classes, names)
            per_class_iou[tag] = _named(m["per_class"]["iou"], classes, names)
            confusion[tag] = conf.tolist()
            # Purity and efficiency per PID, not only F1. Efficiency is the part that does not
            # depend on the pool's class proportions, so it is the one to read as a property of
            # the representation.
            per_class_precision[tag] = _named(m["per_class"]["precision"], classes, names)
            per_class_recall[tag] = _named(m["per_class"]["recall"], classes, names)

    res["per_class_f1"] = per_class_f1
    res["per_class_iou"] = per_class_iou
    res["per_class_precision"] = per_class_precision
    res["per_class_recall"] = per_class_recall
    res["confusion"] = confusion
    res["macro_precision"] = {t: m["m_precision"] for t, m in macro.items()}
    res["macro_recall"] = {t: m["m_recall"] for t, m in macro.items()}
    for head in ("svm", "mlp"):
        res[f"delta_{head}"] = res[f"{head}_feat"] - res[f"{head}_raw"]
        res[f"delta_miou_{head}"] = res[f"miou_{head}_feat"] - res[f"miou_{head}_raw"]
    return res


def run_one(store_root: Path, args) -> dict:
    fx = load_features(store_root, source=args.source, tap=args.tap)
    fx.require("pixel_labels")

    print(f"\n=== {run_label(store_root, args.source)} ===")
    print(f"  events={fx.n_events}  rows={fx.n_pixels}  D={fx.feat.shape[1]}")
    ep = fx.provenance.get("epoch", "?")
    bb = fx.provenance.get("backbone_name", "?")
    print(f"  provenance: epoch={ep} backbone={bb} rows={fx.provenance.get('rows')}")

    raw = raw_charge(fx)
    tr, va = fx.pool("pid_train"), fx.pool("pid_val")
    print(
        f"  pools: {len(tr)} train / {len(va)} val pixels, drawn at extraction "
        f"(per_class={fx.pool_spec.per_class}, seed={fx.pool_spec.seed})"
    )

    entry = run_header(fx, fx.pool_spec.seed, fx.pool_spec.per_class)
    entry["metrics_run"] = ["pid"]

    t0 = time.time()
    entry["pid"] = pid_metric(fx, raw, args.seed, args.device)
    p = entry["pid"]
    if "error" in p:
        print(f"  pid     ERROR: {p['error']}")
        return entry

    print(f"  pid balanced  (n_val={p['n_val']})")
    print(
        f"    macro-F1   svm {p['svm_feat']:.4f} (raw {p['svm_raw']:.4f}, "
        f"delta {p['delta_svm']:+.4f})  mlp {p['mlp_feat']:.4f} "
        f"(raw {p['mlp_raw']:.4f}, delta {p['delta_mlp']:+.4f})"
    )
    print(
        f"    macro-IoU  svm {p['miou_svm_feat']:.4f} "
        f"(raw {p['miou_svm_raw']:.4f}, delta {p['delta_miou_svm']:+.4f})  "
        f"mlp {p['miou_mlp_feat']:.4f} (raw {p['miou_mlp_raw']:.4f}, "
        f"delta {p['delta_miou_mlp']:+.4f})"
    )
    print(
        f"    macro      mlp feat purity {p['macro_precision']['mlp_feat']:.4f} "
        f" efficiency {p['macro_recall']['mlp_feat']:.4f}"
    )
    if p.get("chance"):
        c = p["chance"]
        tie = (
            " -- a tie on a balanced pool, so this number is the tie-break"
            if c.get("majority_tied")
            else ""
        )
        print(
            f"    chance     uniform macro-F1 {c['uniform']['m_f1']:.4f}  "
            f"majority macro-F1 {c['majority']['m_f1']:.4f} "
            f"(always {c.get('majority_class', '?')}{tie})"
        )
    for name, v in p["per_class_f1"]["mlp_feat"].items():
        iou = p["per_class_iou"]["mlp_feat"][name]
        n_val = p["class_counts_val"][name]
        n_ev = p["event_counts_val"][name]
        scored = "" if name in p["macro_classes_used"] else "  (not scored)"
        pr = p["per_class_precision"]["mlp_feat"][name]
        rc = p["per_class_recall"]["mlp_feat"][name]
        print(
            f"      {name:10s} n={n_val:>6d} in {n_ev:>4d} events  "
            f"F1 {v:.4f}  IoU {iou:.4f}  P {pr:.4f}  R {rc:.4f}{scored}"
        )

    print(f"  [{time.time() - t0:.0f}s]")
    return entry


def main(argv: list[str] | None = None) -> int:
    from .runner import probe_argv

    args = probe_argv("pid", argv, default_out="pid_probe.json")
    results = {}
    for store in args.features:
        results[run_label(store, args.source)] = run_one(Path(store), args)
        write_json(results, args.out)
    write_json(results, args.out)
    print(f"\nwrote {args.out}")
    return 0
