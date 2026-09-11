"""Pixel-level PID k-NN -- the cheap non-parametric tracking metric.

The complement to `probe_pid`: instead of training a head, it asks whether a pixel's nearest
neighbours in cosine feature space already carry its class. Fast, no fitting -- useful as a
per-epoch curve while a run trains.

Classes are the 7-class pixel taxonomy with Background (label 0) excluded, so this scores motifs
1-6 only. It reports majority-vote accuracy (per class = recall) plus a confusion matrix, and
optionally neighbourhood purity at several k. Every available branch is scored side by side.

This metric has no train/val split -- queries and neighbours come from one pool -- so it is a
relative tracking curve rather than a leakage-free score. Quoting it beside `probe_pid` numbers
reads as though the protocols matched, and `leakage_free: false` is recorded in every entry to
say they do not.

The pool selection depends only on labels and offsets, never on features, which is why it could
move to extraction time with everything else. Its per-image cap is load-bearing rather than a
tuning knob: without it, abundant classes hit their quota almost immediately and on prod-jay at
5000/class the pools came from 8 events for Track and 2 for Shower. A 2-event pool does not
represent its class and the k-NN degenerates towards "are pixels of this one shower near each
other" -- uncapped Track recall was 0.777 against a leakage-free probe's 0.394. See
`wcfm.eval.pools.auto_per_image_cap`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..pools import auto_per_image_cap
from ..taxonomy import PIXEL_CLASS_NAMES
from .features import load_features
from .results import run_header, run_label, write_json

__all__ = ["accuracy", "knn_predict", "knn_purity", "main", "run_one"]

DEFAULT_KNN_K = 20
DEFAULT_BATCH = 4096


def _label_to_class(labels: np.ndarray) -> np.ndarray:
    """Truth labels (int8 0..6) -> class index 0..5; -1 for Background/no-truth."""
    out = np.asarray(labels).astype(np.int32) - 1
    out[np.asarray(labels) == 0] = -1
    return out


def _l2_normalise(X):
    return X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)


def knn_purity(feats, labels, ks, device, batch_size) -> dict:
    """`{k: (overall_purity, per_class[n_classes], per_sample[N])}`."""
    import torch

    n_classes = len(PIXEL_CLASS_NAMES)
    X = _l2_normalise(torch.from_numpy(feats.astype(np.float32)).to(device))
    N = X.shape[0]
    lbls = torch.from_numpy(labels.astype(np.int64)).to(device)
    max_k = min(max(ks), N - 1)

    nn_labels = torch.empty(N, max_k, dtype=torch.int64, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        sim = X[start:end] @ X.T
        sim[torch.arange(B, device=device), torch.arange(start, end, device=device)] = -torch.inf
        _, idx = sim.topk(max_k, dim=1)
        nn_labels[start:end] = lbls[idx]

    out = {}
    for k in ks:
        k_eff = min(k, N - 1)
        same = (nn_labels[:, :k_eff] == lbls[:, None]).float().mean(dim=1)
        per_class = np.full(n_classes, np.nan)
        for c in np.unique(labels):
            per_class[c] = float(same[lbls == c].mean())
        out[k] = (float(same.mean()), per_class, same.cpu().numpy())
    return out


def knn_predict(feats, labels, k, device, batch_size) -> np.ndarray:
    """Majority-vote k-NN predictions `[N]`, self excluded."""
    import torch

    n_classes = len(PIXEL_CLASS_NAMES)
    X = _l2_normalise(torch.from_numpy(feats.astype(np.float32)).to(device))
    N = X.shape[0]
    lbls = torch.from_numpy(labels.astype(np.int64)).to(device)
    k_eff = min(k, N - 1)

    preds = torch.empty(N, dtype=torch.int64, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B = end - start
        sim = X[start:end] @ X.T
        sim[torch.arange(B, device=device), torch.arange(start, end, device=device)] = -torch.inf
        _, idx = sim.topk(k_eff, dim=1)
        nn_lbls = lbls[idx]
        off = torch.arange(B, device=device).unsqueeze(1) * n_classes
        flat = (nn_lbls + off).reshape(-1)
        cnt = torch.bincount(flat, minlength=B * n_classes).reshape(B, n_classes)
        preds[start:end] = cnt.argmax(dim=1)
    return preds.cpu().numpy()


def accuracy(preds: np.ndarray, labels: np.ndarray) -> tuple:
    """`(overall accuracy, per-type recall, per-type F1, macro-F1)`.

    Per-type accuracy is recall, which is blind to false alarms: a type that gets over-predicted
    inflates its own number. F1 adds the precision side and is free here -- the majority vote
    already produced hard labels. Macro-F1 is the figure comparable in form (not in protocol)
    with the trained PID readout.
    """
    from sklearn.metrics import f1_score

    n_classes = len(PIXEL_CLASS_NAMES)
    per_class = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = labels == c
        if m.any():
            per_class[c] = float((preds[m] == c).mean())
    f1 = f1_score(labels, preds, average=None, labels=list(range(n_classes)), zero_division=0)
    present = [c for c in range(n_classes) if (labels == c).any()]
    macro_f1 = float(np.mean([f1[c] for c in present])) if present else float("nan")
    return float((preds == labels).mean()), per_class, np.asarray(f1, float), macro_f1


def run_one(store_root: Path, args) -> dict:
    """Score every branch the store carries. Returns one entry per branch."""
    import torch
    from sklearn.metrics import confusion_matrix

    from ..format import FeatureStore

    sources = list(FeatureStore(store_root).provenance().sources)
    if not sources:
        raise SystemExit(f"{store_root} records no branches to score")

    fxs = {src: load_features(store_root, source=src, tap=args.tap) for src in sources}
    fx0 = fxs[sources[0]]
    fx0.require("pixel_labels")

    print(f"\n=== {run_label(store_root, sources[0])} ===")
    pixel_labels = fx0.truth["pixel_labels"]
    n_truth = int((pixel_labels != 0).sum())
    print(
        f"  events={fx0.n_events}  rows={fx0.n_pixels}  "
        f"with truth={n_truth} ({100 * n_truth / max(1, fx0.n_pixels):.1f}%)  "
        f"D={fx0.feat.shape[1]}  branches={sources}"
    )

    # The pool is read, not collected. `wcfm.eval.pools._knn_pool` ran the same visit order, the
    # same `default_rng(seed)` and the same per-image cap at extraction time.
    pool = fx0.pool("knn_pool")
    if len(pool) < 2:
        return {run_label(store_root, sources[0]): {"knn_pixel": {"error": "empty knn pool"}}}
    pix_cls = _label_to_class(pixel_labels)[pool]
    max_per_class = fx0.pool_spec.knn_max_per_class
    cap = (
        auto_per_image_cap(max_per_class, fx0.n_events)
        if fx0.pool_spec.knn_max_per_image == 0
        else fx0.pool_spec.knn_max_per_image
    )
    counts = np.bincount(pix_cls, minlength=len(PIXEL_CLASS_NAMES))
    ev_per_class = np.array(
        [np.unique(fx0.pixel_event[pool][pix_cls == c]).size for c in range(len(PIXEL_CLASS_NAMES))]
    )

    print(f"  pixels per class per image: {'uncapped (legacy)' if cap < 0 else cap}")
    for c, name in enumerate(PIXEL_CLASS_NAMES):
        short = "  << short of quota" if counts[c] < max_per_class else ""
        print(f"    {name:<9} {int(counts[c]):>8,} from {int(ev_per_class[c]):>5,} events{short}")
    print(f"  total sampled: {len(pix_cls):,}")
    if ev_per_class.min() < 10:
        print(
            "  WARNING: a class pool comes from fewer than 10 events; its score reflects those "
            "events, not the class"
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    k_eff = min(args.knn_k, len(pix_cls) - 1)
    pix_by_src = {src: np.asarray(fx.feat[pool], dtype=np.float32) for src, fx in fxs.items()}
    preds = {
        src: knn_predict(X, pix_cls, k_eff, device, args.batch_size)
        for src, X in pix_by_src.items()
    }
    accs = {src: accuracy(pr, pix_cls) for src, pr in preds.items()}

    print(f"  k-NN majority vote (k={args.knn_k}), recall / F1 per type:")
    for c, name in enumerate(PIXEL_CLASS_NAMES):
        print(
            f"    {name:<9} "
            + "  ".join(f"{src} {accs[src][1][c]:.3f}/{accs[src][2][c]:.3f}" for src in sources)
        )
    print(
        f"    {'Overall':<9} "
        + "  ".join(
            f"{src} acc {accs[src][0]:.3f}  macro-F1 {accs[src][3]:.3f}" for src in sources
        )
    )

    common = {
        "knn_k": int(args.knn_k),
        # Kept inside the metric, not hoisted to the entry, because `compare` merges every
        # metric for one checkpoint into a single entry: a top-level `seed` or `pool_per_class`
        # here would silently overwrite PID's, which uses different values by design.
        "seed": int(fx0.pool_spec.seed),
        "n_pixels_scored": int(len(pix_cls)),
        "max_pixels_per_class": int(max_per_class),
        "max_pixels_per_image": ("uncapped" if cap < 0 else int(cap)),
        "classes": list(PIXEL_CLASS_NAMES),
        "class_counts": {n: int(counts[i]) for i, n in enumerate(PIXEL_CLASS_NAMES)},
        "events_per_class": {n: int(ev_per_class[i]) for i, n in enumerate(PIXEL_CLASS_NAMES)},
        "leakage_free": False,  # one pool, no train/val split -- relative metric
    }
    entries = {}
    for src in sources:
        acc, pr = accs[src], preds[src]
        header = run_header(fxs[src], fx0.pool_spec.seed, max_per_class)
        header.pop("seed", None)
        header.pop("pool_per_class", None)
        entries[run_label(store_root, src)] = {
            **header,
            "metrics_run": ["knn_pixel"],
            "knn_pixel": {
                **common,
                "overall_accuracy": acc[0],
                "macro_f1": acc[3],
                "per_class_accuracy": {
                    n: (None if not np.isfinite(acc[1][i]) else float(acc[1][i]))
                    for i, n in enumerate(PIXEL_CLASS_NAMES)
                },
                "per_class_f1": {n: float(acc[2][i]) for i, n in enumerate(PIXEL_CLASS_NAMES)},
                "confusion": confusion_matrix(
                    pix_cls, pr, labels=list(range(len(PIXEL_CLASS_NAMES)))
                ).tolist(),
            },
        }

    if args.with_purity:
        ks = [int(k) for k in args.ks.split(",")]
        pur = {
            src: knn_purity(X, pix_cls, ks, device, args.batch_size)
            for src, X in pix_by_src.items()
        }
        for k in ks:
            joined = "  ".join(f"{src} {pur[src][k][0]:.3f}" for src in sources)
            print(f"    purity k={k:<3d} {joined}")
        for src in sources:
            entries[run_label(store_root, src)]["knn_pixel"]["purity"] = {
                str(k): pur[src][k][0] for k in ks
            }

    return entries


def main(argv: list[str] | None = None) -> int:
    from .runner import probe_argv

    def extra(ap):
        ap.add_argument("--knn_k", type=int, default=DEFAULT_KNN_K)
        ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
        ap.add_argument("--with_purity", action="store_true")
        ap.add_argument("--ks", default="1,5,10,20")

    args = probe_argv("knn", argv, default_out="knn_pixel.json", extra=extra)
    results: dict = {}
    for store in args.features:
        # Unlike every other probe this returns one entry PER BRANCH, because the branches are
        # scored together against one pool and splitting them into separate invocations would
        # re-draw nothing but would re-read everything.
        results.update(run_one(Path(store), args))
        write_json(results, args.out)
    write_json(results, args.out)
    print(f"\nwrote {args.out}")
    return 0
