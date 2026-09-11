"""Event probe: is the interaction flavor readable from a mean-pooled event vector?

Each event's pixels are mean-pooled to a single vector and scored by cosine k-NN against the
other events -- an untrained readout, so it measures the geometry of the representation rather
than what a head can be fitted to. Purity, majority-vote accuracy and macro-F1 are reported at
several k, for the features and for the raw-charge input, against two degenerate baselines.

The pooling happens at extraction, not here. The sample is `EVENT_POOLED_FROM` in
`wcfm.eval.pools` and the resulting `[n_events, D]` vectors are written to the store, which is
what keeps `rows="pooled"` worth doing: the sample is ~2,000 pixels per event, so keeping it as
rows would put ~20M of the production's ~55M pixels into the row space and pooling would save
almost nothing, for a probe that never looks at a pixel.

The raw-charge baseline is pooled at extraction too, under the same sample, and stored beside
the branches. Both sides therefore pool the same pixels, so the comparison is not confounded by
which pixels each side happened to see.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .features import load_features
from .results import run_header, run_label, write_json

__all__ = ["FLAVOR_NAMES", "KS", "knn_curve", "main", "run_one", "trivial_scores"]

# Event flavor classes (per-event `labels`). Defined here because this is the module that gives
# them meaning -- the same convention `taxonomy` follows for the pixel classes.
FLAVOR_NAMES = ("numuCC", "nueCC", "NC")

KS = (1, 5, 10, 20)

#: The name extraction writes the pooled raw-charge baseline under. Not a branch.
RAW_SOURCE = "raw"


def _flavor(c: int) -> str:
    return FLAVOR_NAMES[c] if c < len(FLAVOR_NAMES) else str(c)


def trivial_scores(y: np.ndarray, n_classes: int, seed: int) -> dict:
    """What the degenerate answers score on this exact event population.

    Two of them, because they differ and neither is 1/3 in general: `uniform` guesses at random,
    and `majority` always predicts the most common flavor, so its accuracy is that flavor's share
    and its macro-F1 a third of a single F1, the other two being zero. `majority_tied` flags an
    arbitrary argmax tie-break.
    """
    from sklearn.metrics import f1_score

    counts = np.bincount(y, minlength=n_classes)
    maj = int(counts.argmax())
    rng = np.random.RandomState(seed + 7)
    out = {
        "majority_fraction": float(counts[maj] / len(y)),
        "majority_class": _flavor(maj),
        "majority_tied": bool((counts == counts[maj]).sum() > 1),
    }
    labels = list(range(n_classes))
    for name, pred in (
        ("uniform", rng.randint(0, n_classes, len(y))),
        ("majority", np.full(len(y), maj)),
    ):
        out[name] = {
            "accuracy": float((pred == y).mean()),
            "macro_f1": float(
                f1_score(y, pred, average="macro", labels=labels, zero_division=0)
            ),
        }
    return out


def knn_curve(F: np.ndarray, y: np.ndarray, n_classes: int, ks=KS, device: str = "cpu") -> dict:
    """Cosine kNN purity, majority-vote accuracy and macro-F1 at each k.

    Self is excluded. Macro-F1 is worth having next to accuracy because the three flavors are
    not equally common: accuracy can be carried by the majority class, F1 averaged over classes
    cannot.
    """
    import torch
    from sklearn.metrics import f1_score

    dev = torch.device(device)
    X = torch.from_numpy(np.ascontiguousarray(F, dtype=np.float32)).to(dev)
    X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    lab = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)).to(dev)

    sim = X @ X.T
    sim.fill_diagonal_(-2.0)
    kmax = min(max(ks), len(X) - 1)
    nn_lab = lab[sim.topk(kmax, dim=1).indices]  # [N, kmax]

    labels = list(range(n_classes))
    out = {}
    for k in ks:
        k_eff = min(k, kmax)
        lab_k = nn_lab[:, :k_eff]
        purity = float((lab_k == lab[:, None]).float().mean())
        votes = torch.nn.functional.one_hot(lab_k, n_classes).sum(1)
        pred = votes.argmax(1)
        acc = float((pred == lab).float().mean())
        macro_f1 = float(
            f1_score(
                lab.cpu().numpy(),
                pred.cpu().numpy(),
                average="macro",
                labels=labels,
                zero_division=0,
            )
        )
        out[str(k)] = {"purity": purity, "accuracy": acc, "macro_f1": macro_f1}
    return out


def run_one(store_root: Path, args) -> dict:
    fx = load_features(store_root, source=args.source, tap=args.tap)
    print(f"\n=== {run_label(store_root, args.source)} ===")

    # Both sides were pooled over the SAME sampled pixels at extraction time, so the comparison
    # is not confounded by which pixels each side happened to see.
    try:
        pooled_feat = np.asarray(fx.event_means())
        pooled_raw = np.asarray(fx.store.event_means(RAW_SOURCE, fx.tap))
    except FileNotFoundError as exc:
        return {"event_knn": {"error": str(exc)}}

    y = fx.labels
    keep = (y >= 0) & np.isfinite(pooled_feat).all(1) & np.isfinite(pooled_raw).all(1)
    n_unknown = int((y < 0).sum())
    n_bad = int(len(y) - keep.sum() - n_unknown)

    entry = run_header(fx, args.seed, 0)
    entry["metrics_run"] = ["event_knn"]
    # `compare` merges every metric for one checkpoint into a single entry, so a top-level
    # `seed` or `pool_per_class` here would silently overwrite PID's, which uses different
    # values by design. Same treatment as probe_knn_pid.
    entry.pop("seed", None)
    entry.pop("pool_per_class", None)

    if keep.sum() < max(KS) + 2:
        entry["event_knn"] = {"error": f"only {int(keep.sum())} usable events"}
        print(f"  [error] {entry['event_knn']['error']}")
        return entry

    y = y[keep]
    pooled_feat, pooled_raw = pooled_feat[keep], pooled_raw[keep]
    n_classes = max(len(FLAVOR_NAMES), int(y.max()) + 1)
    hist = {_flavor(int(c)): int((y == c).sum()) for c in np.unique(y)}

    chance = trivial_scores(y, n_classes, args.seed)
    print(f"  events={len(y)} ({n_unknown} unknown, {n_bad} empty/non-finite dropped)")
    print(
        f"  classes={hist}  majority={chance['majority_class']} "
        f"({chance['majority_fraction']:.4f})"
    )

    t0 = time.time()
    feat = knn_curve(pooled_feat, y, n_classes, device=args.device)
    raw = knn_curve(pooled_raw, y, n_classes, device=args.device)
    metric = {
        "seed": int(args.seed),
        "max_pixels_per_event": int(fx.pool_spec.event_max_per_event),
        "n_events_used": int(len(y)),
        "class_counts": hist,
        "chance": chance,
        "n_unknown_dropped": n_unknown,
        "n_nonfinite_dropped": n_bad,
        "feat": feat,
        "raw": raw,
        "delta_accuracy": {k: feat[k]["accuracy"] - raw[k]["accuracy"] for k in feat},
        "delta_macro_f1": {k: feat[k]["macro_f1"] - raw[k]["macro_f1"] for k in feat},
    }

    for k in (str(x) for x in KS):
        f, r = feat[k], raw[k]
        print(
            f"  k={k:>2}  acc feat {f['accuracy']:.4f} (raw {r['accuracy']:.4f}, "
            f"delta {f['accuracy'] - r['accuracy']:+.4f}, vs majority "
            f"{f['accuracy'] - chance['majority_fraction']:+.4f})  macro-F1 feat "
            f"{f['macro_f1']:.4f} (raw {r['macro_f1']:.4f}, uniform "
            f"{chance['uniform']['macro_f1']:.4f})  purity {f['purity']:.4f}"
        )
    print(f"  [{time.time() - t0:.0f}s]")

    entry["event_knn"] = metric
    return entry


def main(argv: list[str] | None = None) -> int:
    from .runner import probe_argv

    args = probe_argv("event", argv, default_out="event_probe.json")
    results = {}
    for store in args.features:
        results[run_label(store, args.source)] = run_one(Path(store), args)
        write_json(results, args.out)
    write_json(results, args.out)
    print(f"\nwrote {args.out}")
    return 0
