"""Instance probe: do a pixel's nearest neighbours belong to the same particle?

For each query pixel, the k nearest neighbours within its own event vote on an instance id,
and the vote is scored against the query's true one. No head is trained: this reads the geometry
of the representation directly, so it measures what the features are rather than what can be
fitted to them.

The headline is the macro margin over chance, equal weight per particle size. Chance runs from
0.00 in the small size bins to 0.72 in the largest and 52% of pixels sit in that largest bin, so
a pooled figure is mostly chance and can answer the opposite question to the per-bin evidence:
the two have disagreed in sign on the same extraction. The pooled number is still reported,
labelled as not the headline.

Singleton particles have no mate that could be voted for, so they are wrong by construction; that
is the ceiling, and it is a property of the truth rather than of the features. Bins with a zero
ceiling are excluded from the macro, the same reasoning that keeps Background out of PID's
macro-F1.

The pools are read rather than drawn. `instance_queries` is the global query sample, and
`instance_event_rows` is every truthed pixel of every event a query landed in -- the rows the
vote actually needs. Storing only the queries would leave a pool that reads correctly and cannot
be scored, because a query's neighbours would not be on disk.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..pools import instance_truth_mask
from ..taxonomy import SIZE_BIN_NAMES
from .features import Features, load_features, raw_charge
from .results import run_header, run_label, write_json

__all__ = ["SIZE_BINS", "instance_metric", "main", "majority_vote", "run_one"]

# Bins over the size of a pixel's OWN particle. Singletons get their own bin because they are the
# one group that cannot be scored at all.
SIZE_BINS = ((1, 1), (2, 3), (4, 9), (10, 99), (100, 999), (1000, 10**9))

DEFAULT_KNN_K = 5


def majority_vote(ids: np.ndarray):
    """Most common value per row of `ids` [n, k], with its count.

    A plurality rather than a strict majority, the same rule `probe_event` uses: `argmax` over
    votes.
    Ties break towards the smaller id via the sort, which is arbitrary but deterministic, so two
    checkpoints scored on the same pool break them identically.
    """
    s = np.sort(ids, axis=1)
    best = s[:, 0].copy()
    best_n = np.ones(len(s), dtype=np.int64)
    cur, cnt = s[:, 0].copy(), np.ones(len(s), dtype=np.int64)
    for j in range(1, s.shape[1]):
        same = s[:, j] == cur
        cnt = np.where(same, cnt + 1, 1)
        cur = s[:, j]
        upd = cnt > best_n
        best = np.where(upd, cur, best)
        best_n = np.where(upd, cnt, best_n)
    return best, best_n


def _random_neighbours(n: int, pos: np.ndarray, k: int, rng) -> np.ndarray:
    """k neighbour indices per query, drawn uniformly from the event, self excluded.

    This is the chance level, and it is not small: a particle that owns half its event is the
    plurality among random neighbours most of the time. Without it a big-particle score cannot
    be read at all. Drawn with replacement across the k, which is immaterial except in events
    barely larger than k.
    """
    draw = rng.randint(0, n - 1, size=(len(pos), k))
    return draw + (draw >= pos[:, None])  # shift past self


def _event_predictions(feats_by_src, inst_e, pos, k, device, rng):
    """Plurality instance prediction for one event's queries, per source."""
    import torch

    dev = torch.device(device)
    qk = torch.from_numpy(pos).long().to(dev)
    out = {}
    for src, X in feats_by_src.items():
        fe = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
        fe = fe / fe.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sims = fe[qk] @ fe.T
        sims[torch.arange(len(pos), device=dev), qk] = -2.0  # never your own row
        nn = sims.topk(k, dim=1).indices.cpu().numpy()
        out[src], _ = majority_vote(inst_e[nn])
    out["chance"], _ = majority_vote(inst_e[_random_neighbours(len(inst_e), pos, k, rng)])
    return out


def instance_metric(fx: Features, raw: np.ndarray, seed: int, k: int, device: str) -> dict:
    """Per-pixel majority-vote instance accuracy, overall and per particle size."""
    inst, tm = instance_truth_mask(fx.truth["pixel_labels"], fx.truth["pixel_trackid"])
    if not tm.any():
        return {"error": "no pixels carry instance truth"}

    queries = fx.pool("instance_queries")
    if len(queries) == 0:
        return {"error": "the instance query pool is empty"}

    rng = np.random.RandomState(seed)
    ev_of_q = fx.pixel_event[queries]
    order = np.argsort(ev_of_q, kind="stable")
    queries, ev_of_q = queries[order], ev_of_q[order]
    bounds = np.searchsorted(ev_of_q, np.unique(ev_of_q), side="left")
    bounds = np.append(bounds, len(queries))

    SOURCES = ("feat", "raw", "chance")
    correct = {src: [] for src in SOURCES}
    sizes, n_events, n_skipped_q = [], 0, 0

    for i in range(len(bounds) - 1):
        q = queries[bounds[i] : bounds[i + 1]]
        ev = int(ev_of_q[bounds[i]])
        a, b = int(fx.offsets[ev]), int(fx.offsets[ev + 1])
        rows = a + np.where(tm[a:b])[0]  # truthed pixels of this event
        if len(rows) < k + 1:
            n_skipped_q += len(q)  # too few neighbours to vote
            continue
        inst_e = inst[rows]
        pos = np.searchsorted(rows, q)
        # Instance size within this event, for the pixel's own particle.
        uniq, counts = np.unique(inst_e, return_counts=True)
        size_of = dict(zip(uniq.tolist(), counts.tolist(), strict=True))

        preds = _event_predictions(
            {"feat": fx.feat[rows], "raw": raw[rows]}, inst_e, pos, k, device, rng
        )
        truth_q = inst_e[pos]
        for src in SOURCES:
            correct[src].append(preds[src] == truth_q)
        sizes.append(np.array([size_of[v] for v in truth_q.tolist()]))
        n_events += 1

    if not sizes:
        return {"error": "no event had enough truthed pixels to vote"}

    sizes = np.concatenate(sizes)
    correct = {src: np.concatenate(v) for src, v in correct.items()}

    res = {
        "knn_k": int(k),
        "seed": int(seed),
        "n_queries_scored": int(len(sizes)),
        "n_queries_dropped_small_event": int(n_skipped_q),
        "n_events_used": int(n_events),
        "n_truth_pixels": int(tm.sum()),
        "averaged_over": "pixels",
    }

    # A pixel whose particle is a singleton has no mate that could be voted for, so it is wrong
    # by construction. That is the ceiling, and it is a property of the truth, not the features.
    can = sizes >= 2
    res["ceiling"] = float(can.mean())
    res["singleton_fraction"] = float((sizes == 1).mean())

    # Pooled over every scored pixel. Kept, but NOT the headline -- see the module docstring.
    pooled = {}
    for src in SOURCES:
        pooled[f"{src}_accuracy"] = float(correct[src].mean())
    pooled["delta_accuracy"] = pooled["feat_accuracy"] - pooled["raw_accuracy"]
    pooled["margin_feat"] = pooled["feat_accuracy"] - pooled["chance_accuracy"]
    res["pooled"] = pooled

    per_size = {}
    for (lo, hi), name in zip(SIZE_BINS, SIZE_BIN_NAMES, strict=True):
        m = (sizes >= lo) & (sizes <= hi)
        e = {"n_pixels": int(m.sum()), "pixel_fraction": float(m.mean())}
        if m.sum():
            e["ceiling"] = float(can[m].mean())
            for src in SOURCES:
                e[f"{src}_accuracy"] = float(correct[src][m].mean())
            # The margin over chance is the readable quantity: a raw accuracy of 0.79 is
            # excellent in a bin where chance is 0.01 and unremarkable in one where it is 0.72.
            for src in ("feat", "raw"):
                e[f"{src}_margin"] = e[f"{src}_accuracy"] - e["chance_accuracy"]
            e["delta_margin"] = e["feat_margin"] - e["raw_margin"]
        per_size[name] = e
    res["per_size"] = per_size

    used = [
        n
        for n in SIZE_BIN_NAMES
        if per_size[n].get("ceiling", 0.0) > 0.0 and per_size[n]["n_pixels"] >= 100
    ]
    res["macro_bins_used"] = used
    if used:
        for src in ("feat", "raw"):
            res[f"macro_margin_{src}"] = float(
                np.mean([per_size[n][f"{src}_margin"] for n in used])
            )
        res["delta_macro_margin"] = res["macro_margin_feat"] - res["macro_margin_raw"]
    res["headline"] = "macro_margin_feat"
    return res


def run_one(store_root: Path, args) -> dict:
    fx = load_features(store_root, source=args.source, tap=args.tap)
    fx.require("pixel_labels", "pixel_trackid")
    print(f"\n=== {run_label(store_root, args.source)} ===")

    entry = run_header(fx, args.seed, 0)
    entry["metrics_run"] = ["instance"]
    entry.pop("seed", None)
    entry.pop("pool_per_class", None)

    t0 = time.time()
    m = instance_metric(fx, raw_charge(fx), args.seed, args.knn_k, args.device)
    entry["instance"] = m
    if "error" in m:
        print(f"  [error] {m['error']}")
        return entry

    print(
        f"  queries={m['n_queries_scored']} over {m['n_events_used']} events  "
        f"k={m['knn_k']}  ceiling={m['ceiling']:.4f} "
        f"({100 * m['singleton_fraction']:.2f}% singletons)"
    )
    if "macro_margin_feat" in m:
        print(
            f"  HEADLINE macro margin over chance, equal weight per size "
            f"({'/'.join(m['macro_bins_used'])}):"
        )
        print(
            f"    feat {m['macro_margin_feat']:+.4f}   "
            f"raw {m['macro_margin_raw']:+.4f}   "
            f"feat-raw {m['delta_macro_margin']:+.4f}"
        )
    p = m["pooled"]
    print(
        f"  pooled over pixels (not the headline): feat {p['feat_accuracy']:.4f}  "
        f"raw {p['raw_accuracy']:.4f}  chance {p['chance_accuracy']:.4f}"
    )
    print(
        f"    {'size':>8s} {'%px':>7s} {'chance':>8s} {'feat':>8s} {'raw':>8s}"
        f" {'feat-ch':>9s} {'raw-ch':>9s}"
    )
    for name in SIZE_BIN_NAMES:
        b = m["per_size"][name]
        if "feat_accuracy" in b:
            print(
                f"    {name:>8s} {100 * b['pixel_fraction']:6.2f}% "
                f"{b['chance_accuracy']:8.4f} {b['feat_accuracy']:8.4f} "
                f"{b['raw_accuracy']:8.4f} {b['feat_margin']:+9.4f} "
                f"{b['raw_margin']:+9.4f}"
                + ("" if b["ceiling"] > 0 else "   [ceiling 0, excluded]")
            )
    print(f"  [{time.time() - t0:.0f}s]")
    return entry


def main(argv: list[str] | None = None) -> int:
    from .runner import probe_argv

    def extra(ap):
        ap.add_argument(
            "--knn_k",
            type=int,
            default=DEFAULT_KNN_K,
            help=f"neighbours voting on each pixel (default {DEFAULT_KNN_K})",
        )

    args = probe_argv("instance", argv, default_out="instance.json", extra=extra)
    results = {}
    for store in args.features:
        results[run_label(store, args.source)] = run_one(Path(store), args)
        write_json(results, args.out)
    write_json(results, args.out)
    print(f"\nwrote {args.out}")
    return 0
