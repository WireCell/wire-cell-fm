"""Spectrum probe: how many directions of the representation actually carry anything?

It scores a finished checkpoint, so it covers every written tap rather than whatever the
training loop happened to log.

What it measures, per tap:

- participation ratio and RankMe, the two collapse measures. They disagree in a useful way: the
  participation ratio is dominated by the largest eigenvalue, so it falls the moment one
  direction runs away, while RankMe weighs the tail, so it falls when many small directions
  collapse. A run losing rank slowly moves RankMe first.
- the eigen-spectrum itself, so the shape is auditable rather than summarised twice.
- student-teacher cosine, row by row, where the run has both branches. A teacher that has
  stopped leading its student is the failure mode a loss curve hides best.
- per-class centroid separation, which asks the PID question without a head: how far apart the
  class means are relative to the spread within a class.

The arithmetic is imported. `participation_ratio` and `rankme` come from
`wcfm.metrics.collectors`, which is what the training loop's `Spectrum` collector emits during a
run, so the online curve and this offline number are the same function of the same definition
and a disagreement between them is a real difference in the features. A second implementation
would buy independence and cost that.

Taps do not have to be stride-1 here. Every other probe joins features to per-pixel truth
positionally and therefore refuses a strided tap. The spectrum is a property of the feature
matrix alone, so it is computed for every written tap; only the class-separation section needs
truth, and it is skipped, with a recorded reason, where the join is not available.
"""

from __future__ import annotations

import time
import zlib
from pathlib import Path

import numpy as np

from ..format import FeatureStore
from ..taxonomy import PID_CLASSES, pid_name
from .features import load_features
from .results import run_label, write_json

__all__ = ["main", "run_one", "spectrum_of", "teacher_cosine"]

#: Rows sampled per tap before the covariance. Seeded, so two checkpoints see the same rows and
#: their spectra are comparable -- the same rule `Spectrum`'s in-loop sample follows.
DEFAULT_MAX_ROWS = 65536

#: How many leading eigenvalues to record in full.
DEFAULT_TOP_K = 32


def _tap_seed(tap: str, seed: int) -> int:
    """A stable per-tap seed. See the call site for why this is not `hash()`."""
    return (zlib.crc32(tap.encode()) + int(seed)) % (2**31)


def _subsample(n: int, max_rows: int, seed: int) -> np.ndarray:
    if n <= max_rows:
        return np.arange(n, dtype=np.int64)
    return np.sort(np.random.RandomState(seed).choice(n, max_rows, replace=False))


def spectrum_of(block, rows: np.ndarray, top_k: int = DEFAULT_TOP_K) -> dict:
    """Second-order structure of one feature matrix.

    `participation_ratio` and `rankme` are the training loop's own, imported rather than
    rewritten -- see the module docstring.
    """
    from wcfm.metrics.collectors import participation_ratio, rankme

    X = np.asarray(block[rows], dtype=np.float64)
    n_rows, dim = X.shape
    if n_rows < 2:
        return {"error": f"{n_rows} rows is too few for a covariance"}

    cov = np.cov(X, rowvar=False)
    eig = np.linalg.eigvalsh(cov)  # ascending, real
    positive = eig.clip(0)
    variances = np.diag(cov)
    norms = np.linalg.norm(X, axis=1)

    frob_sq = float((cov**2).sum())
    diag_sq = float((variances**2).sum())
    total = float(positive.sum())
    top = positive[::-1][:top_k]

    return {
        "n_rows": int(n_rows),
        "dim": int(dim),
        "participation_ratio": participation_ratio(eig),
        "rankme": rankme(eig),
        # Recorded as a fraction of the total so two runs at different feature scales can be
        # read against each other; the raw leading eigenvalue is kept beside it for scale.
        "eig_top": [float(v) for v in top],
        "eig_top_frac": [float(v / total) if total > 0 else 0.0 for v in top],
        "eig_max": float(positive.max()) if len(positive) else 0.0,
        # A near-singular covariance has a smallest eigenvalue at the level of fp noise, so this
        # saturates rather than dividing by zero.
        "condition_number": (
            float(positive.max() / positive.min()) if positive.min() > 0 else float("nan")
        ),
        "off_diagonal_frac": (frob_sq - diag_sq) / frob_sq if frob_sq > 0 else 0.0,
        "var_mean": float(variances.mean()),
        "var_min": float(variances.min()),
        "var_max": float(variances.max()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        # The fraction of directions holding 90% and 99% of the variance -- the shape of the
        # spectrum in one number each, and the thing a participation ratio compresses away.
        **_energy_fractions(positive),
    }


def _energy_fractions(positive: np.ndarray) -> dict:
    total = float(positive.sum())
    if total <= 0:
        return {"dims_for_90pct": 0, "dims_for_99pct": 0}
    csum = np.cumsum(positive[::-1]) / total
    return {
        "dims_for_90pct": int(np.searchsorted(csum, 0.90) + 1),
        "dims_for_99pct": int(np.searchsorted(csum, 0.99) + 1),
    }


def teacher_cosine(store: FeatureStore, tap: str, rows: np.ndarray, sources) -> dict | None:
    """Row-by-row cosine between the student and teacher features at one tap.

    `None` when the run has no teacher -- a `mae` preset trains without one, and reporting 0
    or NaN for it would read as a collapsed teacher rather than an absent one.
    """
    if not {"student", "teacher"} <= set(sources):
        return None
    s = np.asarray(store.features("student", tap)[rows], dtype=np.float64)
    t = np.asarray(store.features("teacher", tap)[rows], dtype=np.float64)
    ns = np.linalg.norm(s, axis=1)
    nt = np.linalg.norm(t, axis=1)
    ok = (ns > 0) & (nt > 0)
    if not ok.any():
        return {"error": "every row is zero in one branch"}
    cos = (s[ok] * t[ok]).sum(1) / (ns[ok] * nt[ok])
    return {
        "n_rows": int(ok.sum()),
        "mean": float(cos.mean()),
        "std": float(cos.std()),
        "p05": float(np.percentile(cos, 5)),
        "p50": float(np.percentile(cos, 50)),
        "p95": float(np.percentile(cos, 95)),
        # The norm ratio separates "the two branches point the same way" from "they are the
        # same vector": EMA teachers drift in scale before they drift in direction.
        "norm_ratio_mean": float((nt[ok] / ns[ok]).mean()),
    }


def class_separation(feat, labels: np.ndarray, pool: np.ndarray) -> dict:
    """How far apart the class means are, relative to the spread within a class.

    The PID question without a head, and it answers a different one: a linear probe reports
    what is *extractable*, this reports whether the classes are already apart in the raw
    geometry. A representation can score well on the first while being a single blob that a
    head has learned to slice.
    """
    X = np.asarray(feat[pool], dtype=np.float64)
    y = np.asarray(labels)[pool]
    present = [c for c in PID_CLASSES if (y == c).sum() >= 2]
    if len(present) < 2:
        return {"error": "fewer than two classes have enough pixels in the pool"}

    centroids = np.stack([X[y == c].mean(0) for c in present])
    within = np.array([float(np.linalg.norm(X[y == c] - centroids[i], axis=1).mean())
                       for i, c in enumerate(present)])
    # Pairwise centroid distances, upper triangle only.
    d = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    iu = np.triu_indices(len(present), k=1)
    between = d[iu]

    return {
        "classes": [pid_name(c) for c in present],
        "n_pixels": int(len(pool)),
        "within_class_spread_mean": float(within.mean()),
        "between_class_distance_mean": float(between.mean()),
        "between_class_distance_min": float(between.min()),
        # The headline: > 1 means the classes are further apart than they are wide.
        "separation_ratio": (
            float(between.mean() / within.mean()) if within.mean() > 0 else float("nan")
        ),
        "closest_pair": [
            pid_name(present[iu[0][int(between.argmin())]]),
            pid_name(present[iu[1][int(between.argmin())]]),
        ],
    }


def run_one(store_root: Path, args) -> dict:
    store = FeatureStore(store_root)
    prov = store.provenance()
    sources = list(prov.sources)
    source = args.source if args.source in sources else sources[0]

    print(f"\n=== {run_label(store_root, source)} ===")
    print(f"  taps={prov.taps}  branches={sources}  rows={prov.rows}")

    entry: dict = {
        "features_dir": str(store_root),
        "feature_source": source,
        "rows": prov.rows,
        "eval_set_id": prov.eval_set_id,
        "event_key_hash": prov.event_key_hash,
        "checkpoint_sha256": prov.checkpoint_sha256,
        "metrics_run": ["spectrum"],
        "provenance": {"epoch": (prov.extra or {}).get("epoch", "?")},
    }

    t0 = time.time()
    per_tap: dict = {}
    for tap in prov.taps:
        try:
            block = store.features(source, tap)
        except FileNotFoundError as exc:
            per_tap[tap] = {"error": str(exc)}
            continue
        # Seeded from the tap name and the pool seed, so two checkpoints see the same rows
        # and their spectra are comparable. `crc32`, NOT `hash()`: Python randomises string
        # hashes per process unless PYTHONHASHSEED is set, so `hash(tap)` would draw a
        # different sample in every invocation and the comparison would be silently noisy.
        rows = _subsample(len(block), args.max_rows, seed=_tap_seed(tap, args.seed))
        stats = spectrum_of(block, rows, top_k=args.top_k)
        stride = int(prov.tap_strides.get(tap, 1))
        stats["stride"] = stride
        cos = teacher_cosine(store, tap, rows, sources)
        if cos is not None:
            stats["teacher_cosine"] = cos
        per_tap[tap] = stats

        if "error" not in stats:
            line = (
                f"  {tap:<10s} stride {stride}  D={stats['dim']:<4d} n={stats['n_rows']:<7d} "
                f"PR {stats['participation_ratio']:8.3f}  RankMe {stats['rankme']:8.3f}  "
                f"90% in {stats['dims_for_90pct']:>3d} dims"
            )
            if cos and "mean" in cos:
                line += f"  s-t cos {cos['mean']:+.4f}"
            print(line)

    entry["spectrum"] = {"per_tap": per_tap}

    # The class-separation section needs the per-pixel truth join, so it runs only on the final
    # map, which is the one tap guaranteed to be at full resolution.
    try:
        fx = load_features(store_root, source=source, tap="out", verbose=False)
        if fx.has("pixel_labels") and fx.has_pool("pid_val"):
            sep = class_separation(fx.feat, fx.truth["pixel_labels"], fx.pool("pid_val"))
            entry["spectrum"]["class_separation"] = sep
            if "error" not in sep:
                print(
                    f"  class separation: ratio {sep['separation_ratio']:.4f} "
                    f"(between {sep['between_class_distance_mean']:.4f} / "
                    f"within {sep['within_class_spread_mean']:.4f}); "
                    f"closest pair {sep['closest_pair']}"
                )
        else:
            entry["spectrum"]["class_separation"] = {
                "error": "no pixel_labels or no pid_val pool in this extraction"
            }
    except SystemExit as exc:
        entry["spectrum"]["class_separation"] = {"error": str(exc)}

    # Flat headline keys for the final map, so `compare` reaches them without walking per_tap.
    out_stats = per_tap.get("out", {})
    if "error" not in out_stats:
        entry["spectrum"]["participation_ratio"] = out_stats["participation_ratio"]
        entry["spectrum"]["rankme"] = out_stats["rankme"]
        entry["spectrum"]["dims_for_90pct"] = out_stats["dims_for_90pct"]
        if "teacher_cosine" in out_stats and "mean" in out_stats["teacher_cosine"]:
            entry["spectrum"]["teacher_cosine_mean"] = out_stats["teacher_cosine"]["mean"]
    sep = entry["spectrum"].get("class_separation", {})
    if "separation_ratio" in sep:
        entry["spectrum"]["separation_ratio"] = sep["separation_ratio"]

    print(f"  [{time.time() - t0:.0f}s]")
    return entry


def main(argv: list[str] | None = None) -> int:
    from .runner import probe_argv

    def extra(ap):
        ap.add_argument("--max_rows", type=int, default=DEFAULT_MAX_ROWS)
        ap.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)

    args = probe_argv("spectrum", argv, default_out="spectrum.json", extra=extra)
    results = {}
    for store in args.features:
        results[run_label(store, args.source)] = run_one(Path(store), args)
        write_json(results, args.out)
    write_json(results, args.out)
    print(f"\nwrote {args.out}")
    return 0
