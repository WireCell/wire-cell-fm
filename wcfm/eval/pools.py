"""The populations the probe suite scores, drawn once at extraction time.

The draws happen once, before any feature is written, and `pools.npz` records them, so a probe
reads its population rather than inventing one. That is what lets the suite answer whether two
numbers were measured on the same pixels: a probe drawing its own pool makes that question
unanswerable without re-running it, and two probes drawing separately never score the same rows
at all.

It is also what makes `rows="pooled"` possible. Under it, extraction writes only the union of
everything drawn here, which is a few hundred MB per probe job instead of a few GB. It is sound
only because every draw below is a function of truth and geometry alone -- no features, no
weights -- so the population cannot depend on the checkpoint being scored. Anything that does
depend on features, such as a head's predictions or a neighbour list, is computed by the probe,
on the rows this module chose.

A pool is a set of eval-set pixel indices, and the spec that produced it is recorded. `PoolSpec`
carries every constant a draw consumed, and a probe asserts that the spec it is about to score
against matches the constants it was written for, so changing `THRESHOLDS` in `probe_overlap`
fails loudly against an older store instead of scoring the wrong rows under the right name.

The per-view train/val splits inside a probe's sweep are here, one pool per swept value, because
they are the rows that must exist on disk. What a probe does with them -- fitting, predicting,
breaking down by particle type -- stays in the probe. The dividing line is which rows must be
written, and nothing more.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .balancing import balanced_pool, event_split, pixel_event_index
from .geometry import DEFAULT_VERTEX_T0_TICKS, vertex_distance
from .taxonomy import PID_CLASSES, PIXEL_CLASS_NAMES

__all__ = [
    "DEFAULT_POOL_PER_CLASS",
    "EVENT_POOLED_FROM",
    "PoolSpec",
    "draw_pools",
    "instance_truth_mask",
    "overlap_contamination",
    "truth_mask",
]

#: `probe_pid.py:316`'s own default. The pool size is not a free parameter: it is one of the
#: axes `check_comparability` warns on, because a balanced score over a different pool is a
#: different measurement, and every archived probe number was drawn at this value.
DEFAULT_POOL_PER_CLASS = 10_000

#: The pool whose rows are mean-pooled into per-event vectors at extraction time instead of
#: being written as rows. `probe_event` scores one vector per event, so writing its ~20M
#: sampled pixels into the row space would cost most of what `rows="pooled"` saves for a probe
#: that never looks at a pixel. The pooled vectors are feature-dependent, so they live in the
#: feature store next to the blocks rather than in `pools.npz`.
EVENT_POOLED_FROM = "event_sample"


@dataclass
class PoolSpec:
    """Every constant the draws consume, recorded beside the pools they produced.

    Defaults are the archived ones. A field here is a value some probe's own module constant
    must agree with; :meth:`check` is how the probe says so.
    """

    seed: int = 42
    per_class: int = DEFAULT_POOL_PER_CLASS
    train_frac: float = 0.8

    #: `probe_overlap.THRESHOLDS` and its two caps.
    overlap_thresholds: tuple[float, ...] = (0.2, 0.1, 0.3)
    overlap_train_per_class: int = 20_000
    overlap_val_pixels: int = 2_000_000

    #: `probe_vertex.RADII_PX` and its two caps, plus the projection constant.
    vertex_radii_px: tuple[float, ...] = (20.0, 10.0, 30.0)
    vertex_train_per_class: int = 50_000
    vertex_val_pixels: int = 200_000
    vertex_t0_ticks: float = DEFAULT_VERTEX_T0_TICKS

    #: `probe_instance.DEFAULT_MAX_QUERIES`.
    instance_max_queries: int = 100_000

    #: `probe_event.DEFAULT_MAX_PIXELS_PER_EVENT`.
    event_max_per_event: int = 2_000

    #: `probe_knn_pid`'s per-class pool cap. 0 for the per-image cap means "auto".
    knn_max_per_class: int = 10_000
    knn_max_per_image: int = 0

    #: Set by extraction when the projection could not run, so a reader can tell "the vertex
    #: pool is empty because no vertex projected" from "because nobody asked for one".
    notes: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("overlap_thresholds", "vertex_radii_px"):
            d[k] = [float(v) for v in d[k]]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PoolSpec:
        d = dict(d)
        for k in ("overlap_thresholds", "vertex_radii_px"):
            if k in d:
                d[k] = tuple(float(v) for v in d[k])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def check(self, probe: str, **expected) -> None:
        """Refuse a pool drawn under different constants than the probe expects.

        The failure this prevents: someone edits `THRESHOLDS`, re-runs the probes against an
        existing extraction, and every `overlap_train_1` is scored as though it were the pool
        for the new second threshold. The numbers stay plausible and mean nothing.
        """
        for name, want in expected.items():
            got = getattr(self, name)
            if isinstance(want, tuple) or isinstance(got, tuple):
                same = tuple(np.asarray(got, dtype=float).ravel()) == tuple(
                    np.asarray(want, dtype=float).ravel()
                )
            else:
                same = got == want
            if not same:
                raise ValueError(
                    f"{probe}: the pools in this extraction were drawn with {name}={got!r}, "
                    f"but this probe is written against {want!r}. The stored pool would be "
                    "scored under the wrong name. Re-extract, or score a store drawn with "
                    "matching constants."
                )


# ------------------------------------------------------------------ truth-derived quantities
#
# These three are shared with the probes that score them: the pool draw and the probe must
# agree on what a truth pixel *is*, and two copies of the definition is how they stop agreeing.


def truth_mask(pixel_labels: np.ndarray) -> np.ndarray:
    """Pixels carrying truth: label != 0 (0 is Background / no truth)."""
    return np.asarray(pixel_labels).astype(np.int64) != 0


def overlap_contamination(
    pixel_labels: np.ndarray, pixel_energyfrac: np.ndarray
) -> np.ndarray:
    """Contamination per pixel, 0 = pure.

    `pixel_energyfrac` is the leading contributor's share of the pixel's energy, so `1 - frac`
    is what the other contributors deposited. Pixels without truth have no
    meaningful value and are forced to 0; callers must mask them out rather than read that 0
    as "pure".
    """
    ov = 1.0 - np.asarray(pixel_energyfrac).astype(np.float64)
    ov[~truth_mask(pixel_labels)] = 0.0
    return np.clip(ov, 0.0, 1.0)


def instance_truth_mask(pixel_labels: np.ndarray, pixel_trackid: np.ndarray):
    """`(instance id per pixel, mask of pixels carrying instance truth)`.

    `pixel_trackid` is 0 where there is no truth, and `abs()` would fuse every such pixel
    into one enormous pseudo-instance whose members all neighbour each other -- straight
    inflation. Those pixels are excluded, not relabelled.
    """
    tid = np.asarray(pixel_trackid)
    lab = np.asarray(pixel_labels).astype(np.int64)
    return np.abs(tid.astype(np.int64)), (lab != 0) & (tid != 0)


# ------------------------------------------------------------------------------- the draws


def _natural(candidates: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Up to `cap` indices drawn uniformly, so the natural prevalence is preserved.

    `RandomState(seed + 1).choice`, matching `probe_overlap.py:135-137` and
    `probe_vertex.py:165-167` exactly -- both drew their validation population this way and
    every archived number came from it.
    """
    if len(candidates) <= cap:
        return np.sort(candidates).astype(np.int64)
    picked = np.random.RandomState(seed + 1).choice(candidates, cap, replace=False)
    return np.sort(picked).astype(np.int64)


def _event_sample(offsets: np.ndarray, max_per_event: int, seed: int) -> np.ndarray:
    """Up to `max_per_event` pixel indices per event, drawn without replacement.

    `probe_event.sample_pixels`, verbatim in its draw order: one `RandomState(seed)` advanced
    event by event, so the sequence -- and therefore every archived event-probe number --
    depends on visiting events in order and on skipping the empty ones without consuming
    randomness.

    Events with fewer pixels than the cap keep all of them, so only the large events are
    thinned, which also evens out how precisely each event's mean is estimated: event sizes
    here span 0 to ~52,000 pixels.
    """
    rng = np.random.RandomState(seed)
    counts = np.diff(np.asarray(offsets)).astype(np.int64)
    take = np.minimum(counts, max_per_event)
    idx = np.empty(int(take.sum()), dtype=np.int64)
    pos = 0
    for e in range(len(counts)):
        n, t, lo = int(counts[e]), int(take[e]), int(offsets[e])
        if t == 0:
            continue
        idx[pos : pos + t] = (
            np.arange(lo, lo + n) if t == n else lo + rng.choice(n, size=t, replace=False)
        )
        pos += t
    return idx


def mean_pool(
    features: np.ndarray, sample: np.ndarray, offsets: np.ndarray, n_events: int
) -> np.ndarray:
    """Per-event mean of `features[sample]` -> `[n_events, D]` float32.

    `probe_event.mean_pool`, with the sample passed in rather than redrawn, so the vectors are
    pooled over exactly the pixels `EVENT_POOLED_FROM` chose. Only the sample is upcast, which
    is what every other consumer does too -- pooling the whole array instead would upcast all
    55M pixels, 13 GiB on top of what is already held.

    Empty events yield NaN, and the probe drops them: an event with no pixels has no mean, and
    zero would read as one sitting at the origin.
    """
    sample = np.asarray(sample)
    ev_of = np.searchsorted(np.asarray(offsets), sample, side="right") - 1
    counts = np.bincount(ev_of, minlength=n_events).astype(np.float64)
    dim = features.shape[1]
    sums = np.zeros((n_events, dim), dtype=np.float64)
    if len(sample):
        np.add.at(sums, ev_of, np.asarray(features[sample], dtype=np.float32))
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = sums / counts[:, None]
    pooled[counts == 0] = np.nan
    return pooled.astype(np.float32)


def _knn_pool(
    *,
    pixel_labels: np.ndarray,
    offsets: np.ndarray,
    max_per_class: int,
    max_per_image: int,
    seed: int,
) -> np.ndarray:
    """`probe_knn_pid.collect`'s pool, drawn without the features it used to be drawn beside.

    Visits images in a `default_rng(seed)` permutation, a different generator from every other
    draw here: that is the one `collect` is written against, and swapping it for `RandomState`
    would silently redraw the pool. Takes at most `max_per_image` pixels of one
    class from any single image, so the pool spans many events rather than being dominated by
    the few largest.
    """
    labels = np.asarray(pixel_labels).astype(np.int64)
    offsets = np.asarray(offsets)
    n_images = len(offsets) - 1
    n_classes = len(PIXEL_CLASS_NAMES)
    cap = auto_per_image_cap(max_per_class, n_images) if max_per_image == 0 else max_per_image
    if max_per_image < 0:
        cap = None

    rng = np.random.default_rng(seed)
    picked: list[list[np.ndarray]] = [[] for _ in range(n_classes)]
    counts = np.zeros(n_classes, dtype=np.int64)

    for img in rng.permutation(n_images):
        if counts.min() >= max_per_class:
            break
        lo, hi = int(offsets[img]), int(offsets[img + 1])
        cls = labels[lo:hi]
        for c in range(n_classes):
            if counts[c] >= max_per_class:
                continue
            # Class index c means PID class c + 1: Background is not scored by this probe.
            local = np.nonzero(cls == c + 1)[0]
            if not len(local):
                continue
            room = int(max_per_class - counts[c])
            take = min(len(local), room) if cap is None else min(len(local), cap, room)
            if take < len(local):
                local = rng.choice(local, take, replace=False)
            picked[c].append(lo + local)
            counts[c] += len(local)

    flat = [a for per_class in picked for a in per_class]
    return np.sort(np.concatenate(flat)).astype(np.int64) if flat else _empty()


#: Number of events a class pool should ideally be spread over (`probe_knn_pid`).
SPREAD_TARGET_EVENTS = 200


def auto_per_image_cap(max_pixels_per_class: int, n_images: int) -> int:
    """Per-image cap that spreads a pool over ~`SPREAD_TARGET_EVENTS` events.

    `probe_knn_pid.auto_per_image_cap`, ported rather than re-derived: the pool it produces is
    the archived one, and the cap is the whole reason that probe's numbers are trustworthy. The
    quota alone does not spread a pool over events -- abundant classes reach it almost
    immediately, and on prod-jay at 5000/class the uncapped pools came from 8 events for Track
    and 2 for Shower. A 2-event pool does not represent its class, and the k-NN degenerates
    towards "are pixels of this one shower near each other": uncapped Track recall was 0.777
    against a leakage-free probe's 0.394.

    Adapts to the extraction: when it holds fewer events than the target, the cap grows so the
    quota can still be filled from what is there. Always >= 1.
    """
    target = max(1, min(int(n_images), SPREAD_TARGET_EVENTS))
    return max(1, -(-int(max_pixels_per_class) // target))  # ceil division


def _empty() -> np.ndarray:
    return np.zeros(0, dtype=np.int64)


def draw_pools(
    *,
    truth: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    spec: PoolSpec,
    apa: int = -1,
    view: str = "",
) -> dict[str, np.ndarray]:
    """Every pool the suite scores, in eval-set pixel space.

    A pool whose truth is absent is simply not returned: a reader distinguishes "this probe
    cannot run against this extraction" from "it ran and found nothing", which an empty array
    under the right name would not. `spec.notes` records why.
    """
    offsets = np.asarray(geometry["offsets"])
    n_events = len(offsets) - 1
    pools: dict[str, np.ndarray] = {}

    labels = truth.get("pixel_labels")
    if labels is None:
        spec.notes["all"] = "no pixel_labels: the reader was not asked for per-pixel truth"
        return pools
    labels = np.asarray(labels).astype(np.int64)

    is_train = event_split(n_events, spec.seed, spec.train_frac)[pixel_event_index(offsets)]
    train_idx = np.where(is_train)[0]
    val_idx = np.where(~is_train)[0]

    # -- probe_pid: balanced on both sides, because it reports a macro average over classes
    #    whose natural prevalence spans two orders of magnitude.
    pools["pid_train"] = balanced_pool(
        train_idx, labels, PID_CLASSES, spec.per_class, spec.seed
    )
    pools["pid_val"] = balanced_pool(
        val_idx, labels, PID_CLASSES, spec.per_class, spec.seed + 1
    )

    tm = truth_mask(labels)

    # -- probe_knn_pid: one untrained pool, no split (it is a neighbour search, not a head).
    pools["knn_pool"] = _knn_pool(
        pixel_labels=labels,
        offsets=offsets,
        max_per_class=spec.knn_max_per_class,
        max_per_image=spec.knn_max_per_image,
        seed=spec.seed,
    )

    # -- probe_event: a per-event cap. These rows are NOT part of the pooled row space -- see
    #    `EVENT_POOLED_FROM`. Extraction mean-pools them into one vector per event while it
    #    still holds the full feature block. Keeping them as rows instead would put ~20M of the
    #    55M pixels into the union, and `rows="pooled"` would save almost nothing.
    pools["event_sample"] = _event_sample(offsets, spec.event_max_per_event, spec.seed)

    # -- probe_overlap: natural validation, balanced training per swept threshold.
    if "pixel_energyfrac" in truth:
        ov = overlap_contamination(labels, truth["pixel_energyfrac"])
        tr_cand = np.where(is_train & tm)[0]
        pools["overlap_val"] = _natural(
            np.where(~is_train & tm)[0], spec.overlap_val_pixels, spec.seed
        )
        for i, t in enumerate(spec.overlap_thresholds):
            y_all = (ov > t).astype(np.int64)
            pools[f"overlap_train_{i}"] = balanced_pool(
                tr_cand, y_all, [0, 1], spec.overlap_train_per_class, spec.seed
            )
    else:
        spec.notes["overlap"] = "no pixel_energyfrac: contamination is undefined"

    # -- probe_instance: query pixels drawn globally, not per event -- a per-event cap would
    #    give a 50,000-pixel event the same say as a 500-pixel one. But the probe votes among
    #    a query's neighbours *within its own event*, so the rows it needs are every truthed
    #    pixel of every event a query landed in, which is what `instance_event_rows` carries.
    if "pixel_trackid" in truth:
        _, itm = instance_truth_mask(labels, truth["pixel_trackid"])
        truth_idx = np.where(itm)[0]
        if len(truth_idx) <= spec.instance_max_queries:
            queries = truth_idx.astype(np.int64)
        else:
            queries = np.sort(
                np.random.RandomState(spec.seed).choice(
                    truth_idx, spec.instance_max_queries, replace=False
                )
            ).astype(np.int64)
        pools["instance_queries"] = queries
        pixel_event = pixel_event_index(offsets)
        wanted = np.zeros(n_events, dtype=bool)
        wanted[pixel_event[queries]] = True
        pools["instance_event_rows"] = np.where(itm & wanted[pixel_event])[0].astype(np.int64)
    else:
        spec.notes["instance"] = "no pixel_trackid: instance truth is absent"

    # -- probe_vertex: needs the projection, so it needs to know which wire plane this is.
    if "vertex_xyz" not in truth:
        spec.notes["vertex"] = "no vertex_xyz in the eval set"
    elif not view or apa < 0:
        spec.notes["vertex"] = (
            "no apa/view recorded, so the true vertex cannot be projected into this view"
        )
    else:
        dist, valid, info = vertex_distance(
            positions=geometry["positions"],
            offsets=offsets,
            vertex_xyz=truth["vertex_xyz"],
            apa=apa,
            view=view,
            t0_ticks=spec.vertex_t0_ticks,
        )
        spec.notes["vertex_projection"] = (
            f"{info['n_events_projected']} events projected, "
            f"{info['n_events_vertex_outside_volume']} outside the volume"
        )
        tr_cand = np.where(is_train & valid)[0]
        pools["vertex_val"] = _natural(
            np.where(~is_train & valid)[0], spec.vertex_val_pixels, spec.seed
        )
        for i, r in enumerate(spec.vertex_radii_px):
            y_all = np.zeros(len(dist), dtype=np.int64)
            y_all[valid & (dist <= r)] = 1
            pools[f"vertex_train_{i}"] = balanced_pool(
                tr_cand, y_all, [0, 1], spec.vertex_train_per_class, spec.seed
            )

    return pools
