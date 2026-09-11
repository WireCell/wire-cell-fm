"""One extraction, presented to a probe as the object the probe bodies are written against.

This module is the shield that keeps a change of on-disk layout out of the probes. It reads a
`FeatureStore` of mmapped `.npy` blocks, an `EvalSet` of truth written once beside them, and
`pools.npz`, and presents the attributes a probe expects.

Everything a probe sees is in row space. A probe takes for granted that row `i` of `feat` is
pixel `i` of everything else, and under `rows="pooled"` that is false on disk: the store holds
only the union of the pools while truth in the eval set is still full length. So this module
subsets truth to the written rows on the way through -- positions, charges, every per-pixel
column, and a recomputed CSR `offsets` -- and hands the probe back the invariant. Under
`rows="all"` the subset is the identity and nothing is copied.

That is why `row_index` is mandatory in the format and why it is written under both row spaces:
this module indexes through it unconditionally and is correct either way, rather than branching
on `provenance.rows` and being correct only where it was tested.

The recomputed offsets stay valid because `row_index` is sorted, so rows stay grouped by event
and an event is still one contiguous slice. `wcfm.eval.extract` builds it with `np.unique`, and
`load_features` re-checks rather than trusting it, because every per-event probe silently mixes
two events' pixels if it is ever false.

## What a probe must NOT do

Redraw its own pool. The population is drawn once, at extraction, by `wcfm.eval.pools` --
see that module for why. :meth:`Features.pool` is how a probe asks for its own, and it raises
rather than falling back to a fresh draw, because a fallback under `rows="pooled"` would draw
from rows that are not on disk and quietly score the wrong pixels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..format import EvalSet, FeatureStore, Provenance
from ..pools import PoolSpec
from ..rawcharge import raw_charge_from, raw_charge_kind_of

__all__ = [
    "PIXEL_TRUTH_KEYS",
    "Features",
    "load_features",
    "raw_charge",
    "raw_charge_kind",
]

#: Per-pixel truth channels a probe may use. These names are what `fx.truth` is keyed by.
PIXEL_TRUTH_KEYS = ("pixel_labels", "pixel_energyfrac", "pixel_trackid", "pixel_truth_q")

#: Event-level truth, carried as its own attribute rather than through `truth`.
EVENT_TRUTH_KEYS = ("labels", "vertex_xyz", "nu_pdg", "nu_ccnc", "nu_intType", "nu_energy")

#: Loaded extractions, keyed by (store, source, tap). Capped, because an entry was
#: a ~7 GB inflated array; here an entry is a handful of mmaps and small truth columns, so the
#: cap is about bounding the *subset* copies made under `rows="pooled"`, which are real memory.
_CACHE: dict[tuple, Features] = {}
_CACHE_MAX = 4


@dataclass
class Features:
    """One extraction's features and truth, all indexed by the same row space."""

    path: Path
    source: str
    tap: str
    feat: np.ndarray  # [n_rows, D] float16, memory-mapped
    positions: np.ndarray  # [n_rows, 2] int32 (channel, tick)
    charges: np.ndarray  # [n_rows] float32, raw ADC pre-normalization
    offsets: np.ndarray  # [n_events + 1] int64, CSR over ROW space
    pixel_event: np.ndarray  # [n_rows] int64
    labels: np.ndarray  # [n_events] int64, flavour, -1 = unknown
    event_key: np.ndarray  # [n_events] str
    vertex_xyz: np.ndarray  # [n_events, 3] float64, true vertex in cm
    truth: dict[str, np.ndarray] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    #: The underlying store and eval set, for anything that wants them directly.
    store: FeatureStore | None = None
    eval_set: EvalSet | None = None
    prov: Provenance | None = None
    pool_spec: PoolSpec = field(default_factory=PoolSpec)
    charge_params: dict[str, Any] = field(default_factory=dict)
    _pools: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.offsets) - 1

    @property
    def n_pixels(self) -> int:
        """Rows, which under `rows="all"` really are the eval set's pixels.

        Named for the pixels because that is what `run_header` records and what every probe
        prints. `provenance["rows"]` says which of the two a given number is.
        """
        return len(self.feat)

    def has(self, *keys: str) -> bool:
        return all(k in self.truth for k in keys)

    def require(self, *keys: str) -> None:
        missing = [k for k in keys if k not in self.truth]
        if missing:
            raise SystemExit(
                f"{self.path} lacks per-pixel truth {missing}. Extraction writes the tiers the "
                "reader was asked for, so re-extract with `data.return_pixel_truth=true` "
                "against a production that carries them."
            )

    # ------------------------------------------------------------------------------- pools

    def pool(self, name: str) -> np.ndarray:
        """The named population, as positions in this object's row space.

        Raises rather than redrawing. A fallback draw would be right under `rows="all"` and
        silently wrong under `rows="pooled"`, where the rows it chose are not on disk -- and
        the two are indistinguishable from the array shapes.
        """
        if name not in self._pools:
            drawn = sorted(k for k in self._pools if k != "row_index")
            raise KeyError(
                f"{self.path} has no pool {name!r}; it has {drawn}. Pools are drawn once, at "
                "extraction time, by `wcfm.eval.pools` -- a probe does not draw its own. If "
                "the pool is missing because its truth tier was not read, `pool_spec.notes` "
                f"says so: {self.pool_spec.notes or '(no notes recorded)'}"
            )
        return self._pools[name]

    def has_pool(self, name: str) -> bool:
        return name in self._pools

    def event_means(self) -> np.ndarray:
        """`[n_events, D]`, one mean vector per event, pooled at extraction time.

        `probe_event` scores these instead of pixels, which is what keeps its per-event sample
        out of the row space. NaN marks an event with no sampled pixels.
        """
        if self.store is None:
            raise RuntimeError("no feature store behind this object")
        return self.store.event_means(self.source, self.tap)


def load_features(
    store_root: Path | str,
    source: str = "student",
    *,
    tap: str = "out",
    eval_set_root: Path | str | None = None,
    verbose: bool = True,
) -> Features:
    """Read one extraction as a :class:`Features`.

    `store_root` is a checkpoint's feature directory, `<run>/features/epoch<N>/`. The eval set
    defaults to `<store_root>/../eval_set`, which is where `wcfm eval extract` puts
    it, and is passed explicitly when a sweep shares one set across runs.

    Nothing here is decompressed and the feature block is never copied: it is mmapped, and
    every consumer indexes a pool first, so the fp16->fp32 cast lands on the small slice. That
    is what keeps a CPU probe job reading a few hundred MB rather than the whole block.
    """
    store_root = Path(store_root)
    key = (str(store_root.resolve()), source, tap)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    started = time.time()
    store = FeatureStore(store_root)
    prov = store.provenance()
    if source not in prov.sources:
        raise SystemExit(
            f"{store_root} holds no {source!r} features; it has {prov.sources}. A run trained "
            "without a teacher has only a student branch."
        )
    if tap not in prov.taps:
        raise SystemExit(f"{store_root} has no tap {tap!r}; it has {prov.taps}")

    es_root = Path(eval_set_root) if eval_set_root else store_root.parent / "eval_set"
    eval_set = EvalSet.load(es_root)
    eval_set.verify(es_root)

    feat = store.features(source, tap)
    stride = int(prov.tap_strides.get(tap, 1))
    if stride != 1:
        raise SystemExit(
            f"tap {tap!r} has stride {stride}, so its rows sit on a coarser grid than the "
            "per-pixel truth and cannot be joined to it positionally. The probe suite scores "
            f"stride-1 taps; {store_root}/{tap}.coords.npy carries this tap's own coordinates "
            "for anything that wants them."
        )

    pools = store.pools()
    row_index = pools.pop("row_index", None)
    if row_index is None:
        raise SystemExit(
            f"{store_root} has no `row_index` in pools.npz, so its rows cannot be joined to "
            "truth. Every extraction writes one; re-extract."
        )
    row_index = np.asarray(row_index)
    if len(row_index) != len(feat):
        raise SystemExit(
            f"{store_root}: row_index covers {len(row_index)} rows but {source}/{tap} has "
            f"{len(feat)}. The store is inconsistent; re-extract."
        )
    # Load-bearing, and cheap: the CSR rebuild below assumes rows stay grouped by event, which
    # is true only while row_index ascends. If it ever did not, every per-event probe would
    # mix two events' pixels and score them as one.
    if len(row_index) > 1 and not np.all(np.diff(row_index) > 0):
        raise SystemExit(
            f"{store_root}: row_index is not strictly increasing, so rows are no longer "
            "grouped by event and per-event probes would silently mix them."
        )

    full_offsets = np.asarray(eval_set.read(es_root, "offsets"))
    # The eval set's CSR counts pixels; the store may hold a subset of them. `searchsorted`
    # turns pixel boundaries into row boundaries, and an event whose rows were all dropped
    # becomes an empty slice rather than an error -- probes already handle empty events.
    offsets = np.searchsorted(row_index, full_offsets).astype(np.int64)

    truth: dict[str, np.ndarray] = {}
    for k in PIXEL_TRUTH_KEYS:
        if k in eval_set.truth_arrays:
            truth[k] = np.asarray(eval_set.read(es_root, k))[row_index]

    def _event(name: str, default):
        return (
            np.asarray(eval_set.read(es_root, name))
            if name in eval_set.truth_arrays
            else default
        )

    n_events = len(full_offsets) - 1
    fx = Features(
        path=store_root,
        source=source,
        tap=tap,
        feat=feat,
        positions=np.asarray(eval_set.read(es_root, "positions"))[row_index].astype(np.int32),
        charges=np.asarray(eval_set.read(es_root, "charges"))[row_index].astype(np.float32),
        offsets=offsets,
        pixel_event=np.repeat(np.arange(n_events, dtype=np.int64), np.diff(offsets)),
        labels=_event("labels", np.full(n_events, -1, dtype=np.int64)).astype(np.int64),
        event_key=np.load(es_root / "event_keys.npy"),
        vertex_xyz=_event("vertex_xyz", np.zeros((n_events, 3))).astype(np.float64),
        truth=truth,
        provenance=_provenance_dict(prov, eval_set),
        store=store,
        eval_set=eval_set,
        prov=prov,
        pool_spec=PoolSpec.from_dict(prov.pool_spec),
        charge_params=dict(prov.charge_transform_params or {}),
        _pools={k: np.asarray(v) for k, v in pools.items()},
    )

    if verbose:
        print(
            f"  [load_features {time.time() - started:.1f}s] {fx.n_pixels} rows "
            f"({prov.rows}), {fx.n_events} events, D={fx.feat.shape[1]}",
            flush=True,
        )
    while len(_CACHE) >= _CACHE_MAX:
        del _CACHE[next(iter(_CACHE))]
    _CACHE[key] = fx
    return fx


def _provenance_dict(prov: Provenance, eval_set: EvalSet) -> dict[str, Any]:
    """The provenance keys a probe reads.

    They come off the typed `Provenance`, which is why `epoch` has to be dug out of `extra`: it
    is a property of the checkpoint, and `Provenance` names the extraction.
    """
    module = (prov.extra or {}).get("module") or {}
    params = prov.charge_transform_params or {}
    out: dict[str, Any] = {
        "epoch": (prov.extra or {}).get("epoch", "?"),
        "backbone_name": module.get("backbone", "?"),
        "apa": prov.apa,
        "view": prov.view,
        "rows": prov.rows,
        "eval_set_id": prov.eval_set_id,
        "event_key_hash": prov.event_key_hash,
        "sample": eval_set.sample,
        "checkpoint": prov.checkpoint,
        "checkpoint_sha256": prov.checkpoint_sha256,
        "git_sha": prov.git_sha,
        "charge_transform": prov.charge_transform,
        "extract_seconds": prov.extract_seconds,
        "events_per_second": prov.events_per_second,
    }
    if params.get("kind") == "log":
        # These spellings are what `raw_charge_kind` and every recorded result are keyed by.
        out["use_log_transform"] = True
        out["feat_min_val"] = float(params["min_val"])
        out["feat_max_val"] = float(params["max_val"])
    return out


# --------------------------------------------------------------------------- raw charge
#
# The arithmetic lives in `wcfm.eval.rawcharge`, because extraction needs it too: `probe_event`
# scores pooled event vectors, and under `rows="pooled"` the pixels those pool over are not
# written, so the baseline has to be pooled while extraction still holds every row.


def raw_charge_kind(fx: Features) -> str:
    """Which charge transform the raw-charge input used: `trained` or `log10_1p`."""
    return raw_charge_kind_of(fx.charge_params)


def raw_charge(fx: Features) -> np.ndarray:
    """`[channel, tick, log_charge]` per row: the model's own input and nothing else."""
    if raw_charge_kind(fx) == "log10_1p":
        print(
            f"  [warn] {fx.path} carries no charge-transform parameters, so the raw-charge "
            "baseline uses log10(1+q) and its feat-raw deltas are NOT comparable with runs "
            "extracted from a config that recorded them.",
            flush=True,
        )
    return raw_charge_from(fx.positions, fx.charges, fx.charge_params)
