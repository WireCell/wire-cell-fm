"""The eval set's truth, then every branch's features, written in the evaluation format.

This is the step everything else in `wcfm/eval/` reads from. Five of its properties are what
make the results comparable at all.

The event cap is applied to events, never to batches, when the set is built. `max_images` caps
the event count: the truth pass keeps its own tally and truncates the batch that crosses it, so
on a fixed stream the first `max_images` events are the same set at any batch size. Capping
batches instead makes `batch_size` silently change which events were scored.

That holds only while the cap binds. Both readers drop a short final batch -- the sharded one by
construction, the map-style one by `drop_last`, which extraction turns off -- so an extraction
that runs out of data before reaching `max_images` ends on a set whose last few events depend on
the batch size after all. `cap_bound` in the provenance records which of the two happened, and
the caller is warned, because the alternative is finding out from a key-hash mismatch after a
second GPU pass. Either way the set is pinned by `EvalSet`'s SHA-256 over its sorted event keys.

Truth is written once, before any feature. It does not depend on the checkpoint: the first
extraction against an eval-set root reads the loader once without the model to write it, and
every extraction then draws its pools from the stored set.

The set, not the loader, decides which events are scored and where their rows go. The feature
pass places each event by its key, skips events the set does not hold, and ends once it has seen
every one. A multi-worker loader interleaves its workers batch by batch, so another batch size
reads the events in another order and a slightly different first N: a positional join would pair
features with another event's truth and raise nothing.

Features are streamed. Pools are a function of truth and geometry alone, so `row_index` is known
before the forward, and each batch writes only its rows of it straight into the on-disk block;
event vectors are pooled batch by batch. Host memory is one batch plus the truth arrays, however
large the row space: under `rows="pooled"` the instance pool alone can hold most pixels.

Raw charge is captured before the transform. The probes score a raw floor against it and the
model normalises in place, so reading it after the forward would read the normalised value under
the name `charges`.

The no-reorder invariant covers every tap, and it is stride-aware. Per-pixel truth is joined to
features positionally, which is legal only if the backbone returns the input's voxels in the
input's order. The same check cannot simply be repeated on a tap: a stride-2 tap cannot have the
input's coordinates, so an exact comparison would fire on a correct backbone, and dropping the
check would give up the guarantee. A stride-1 tap is checked for exact coordinate and offset
equality, and a strided tap for the per-image count its stride implies, which still catches a
layer that prunes or reorders.

Nothing here names `wcfm.model`: `wcfm/eval/` is a framework package, and
`tests/test_import_graph.py` fails the build if it does. The module is rebuilt from its own
config by `loading.py` and asked for its branches through the `inference_step` hook.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wcfm.data import truth as tiers
from wcfm.data.voxels import voxels_from

from .format import EvalSet, FeatureBlock, FeatureStore, Provenance, event_key_hash
from .loading import checkpoint_sha256, inference_sources, inference_step, load_module
from .pools import (
    DEFAULT_POOL_PER_CLASS,
    EVENT_POOLED_FROM,
    PoolSpec,
    draw_pools,
    mean_pool,
)
from .rawcharge import raw_charge_from

#: Per-pixel truth of every tier and the dtype each is stored as. A key is stored only if the
#: reader was asked for its tier; `wcfm.data.truth` lists the tiers.
PIXEL_TRUTH: dict[str, Any] = tiers.pixel_keys(pixel=True, extra=True, rich=True)

#: Event-level truth: the reader's meta key -> the name it is stored under, and its dtype.
#: `label` is renamed to `labels` on disk because that is the name the probe suite reads.
EVENT_TRUTH: dict[str, tuple[str, Any]] = {
    "label": ("labels", np.int64),
    "nu_pdg": ("nu_pdg", np.int64),
    "nu_ccnc": ("nu_ccnc", np.int64),
    "nu_intType": ("nu_intType", np.int64),
    "nu_energy": ("nu_energy", np.float32),
}

#: The names event-level truth is written under, parallel to `EVENT_TRUTH`.
EVENT_TRUTH_NAMES: tuple[str, ...] = tuple(name for name, _ in EVENT_TRUTH.values())

#: The eval-set arrays that are geometry rather than truth.
GEOMETRY: tuple[str, ...] = ("positions", "charges", "offsets")

#: Re-exported from `pools`, where the rest of the draw constants live, because
#: `wcfm/cli/eval.py` has imported it from here since the extract stage landed.

#: The name the raw-charge baseline's pooled event vectors are written under. Not a branch --
#: it never appears in `Provenance.sources` -- but it shares the store's naming so `probe_event`
#: reads it through the same call it reads a real branch through.
RAW_SOURCE = "raw"

#: The name the final feature map is written under. Taps are named by the backbone; `out` is
#: what `FeatureBundle.out` is called on disk, and no backbone may declare a tap by that name.
OUT_TAP = "out"

_NO_EVENTS = (
    "the loader yielded no events, so there is nothing to extract. Check `data.backend` and "
    "its path -- an empty shard directory reads as an empty dataset rather than as an error."
)


@dataclass
class ExtractResult:
    """What one extraction produced, for a caller that wants it without re-reading the disk."""

    eval_set: EvalSet
    provenance: Provenance
    store_root: Path
    eval_set_root: Path
    n_events: int
    n_pixels: int
    n_rows: int


def extract(
    checkpoint: Path | str,
    *,
    store_root: Path | str,
    eval_set_root: Path | str,
    loader: Iterable,
    eval_set_id: str = "",
    sources: Sequence[str] = ("student", "teacher"),
    taps: Sequence[str] = (),
    max_images: int = -1,
    rows: str = "all",
    pool_per_class: int = DEFAULT_POOL_PER_CLASS,
    pool_seed: int = 42,
    device: str = "cpu",
    batch_size: int = -1,
    seed: int = -1,
    charge_transform: str = "",
    charge_transform_params: dict | None = None,
    apa: int = -1,
    view: str = "",
    pool_spec: PoolSpec | None = None,
    gradient_batches: int = 0,
    git_sha: str = "",
    map_location: str | None = None,
    progress: bool = False,
) -> ExtractResult:
    """Score one checkpoint over `loader`, writing an eval set and a feature store.

    `sources` is what the caller wants. A branch the run does not have is dropped rather than
    raising, so `--sources=student,teacher` is a sensible default across a sweep in which some
    runs trained without a teacher. Asking for no available branch at all does raise.

    `loader` is iterated twice when `eval_set_root` holds no set yet. Against an existing set it
    must reach every event the set holds; its order and batch size are free.
    """
    import torch

    checkpoint = Path(checkpoint)
    store_root, eval_set_root = Path(store_root), Path(eval_set_root)
    if rows not in ("all", "pooled"):
        raise ValueError(f"rows must be 'all' or 'pooled', not {rows!r}")

    module, ckpt = load_module(checkpoint, map_location=map_location or device)
    available = inference_sources(module)
    use = [s for s in sources if s in available]
    if not use:
        raise ValueError(
            f"none of the requested sources {list(sources)} exist in {checkpoint}; it has "
            f"{list(available)}"
        )
    module = module.to(device)

    taps = tuple(taps)
    if OUT_TAP in taps:
        raise ValueError(f"{OUT_TAP!r} is the name of the final feature map, not a tap")

    eval_set = _eval_set(
        eval_set_root, eval_set_id=eval_set_id, loader=loader, max_images=max_images
    )
    if 0 < max_images < eval_set.n_events:
        raise ValueError(
            f"eval set {eval_set.id!r} at {eval_set_root} holds {eval_set.n_events} events but "
            f"max_images={max_images}; the set decides what is scored, so extract to a new "
            "eval-set root to score fewer events"
        )
    truth, geometry = _read_eval_set(eval_set_root, eval_set)
    offsets = np.asarray(geometry["offsets"], dtype=np.int64)
    n_pixels = int(offsets[-1])
    # `pool_per_class` and `pool_seed` stay first-class arguments because they are the two a
    # caller actually varies and both are comparability axes; everything else a draw consumes
    # comes from `pool_spec`, whose defaults are the archived constants.
    spec = pool_spec or PoolSpec()
    spec.per_class = int(pool_per_class)
    spec.seed = int(pool_seed)
    row_index, pools, event_sample = _draw_pools(
        truth=truth,
        geometry=geometry,
        rows=rows,
        spec=spec,
        apa=apa,
        view=view,
    )
    # The raw-charge baseline, pooled per event under the SAME sample the features use. It is
    # feature-independent, so it is computed once per store from the set's geometry rather
    # than per branch.
    raw_means = None
    if len(event_sample):
        raw_means = mean_pool(
            raw_charge_from(geometry["positions"], geometry["charges"], charge_transform_params),
            event_sample,
            offsets,
            eval_set.n_events,
        )
    del truth, geometry

    store = FeatureStore(store_root)
    acc = _FeaturePass(
        module,
        store,
        sources=use,
        taps=taps,
        rows=rows,
        event_keys=np.load(eval_set_root / "event_keys.npy"),
        offsets=offsets,
        row_index=row_index,
        event_sample=event_sample,
    )
    started = time.perf_counter()
    try:
        with torch.no_grad():
            _drive(acc, loader, device=device, progress=progress)
        elapsed = time.perf_counter() - started
        acc.finish()
    except BaseException:
        acc.abort()
        raise

    cap_bound = max_images > 0 and eval_set.n_events >= max_images
    if not cap_bound:
        print(
            f"  note: the eval set holds {eval_set.n_events} events, so max_images="
            f"{max_images} never bound. Both readers drop a short final batch, so this event "
            "set depends on batch_size; an extraction at another batch size may not be "
            "comparable to it. Lower max_images, or keep batch_size fixed across the sweep.",
            flush=True,
        )


    if raw_means is not None:
        store.write_event_means(RAW_SOURCE, OUT_TAP, raw_means)
    for (source, tap), means in acc.event_means.items():
        store.write_event_means(source, tap, means)
    for tap, coords in acc.tap_coords.items():
        store.write_coords(tap, coords)
    store.write_pools(row_index=row_index, **pools)

    # The gradient probe, after the no-grad pass: it needs its own forward with autograd on, so
    # it cannot ride along with extraction's. Off by default -- it costs a backward per term per
    # batch, and the feature pass is what most callers want.
    if gradient_batches > 0:
        from .gradients import gradient_report, write_gradients

        report = gradient_report(
            module, loader, device=device, max_batches=gradient_batches, step=int(ckpt.step)
        )
        write_gradients(store_root, report)
        if "error" in report:
            print(f"  gradients: {report['error']}", flush=True)
        else:
            print(
                f"  gradients: {report['terms']} over {report['n_batches']} batches, "
                f"min cosine {report['min_cosine']}",
                flush=True,
            )

    prov = Provenance(
        eval_set_id=eval_set.id,
        event_key_hash=eval_set.key_hash,
        checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256(checkpoint),
        sources=list(use),
        taps=[OUT_TAP, *taps],
        rows=rows,
        tap_strides=acc.tap_strides,
        git_sha=git_sha,
        max_images=int(max_images),
        batch_size=int(batch_size),
        seed=int(seed),
        charge_transform=charge_transform,
        charge_transform_params=dict(charge_transform_params or {}),
        apa=int(apa),
        view=str(view),
        pool_spec=spec.as_dict(),
        pool_per_class=int(pool_per_class),
        pool_seed=int(pool_seed),
        extract_seconds=round(elapsed, 3),
        events_per_second=round(acc.n_events / elapsed, 3) if elapsed > 0 else 0.0,
        extra={
            "epoch": int(ckpt.epoch),
            "step": int(ckpt.step),
            "n_pixels": n_pixels,
            "n_rows": int(len(row_index)),
            "device": str(device),
            "cap_bound": bool(cap_bound),
            "module": _provenance_of(module),
        },
    )
    store.write_provenance(prov)
    return ExtractResult(
        eval_set=eval_set,
        provenance=prov,
        store_root=store_root,
        eval_set_root=eval_set_root,
        n_events=acc.n_events,
        n_pixels=n_pixels,
        n_rows=int(len(row_index)),
    )


# --------------------------------------------------------------------------- the passes


def _drive(acc, loader: Iterable, *, device=None, progress=False) -> None:
    """Feed `loader` to a pass until it runs out or the pass is full."""
    for batch in loader:
        if acc.full:
            break
        if device is not None:
            batch = batch.to(device)
        acc.add(batch)
        if progress and acc.n_batches % 20 == 0:
            print(f"  {acc.n_events} events, {acc.n_pixels} pixels", flush=True)


class _TruthPass:
    """Truth and geometry for a new eval set, read without the model, capped at `max_images`."""

    def __init__(self, max_images: int = -1):
        self.max_images = max_images
        self.event: dict[str, list] = {k: [] for k in EVENT_TRUTH}
        self.pixel: dict[str, list] = {}
        self.tables: list[dict] = []
        self.vertex: list = []
        self.event_keys: list[str] = []
        self.positions: list = []
        self.charges: list = []
        self.counts: list[int] = []
        self.n_events = 0
        self.n_pixels = 0
        self.n_batches = 0

    @property
    def full(self) -> bool:
        return 0 < self.max_images <= self.n_events

    def add(self, batch) -> None:
        source_voxels = batch.voxels
        meta = batch.meta
        # Copied: `.cpu()` on a tensor already on the CPU returns the SAME storage, and a
        # caller may pass a list of batches that the feature pass then normalises in place.
        raw_charge = source_voxels.feature_tensor[:, :1].detach().float().cpu().clone().numpy()
        in_coords = source_voxels.coordinate_tensor.detach().cpu().clone().numpy()
        in_offsets = source_voxels.offsets.detach().cpu().clone().numpy()
        take = len(in_offsets) - 1
        if self.max_images > 0:
            take = min(take, self.max_images - self.n_events)
        keep_pixels = int(in_offsets[take])

        self.positions.append(in_coords[:keep_pixels])
        self.charges.append(raw_charge[:keep_pixels, 0])
        for b in range(take):
            self.counts.append(int(in_offsets[b + 1]) - int(in_offsets[b]))

        for key in EVENT_TRUTH:
            if key in meta:
                self.event[key].extend(_as_list(meta[key])[:take])
        if "vertex_xyz" in meta:
            self.vertex.append(np.asarray(_to_numpy(meta["vertex_xyz"]))[:take])
        self.event_keys.extend([str(k) for k in meta["event_key"]][:take])

        for key in PIXEL_TRUTH:
            if key in meta:
                col = _concat_pixel(meta[key])
                self.pixel.setdefault(key, []).append(col[:keep_pixels])
        # The rich tier's tables, one row per event; stored only if every batch carries them,
        # since `stack_tables` needs every event's rows.
        if all(name in meta for name in tiers.TABLE_COLUMNS):
            for b in range(take):
                self.tables.append({name: _to_numpy(meta[name][b]) for name in tiers.TABLE_COLUMNS})

        self.n_events += take
        self.n_pixels += keep_pixels
        self.n_batches += 1

    def finish(self):
        if not self.n_events:
            raise ValueError(_NO_EVENTS)
        offsets = np.zeros(len(self.counts) + 1, dtype=np.int64)
        np.cumsum(self.counts, out=offsets[1:])

        truth: dict[str, np.ndarray] = {}
        for key, (name, dtype) in EVENT_TRUTH.items():
            if self.event[key]:
                truth[name] = np.asarray(self.event[key], dtype=dtype)
        if self.vertex:
            truth["vertex_xyz"] = np.concatenate(self.vertex, axis=0).astype(np.float32)
        for key, dtype in PIXEL_TRUTH.items():
            if key in self.pixel:
                truth[key] = np.concatenate(self.pixel[key], axis=0).astype(dtype)
        if self.tables:
            truth.update(tiers.stack_tables(self.tables))

        geometry = {
            "positions": np.concatenate(self.positions, axis=0).astype(np.int32),
            "charges": np.concatenate(self.charges, axis=0).astype(np.float32),
            "offsets": offsets,
        }
        keys = np.asarray(self.event_keys)
        _check_columns(truth, geometry, keys, n_events=self.n_events, n_pixels=self.n_pixels)
        return truth, geometry, keys


class _FeaturePass:
    """Each branch's features, written event by event into the store at the set's positions.

    An event is found by its key in the set's `event_keys` and checked against the set's pixel
    count. A stride-1 tap writes the event's rows of `row_index` to their positions in the block
    `FeatureStore.open_features` opened, so the finished block is `full_block[row_index]` in the
    set's order whatever order the loader read. Its event vectors are `mean_pool` over the batch:
    an event never spans two batches, and each event's sample is visited in its stored order, so
    each sum is the one pooling the whole block would compute. Features are cast to fp16 before
    either, which is the precision the block is stored at.

    A strided tap's rows are not pixels, so it is held per event and written at `finish`.
    """

    def __init__(
        self,
        module,
        store: FeatureStore,
        *,
        sources: Sequence[str],
        taps: Sequence[str],
        rows: str,
        event_keys: np.ndarray,
        offsets: np.ndarray,
        row_index: np.ndarray,
        event_sample: np.ndarray,
    ):
        self.module = module
        self.store = store
        self.sources = list(sources)
        self.taps = tuple(taps)
        self.rows = rows
        self.index = {str(k): i for i, k in enumerate(event_keys)}
        self.seen = np.zeros(len(event_keys), dtype=bool)
        self.offsets = offsets
        self.row_index = row_index
        self.event_sample = event_sample
        n_events = len(offsets) - 1
        # `_event_sample` visits events in order, so the sample is grouped by event and
        # `sample_ptr[e]:sample_ptr[e + 1]` is event e's slice of it.
        ev_of = np.searchsorted(offsets, event_sample, side="right") - 1
        if np.any(np.diff(ev_of) < 0):
            raise ValueError("the event sample is not grouped by event")
        self.sample_ptr = np.searchsorted(ev_of, np.arange(n_events + 1))
        self.row_ptr = np.searchsorted(row_index, offsets)
        self.blocks: dict[tuple[str, str], FeatureBlock] = {}
        self.event_means: dict[tuple[str, str], np.ndarray] = {}
        self.strided: dict[tuple[str, str], dict[int, np.ndarray]] = {}
        self.tap_strides: dict[str, int] = {}
        self._tap_coords: dict[str, dict[int, np.ndarray]] = {}
        self.n_events = 0
        self.n_pixels = 0
        self.n_batches = 0

    @property
    def full(self) -> bool:
        return bool(self.seen.all())

    def add(self, batch) -> None:
        source_voxels = batch.voxels
        in_offsets = source_voxels.offsets.detach().cpu().numpy()
        self.n_batches += 1

        local, where = [], []  # batch event b -> set event e, for the events the set holds
        for b, key in enumerate(str(k) for k in batch.meta["event_key"]):
            e = self.index.get(key)
            if e is None:
                continue
            if self.seen[e]:
                raise ValueError(f"event {key!r} appeared twice in the loader")
            here = int(in_offsets[b + 1] - in_offsets[b])
            there = int(self.offsets[e + 1] - self.offsets[e])
            if here != there:
                raise ValueError(
                    f"the eval set was built from different events than this pass read: event "
                    f"{key!r} has {here} pixels here and {there} in the set. Extract to a new "
                    "eval-set root, or check that the loader reads the set's data."
                )
            local.append(b)
            where.append(e)
        if not local:
            return

        # The model's in-place normalisation must not reach the caller's batch. `extract`
        # takes any iterable, so a caller may legitimately pass a list of batches and score two
        # checkpoints over it; the second pass would otherwise see log(log(q)).
        xs = voxels_from(
            source_voxels.coordinate_tensor,
            source_voxels.feature_tensor.clone(),
            source_voxels.offsets,
        )
        bundles = inference_step(self.module, xs, self.sources, self.taps)

        # Per kept event: the block row its rows start at, and their batch positions; and the
        # batch positions of its sample, grouped by event in `local` order.
        placed, sample = [], []
        for b, e in zip(local, where, strict=True):
            shift = int(in_offsets[b]) - int(self.offsets[e])
            lo, hi = self.row_ptr[e], self.row_ptr[e + 1]
            placed.append((int(lo), self.row_index[lo:hi] + shift))
            sample.append(self.event_sample[self.sample_ptr[e] : self.sample_ptr[e + 1]] + shift)
        sample = np.concatenate(sample)
        n_batch = len(in_offsets) - 1

        expected: dict[int, list[int]] = {}  # stride -> per-event counts, shared by the branches
        for source in self.sources:
            bundle = bundles[source]
            named = {OUT_TAP: bundle.out, **{t: bundle.taps[t] for t in self.taps}}
            for tap, vox in named.items():
                stride = self._stride(tap)
                if stride > 1 and stride not in expected:
                    expected[stride] = _strided_counts(xs, stride)
                _check_alignment(source, tap, vox, xs, stride, expected.get(stride))
                feats = vox.feature_tensor.detach().float().cpu().numpy()
                if stride == 1:
                    feats = feats.astype(np.float16)
                    self._write(source, tap, feats, placed)
                    if len(self.event_sample):
                        means = self.event_means.setdefault(
                            (source, tap),
                            np.full((len(self.offsets) - 1, feats.shape[1]), np.nan, np.float32),
                        )
                        pooled = mean_pool(feats, sample, in_offsets, n_batch)
                        means[where] = pooled[local]
                    continue
                # A strided conv drops trailing empty images from its offsets, so the tap may
                # describe fewer events than the input batch.
                tap_offsets = vox.offsets.detach().cpu().numpy()
                coords = vox.coordinate_tensor.detach().cpu().numpy()
                for b, e in zip(local, where, strict=True):
                    lo = int(tap_offsets[min(b, len(tap_offsets) - 1)])
                    hi = int(tap_offsets[min(b + 1, len(tap_offsets) - 1)])
                    part = self.strided.setdefault((source, tap), {})
                    part[e] = feats[lo:hi].astype(np.float16)
                    if source == self.sources[0]:
                        self._tap_coords.setdefault(tap, {})[e] = coords[lo:hi]

        self.seen[where] = True
        self.n_events += len(local)
        self.n_pixels += int(sum(in_offsets[b + 1] - in_offsets[b] for b in local))

    def _write(self, source: str, tap: str, feats: np.ndarray, placed: list) -> None:
        key = (source, tap)
        if key not in self.blocks:
            self.blocks[key] = self.store.open_features(
                source, tap, len(self.row_index), feats.shape[1]
            )
        for start, src in placed:
            self.blocks[key].write(start, feats[src])

    def _stride(self, tap: str) -> int:
        if tap in self.tap_strides:
            return self.tap_strides[tap]
        stride = 1 if tap == OUT_TAP else int(_tap_stride(self.module, tap))
        self.tap_strides[tap] = stride
        return stride

    @property
    def tap_coords(self) -> dict[str, np.ndarray]:
        return {t: np.concatenate([v[e] for e in sorted(v)]) for t, v in self._tap_coords.items()}

    def abort(self) -> None:
        """Delete the half-filled blocks, which are allocated at full size."""
        for block in self.blocks.values():
            block.discard()
        self.blocks.clear()

    def finish(self) -> None:
        """Refuse a pass that missed any of the set's events, then commit every block."""
        if not self.n_events and not self.n_batches:
            raise ValueError(_NO_EVENTS)
        if not self.seen.all():
            missing = np.flatnonzero(~self.seen)
            raise ValueError(
                f"the eval set was built from different events than this pass read: "
                f"{len(missing)} of its {len(self.seen)} events never appeared (set positions "
                f"{missing[:5].tolist()}...). Results already written against the set were "
                "scored on its events -- extract to a new eval-set root, or check that the "
                "loader reads the set's data."
            )
        for block in self.blocks.values():
            block.commit()
        for (source, tap), parts in self.strided.items():
            block = np.concatenate([parts[e] for e in sorted(parts)], axis=0)
            self.store.write_features(
                source, tap, block[self.row_index] if self.rows == "pooled" else block
            )


def _check_columns(truth, geometry, keys, *, n_events: int, n_pixels: int) -> None:
    """Every column is the length of the thing it describes, and no event key repeats.

    A truth key present in some batches and absent in others -- a reader configured differently
    part way through, a shard missing a tier -- yields a column shorter than the set, and every
    join against it is then off by the gap with no error anywhere. Length is the cheapest thing
    that catches it.

    Duplicate event keys matter for a second reason: two copies of one event put its pixels on
    both sides of `event_split`, which is the leak the event-level split exists to prevent, and
    `event_key_hash` sorts a list rather than a set so it would not show up there either.
    """
    expect = {**dict.fromkeys(EVENT_TRUTH_NAMES, n_events), "vertex_xyz": n_events}
    expect.update(dict.fromkeys(PIXEL_TRUTH, n_pixels))
    expect.update(dict.fromkeys(tiers.TABLE_OFFSETS, n_events + 1))
    for name, (table, _, _) in tiers.TABLE_COLUMNS.items():
        if name in truth:
            expect[name] = int(truth[f"{table}_offsets"][-1])
    for name, arr in truth.items():
        want = expect.get(name)
        if want is not None and len(arr) != want:
            raise ValueError(
                f"truth column {name!r} has {len(arr)} rows but the eval set has {want}. A "
                "column that is present in some batches and absent in others comes out short, "
                "and every join against it is then silently off."
            )
    if len(keys) != n_events:
        raise ValueError(f"{len(keys)} event keys for {n_events} events")
    if len(geometry["positions"]) != n_pixels:
        raise ValueError(f"{len(geometry['positions'])} positions for {n_pixels} pixels")
    if len(np.unique(keys)) != len(keys):
        dupes = sorted({k for k in keys.tolist() if keys.tolist().count(k) > 1})[:5]
        raise ValueError(
            f"the eval set repeats {len(keys) - len(np.unique(keys))} event key(s), e.g. "
            f"{dupes}. A duplicated event lands on both sides of the train/val split, which is "
            "exactly the leak splitting by event exists to prevent."
        )


def _strided_counts(xs, stride: int) -> list[int]:
    """Per-event voxel counts the input's coordinates imply at `stride`.

    Computed once per batch and shared by every branch: it depends on the input alone, and a
    `unique(dim=0)` per image per source is real work in the most expensive step of the
    pipeline.
    """
    import torch

    coords = xs.coordinate_tensor
    out = []
    for b in range(len(xs.offsets) - 1):
        s, e = int(xs.offsets[b]), int(xs.offsets[b + 1])
        low = torch.div(coords[s:e], stride, rounding_mode="floor")
        out.append(int(torch.unique(low, dim=0).shape[0]))
    return out


def _check_alignment(source: str, tap: str, vox, xs, stride: int, expected=None) -> None:
    """Refuse a backbone that reordered, dropped or invented voxels.

    Truth is joined to features by row position, so this is the one property that makes the
    join legal. At stride 1 it is exact: the same coordinates, in the same order, split into
    the same events. At stride > 1 the tap has its own coarser grid, so what is checked is the
    per-event count the stride implies -- a pruning or generative layer changes it, which is
    what this is here to catch.
    """
    import torch

    if stride == 1:
        if not torch.equal(vox.offsets, xs.offsets):
            raise RuntimeError(
                f"{source}/{tap}: the backbone changed the per-event voxel counts "
                f"({vox.offsets.tolist()[:5]}... vs input {xs.offsets.tolist()[:5]}...). "
                "Per-pixel truth alignment is no longer positional; extraction would have to "
                "join on coordinates."
            )
        if not torch.equal(vox.coordinate_tensor, xs.coordinate_tensor):
            n = int((vox.coordinate_tensor != xs.coordinate_tensor).any(1).sum())
            raise RuntimeError(
                f"{source}/{tap}: the backbone reordered or moved {n} voxel coordinates. "
                "Per-pixel truth alignment is no longer positional; extraction would have to "
                "join features to truth on (channel, tick)."
            )
        return

    if expected is None:
        expected = _strided_counts(xs, stride)
    actual = [
        int(vox.offsets[b + 1]) - int(vox.offsets[b]) for b in range(len(vox.offsets) - 1)
    ]
    if actual != expected[: len(actual)]:
        raise RuntimeError(
            f"{source}/{tap}: a stride-{stride} tap holds {actual[:5]}... voxels per event, "
            f"but the input's coordinates at that stride are {expected[:5]}.... The tap "
            "pruned or invented voxels, so its rows are not the strided input and its "
            "coordinates cannot be reconstructed."
        )


# ------------------------------------------------------------------------ eval set, pools


def _eval_set(root: Path, *, eval_set_id: str, loader: Iterable, max_images: int) -> EvalSet:
    """The set at `root`, or a new one written from a truth-only pass over `loader`.

    Truth does not depend on the checkpoint, so the second checkpoint scored against a set
    writes no truth at all; the feature pass proves it read every event the set holds.
    """
    if (root / "eval_set.json").exists():
        return EvalSet.load(root)
    acc = _TruthPass(max_images)
    _drive(acc, loader)
    truth, geometry, keys = acc.finish()
    return EvalSet.create(
        root,
        id=eval_set_id or f"n{len(keys)}-{event_key_hash(keys)[:12]}",
        event_keys=keys,
        truth=truth,
        geometry=geometry,
    )


def _read_eval_set(root: Path, eval_set: EvalSet) -> tuple[dict, dict]:
    """The set's truth and geometry columns, memory-mapped."""
    arrays = {name: np.asarray(eval_set.read(root, name)) for name in eval_set.truth_arrays}
    geometry = {name: arrays.pop(name) for name in GEOMETRY if name in arrays}
    return arrays, geometry


def _draw_pools(*, truth, geometry, rows: str, spec: PoolSpec, apa: int, view: str):
    """The suite's pools, and the row space they imply.

    Drawn here rather than in each probe so the population is definitionally the one every
    probe scores. `row_index` is the join back to truth and is written whichever row space is
    in force, so a reader indexes truth through it unconditionally.

    Pools are returned as positions in **row space**, so `feat[pool]` and
    `truth[row_index[pool]]` line up without the caller knowing which space it is in.

    Under `rows="pooled"` the row space is the sorted union of every pool: those are exactly
    the rows some probe will ask for, so writing any others is what the ~7 GB per probe job was
    paying for. That is only sound because `draw_pools` is a function of truth and geometry
    alone -- see its module docstring.
    """
    n_pixels = int(np.asarray(geometry["offsets"])[-1])
    if "pixel_labels" not in truth:
        # No per-pixel taxonomy was read, so there is nothing to balance over. The row space
        # is everything; a pooled request cannot be honoured and says so rather than writing
        # an empty pool that a probe would read as "no pixels of any class".
        if rows == "pooled":
            raise ValueError(
                "rows='pooled' needs `pixel_labels` to balance over, and the reader was not "
                "asked for per-pixel truth. Set data.return_pixel_truth=true, or extract with "
                "rows='all'."
            )
        return np.arange(n_pixels, dtype=np.int64), {}, np.zeros(0, dtype=np.int64)

    pools = draw_pools(truth=truth, geometry=geometry, spec=spec, apa=apa, view=view)

    # `EVENT_POOLED_FROM` leaves `pools` here and never reaches `pools.npz`. Its rows are
    # mean-pooled into one vector per event by the caller, while the full feature block is
    # still in hand, so the probe that reads them never needs a pixel -- and keeping them
    # would put ~20M of 55M rows back into the union, which is most of what pooling saves.
    # It is not stored in another index space either: `pools.npz` is uniformly row space, and
    # the sample is fully determined by `event_max_per_event` and the seed in `pool_spec`.
    event_sample = pools.pop(EVENT_POOLED_FROM, np.zeros(0, dtype=np.int64))

    if rows == "all":
        return np.arange(n_pixels, dtype=np.int64), pools, event_sample

    non_empty = [p for p in pools.values() if len(p)]
    row_index = (
        np.unique(np.concatenate(non_empty)).astype(np.int64)
        if non_empty
        else np.zeros(0, dtype=np.int64)
    )
    # `searchsorted` maps eval-set positions into row space. Every index is in `row_index` by
    # construction -- it is their union -- so no lookup can miss, and one that did would land
    # on a neighbouring row rather than raise, which is why the union is built from the very
    # arrays being mapped.
    mapped = {k: np.searchsorted(row_index, v).astype(np.int64) for k, v in pools.items()}
    return row_index, mapped, event_sample


# ------------------------------------------------------------------------------- helpers


def _tap_stride(module, tap: str) -> int:
    """The tap's stride, asked of whichever object publishes `TAP_STRIDE`.

    Reached by attribute rather than by import: the backbone is a `wcfm.model` class and this
    is a framework module. A tap with no published stride is refused, because assuming 1 is
    exactly the assumption that would make a strided tap's truth join silently wrong.
    """
    for owner in (module, getattr(module, "backbone", None)):
        table = getattr(owner, "TAP_STRIDE", None)
        if table and tap in table:
            return int(table[tap])
    raise ValueError(
        f"tap {tap!r} publishes no stride (no TAP_STRIDE entry), so extraction cannot tell "
        "whether its rows are the input's pixels. Add it to the backbone's TAP_STRIDE."
    )


def _provenance_of(module) -> dict:
    hook = getattr(module, "provenance", None)
    return dict(hook()) if callable(hook) else {}


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _as_list(x) -> list:
    return _to_numpy(x).tolist()


def _concat_pixel(value) -> np.ndarray:
    """A per-pixel truth column arrives either already concatenated or as one array per event."""
    if isinstance(value, (list, tuple)):
        return np.concatenate([_to_numpy(v).reshape(-1) for v in value], axis=0)
    return _to_numpy(value).reshape(-1)
