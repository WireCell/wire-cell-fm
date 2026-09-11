"""One pass over the eval set, every branch, written in the evaluation format.

This is the step everything else in `wcfm/eval/` reads from. Four of its properties are what
make the results comparable at all.

The event cap is applied to events, never to batches. `max_images` caps the event count: the
loop keeps its own tally and truncates the batch that crosses it, so on a fixed stream the first
`max_images` events are the same set at any batch size. Capping batches instead makes
`batch_size` silently change which events were scored.

That holds only while the cap binds. Both readers drop a short final batch -- the sharded one by
construction, the map-style one by `drop_last`, which extraction turns off -- so an extraction
that runs out of data before reaching `max_images` ends on a set whose last few events depend on
the batch size after all. `cap_bound` in the provenance records which of the two happened, and
the caller is warned, because the alternative is finding out from a key-hash mismatch after a
second GPU pass. Either way the set is pinned by `EvalSet`'s SHA-256 over its sorted event keys.

Truth is written once. It does not depend on the checkpoint, so a second extraction against an
existing eval set verifies the hash and writes only features.

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

from wcfm.data.voxels import voxels_from

from .format import EvalSet, FeatureStore, Provenance, event_key_hash
from .loading import checkpoint_sha256, inference_sources, inference_step, load_module
from .pools import (
    DEFAULT_POOL_PER_CLASS,
    EVENT_POOLED_FROM,
    PoolSpec,
    draw_pools,
    mean_pool,
)
from .rawcharge import raw_charge_from

#: Per-pixel truth tiers and the dtype each is stored as, from the old extractor's
#: `PIXEL_TRUTH_KEYS`. Present only if the reader was asked for them.
PIXEL_TRUTH: dict[str, Any] = {
    "pixel_labels": np.int8,
    "pixel_energyfrac": np.float32,
    "pixel_trackid": np.int32,
    "pixel_truth_q": np.float32,
}

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

#: Re-exported from `pools`, where the rest of the draw constants live, because
#: `wcfm/cli/eval.py` has imported it from here since the extract stage landed.

#: The name the raw-charge baseline's pooled event vectors are written under. Not a branch --
#: it never appears in `Provenance.sources` -- but it shares the store's naming so `probe_event`
#: reads it through the same call it reads a real branch through.
RAW_SOURCE = "raw"

#: The name the final feature map is written under. Taps are named by the backbone; `out` is
#: what `FeatureBundle.out` is called on disk, and no backbone may declare a tap by that name.
OUT_TAP = "out"


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

    acc = _Accumulator(sources=use, taps=taps)
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if 0 < max_images <= acc.n_events:
                break
            batch = batch.to(device)
            room = max_images - acc.n_events if max_images > 0 else -1
            acc.add(module, batch, room=room)
            if progress and acc.n_batches % 20 == 0:
                print(f"  {acc.n_events} events, {acc.n_pixels} pixels", flush=True)
    elapsed = time.perf_counter() - started

    cap_bound = max_images > 0 and acc.n_events >= max_images
    if not cap_bound:
        print(
            f"  note: the loader ran out after {acc.n_events} events, so max_images="
            f"{max_images} never bound. Both readers drop a short final batch, so this event "
            "set depends on batch_size; an extraction at another batch size may not be "
            "comparable to it. Lower max_images, or keep batch_size fixed across the sweep.",
            flush=True,
        )

    truth, geometry, event_keys = acc.finish()
    eval_set = _eval_set(
        eval_set_root,
        eval_set_id=eval_set_id,
        event_keys=event_keys,
        truth=truth,
        geometry=geometry,
    )

    n_pixels = int(geometry["offsets"][-1])
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

    store = FeatureStore(store_root)
    # The raw-charge baseline, pooled per event under the SAME sample the features use. It is
    # feature-independent, so it is written once per store rather than per branch -- but it
    # must be pooled here: under `rows="pooled"` the pixels it averages are about to stop
    # existing, and a probe recomputing it from the subset would pool a different population.
    if len(event_sample):
        store.write_event_means(
            RAW_SOURCE,
            OUT_TAP,
            mean_pool(
                raw_charge_from(
                    geometry["positions"], geometry["charges"], charge_transform_params
                ),
                event_sample,
                geometry["offsets"],
                acc.n_events,
            ),
        )
    for source in use:
        for tap, block in acc.blocks[source].items():
            # Pool BEFORE subsetting: the event vectors are means over the sampled pixels of
            # the whole eval set, and under `rows="pooled"` most of those rows are about to
            # stop existing. Doing it in the other order would silently average whichever of
            # them happened to survive into the union.
            if len(event_sample) and _stride_of(acc, tap) == 1:
                store.write_event_means(
                    source, tap, mean_pool(block, event_sample, geometry["offsets"], acc.n_events)
                )
            store.write_features(source, tap, block[row_index] if rows == "pooled" else block)
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


# --------------------------------------------------------------------------- the pass


class _Accumulator:
    """Per-batch slices, kept as fp16 from the start.

    The cast happens per batch rather than after the concatenate for the reason the old
    extractor gives at `extract_features.py:230-236`: accumulating fp32 costs 2x and peaks at
    5x while the list, the concatenated copy and the cast copy are all live -- 35 GB of
    features alone at 10k events, against a 32 GB request.
    """

    def __init__(self, sources: Sequence[str], taps: Sequence[str]):
        self.sources = list(sources)
        self.taps = tuple(taps)
        self.blocks: dict[str, dict[str, list]] = {s: {} for s in self.sources}
        self.tap_strides: dict[str, int] = {}
        self._tap_coords: dict[str, list] = {}
        self.event: dict[str, list] = {k: [] for k in EVENT_TRUTH}
        self.pixel: dict[str, list] = {}
        self.vertex: list = []
        self.event_keys: list[str] = []
        self.positions: list = []
        self.charges: list = []
        self.counts: list[int] = []
        self.n_events = 0
        self.n_pixels = 0
        self.n_batches = 0

    def add(self, module, batch, room: int = -1) -> None:
        source_voxels = batch.voxels
        meta = batch.meta
        b_total = len(source_voxels.offsets) - 1
        take = b_total if room < 0 else min(room, b_total)

        # BEFORE the forward: `inference_step` normalises in place, so the raw ADC
        # this column is named for exists only until then -- and `.cpu()` on a tensor already
        # on the CPU is a no-op that returns the SAME storage, so without the clone this reads
        # back as log(q) under the name `charges`. A GPU run would have hidden that, since
        # `.cpu()` does copy off the device: the column would have been right on the cluster
        # and wrong in every CPU test, which is the wrong way round.
        raw_charge = source_voxels.feature_tensor[:, :1].detach().float().cpu().clone().numpy()
        in_coords = source_voxels.coordinate_tensor.detach().cpu().clone().numpy()
        in_offsets = source_voxels.offsets.detach().cpu().clone().numpy()

        # And the model's in-place normalisation must not reach the caller's batch either.
        # `extract` takes any iterable, so a caller may legitimately pass a list of batches and
        # score two checkpoints over it; the second pass would otherwise see log(log(q)).
        xs = voxels_from(
            source_voxels.coordinate_tensor,
            source_voxels.feature_tensor.clone(),
            source_voxels.offsets,
        )
        bundles = inference_step(module, xs, self.sources, self.taps)

        keep_pixels = int(in_offsets[take])
        expected: dict[int, list[int]] = {}  # stride -> per-event counts, shared by the branches
        for source in self.sources:
            bundle = bundles[source]
            named = {OUT_TAP: bundle.out, **{t: bundle.taps[t] for t in self.taps}}
            for tap, vox in named.items():
                stride = self._stride(module, tap)
                if stride > 1 and stride not in expected:
                    expected[stride] = _strided_counts(xs, stride)
                _check_alignment(source, tap, vox, xs, stride, expected.get(stride))
                feats = vox.feature_tensor.detach().float().cpu().numpy()
                if stride == 1:
                    feats = feats[:keep_pixels]
                else:
                    # A strided conv drops trailing empty images from its offsets, so the tap
                    # may describe fewer events than the input batch.
                    last = min(take, len(vox.offsets) - 1)
                    feats = feats[: int(vox.offsets[last])]
                self.blocks[source].setdefault(tap, []).append(feats.astype(np.float16))
                if stride > 1 and source == self.sources[0]:
                    coords = vox.coordinate_tensor.detach().cpu().numpy()
                    last = min(take, len(vox.offsets) - 1)
                    self._tap_coords.setdefault(tap, []).append(coords[: int(vox.offsets[last])])

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

        self.n_events += take
        self.n_pixels += keep_pixels
        self.n_batches += 1

    def _stride(self, module, tap: str) -> int:
        if tap in self.tap_strides:
            return self.tap_strides[tap]
        stride = 1 if tap == OUT_TAP else int(_tap_stride(module, tap))
        self.tap_strides[tap] = stride
        return stride

    @property
    def tap_coords(self) -> dict[str, np.ndarray]:
        return {t: np.concatenate(v, axis=0) for t, v in self._tap_coords.items()}

    def finish(self):
        if not self.n_events:
            raise ValueError(
                "the loader yielded no events, so there is nothing to extract. Check "
                "`data.backend` and its path -- an empty shard directory reads as an empty "
                "dataset rather than as an error."
            )
        for source in self.sources:
            self.blocks[source] = {
                tap: np.concatenate(parts, axis=0)
                for tap, parts in self.blocks[source].items()
            }
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

        geometry = {
            "positions": np.concatenate(self.positions, axis=0).astype(np.int32),
            "charges": np.concatenate(self.charges, axis=0).astype(np.float32),
            "offsets": offsets,
        }
        keys = np.asarray(self.event_keys)
        _check_columns(truth, geometry, keys, n_events=self.n_events, n_pixels=self.n_pixels)
        return truth, geometry, keys


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


def _eval_set(root: Path, *, eval_set_id: str, event_keys, truth, geometry) -> EvalSet:
    """Create the set, or verify an existing one describes these very events.

    Truth does not depend on the checkpoint, so the second checkpoint scored against a set
    writes no truth at all -- it checks that the events it just read are the ones the set was
    built from and moves on. A mismatch is an error here rather than a silent overwrite: the
    results already sitting beside that set were scored on the old events.
    """
    keys = np.asarray(event_keys)
    default_id = eval_set_id or f"n{len(keys)}-{event_key_hash(keys)[:12]}"
    if (root / "eval_set.json").exists():
        existing = EvalSet.load(root)
        if existing.key_hash != event_key_hash(keys):
            raise ValueError(
                f"eval set {existing.id!r} at {root} was built from different events than this "
                f"pass read ({existing.n_events} vs {len(keys)}). Results already written "
                "against it were scored on the old set, so this is refused rather than "
                "overwritten -- extract to a new eval-set root, or delete this one knowingly."
            )
        return existing
    return EvalSet.create(
        root,
        id=default_id,
        event_keys=keys,
        truth=truth,
        geometry=geometry,
    )


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


def _stride_of(acc, tap: str) -> int:
    """The stride extraction recorded for a written tap. `OUT_TAP` is always full resolution.

    Per-event means are pooled against the eval set's own `offsets`, which count *pixels*, so
    a strided tap's rows do not line up with them and it gets no event vectors rather than a
    silently misaligned set.
    """
    return 1 if tap == OUT_TAP else int(acc.tap_strides.get(tap, 1))


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
