"""The "sharded" backend: pre-built shard files, streamed through a shuffle buffer.

This is an `IterableDataset` that yields batches already assembled, 
so it is driven with `DataLoader(batch_size=None)` (see build.py),
which owns that branch so the engine never sees it. Its `batch_size` is the
PER-RANK batch: each rank reads its own shards and yields batches of exactly 
that many samples.

Only a few shards are open at any time, making the memory footprint more manageable.
However, batching requires more attention because buffer is not the full dataset.

"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset

from wcfm.data.collate import collate_meta
from wcfm.data.voxels import Batch, offsets_from_counts, voxels_from

# Per-image event-truth datasets (int / float / special-cased below).
_EVENT_INT_KEYS   = ("labels", "nu_pdg", "nu_ccnc", "nu_intType")
_EVENT_FLOAT_KEYS = ("nu_energy",)
# Per-pixel truth datasets (CSR-aligned with /coords via /offsets).
_PIXEL_KEYS = ("pixel_labels", "pixel_energyfrac", "pixel_trackid", "pixel_truth_q")


class ShardedDataset(IterableDataset):
    """
    IterableDataset over pre-sharded HDF5 files.

    Each shard holds ~shard_size samples in a flat layout (/coords, /features,
    /offsets, plus whatever truth tiers the shards were created with).  At
    training time a shuffle buffer spanning multiple shards approximates
    per-epoch reshuffling while keeping IO to one file read per buffer-refill
    step.

    Event-level truth is returned automatically when the shards carry it.
    Per-pixel tiers are opt-in (return_pixel_truth / return_extra_truth,
    same API as DirectDataset / PackedDataset) because HDF5
    datasets present in a shard are decompressed on every read — training
    should leave them off even on full-truth shard sets.

    Each item yielded is a Batch(voxels, meta):
        voxels: batched Voxels of exactly batch_size samples
        meta:   dict in the same format as collate_meta:
                    label/nu_pdg/nu_ccnc/nu_intType → LongTensor[B]
                    nu_energy                       → FloatTensor[B]
                    vertex_xyz                      → FloatTensor[B, 3]
                    event_key                       → list[str]
                    pixel_labels / pixel_*          → list of B np.ndarray[N_i]
                {} when the shards carry no truth datasets.

    Use with DataLoader(batch_size=None, ...) to disable PyTorch auto-batching.
    Training loops that only need pixels just unpack and drop the meta.

    shuffle=True  (training):    shard order reshuffled each epoch + buffer
                                 shuffle; samples dropped into random batches.
    shuffle=False (diagnostics): deterministic shard order (sorted) and
                                 in-shard sample order — identical event
                                 sequence on every pass, e.g. for comparing
                                 features across checkpoints.

    With num_workers > 0 the shard list is split round-robin across workers so
    every sample is seen exactly once per epoch and workers never duplicate work.

    Under DDP (world_size > 1) the split is over `world_size * num_workers`
    readers instead, so ranks do not read each other's shards. That path also
    equalises the work: DDP's gradient all-reduce is a collective, so a rank
    that runs out of batches early leaves the others blocked in it. See
    `_ddp_shards` for how equal batch counts are guaranteed.
    Each worker maintains its own independent buffer.

    Samples that don't fill a complete final batch are dropped (implicit drop_last).

    Attributes:
        apa  (int|None):  APA number from metadata.json (None if absent).
        view (str|None):  wire-plane view from metadata.json (None if absent).
    """

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 100,
        buffer_size: int = 3000,
        shuffle: bool = True,
        return_pixel_truth: bool = False,
        return_extra_truth: bool = False,
        n_subset: int = -1,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        seed: int = 42,
    ):
        self.root_dir   = Path(root_dir)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        # Needed at construction (not just in __iter__) because __len__ must report
        # the batches THIS rank runs: the schedules are built from it. It must match
        # the DataLoader's num_workers -- __iter__ reads the real count from
        # get_worker_info(), so a mismatch makes __len__ disagree with the steps an
        # epoch actually takes, and the schedules drift from the run.
        self.num_workers = num_workers
        self.seed = seed
        self.epoch = 0
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        if return_extra_truth:
            return_pixel_truth = True
        self.return_pixel_truth = return_pixel_truth
        self.return_extra_truth = return_extra_truth

        self.shards: list[Path] = sorted(self.root_dir.glob("shard_*.h5"))
        if not self.shards:
            raise RuntimeError(f"No shard_*.h5 files found in {root_dir}")

        meta_path = self.root_dir / "metadata.json"
        self.apa: int | None = None
        self.view: str | None = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            n_samples = int(meta["n_samples"])
            self.apa  = int(meta["apa"]) if "apa" in meta else None
            self.view = str(meta["view"]) if "view" in meta else None
        else:
            # fallback: assume full shards (last may be partial, this is approximate)
            n_samples = len(self.shards) * 1000
            print(f"Warning: metadata.json not found in {root_dir}, __len__ is approximate")

        if n_subset > 0:
            # Cap at shard granularity: shards are creation-shuffled, so a
            # prefix of the shard list is an unbiased subsample. Lets smoke
            # configs (n_subset in the run config) keep epochs short.
            with h5py.File(self.shards[0], "r") as f:
                per_shard = int(f["offsets"].shape[0]) - 1
            n_keep = -(-n_subset // per_shard)  # ceil division
            self.shards = self.shards[:n_keep]
            n_samples = min(n_samples, n_keep * per_shard)

        # Per-shard sample count, read once: shard 0 is full by construction, and
        # metadata gives the total, so a short trailing shard is arithmetic rather
        # than 200 file opens at startup.
        with h5py.File(self.shards[0], "r") as f:
            self._per_shard = int(f["offsets"].shape[0]) - 1
        self._n_full_shards = min(n_samples // self._per_shard, len(self.shards))

        if world_size > 1:
            readers = world_size * max(num_workers, 1)
            if self._n_full_shards < readers:
                raise ValueError(
                    f"{root_dir}: {self._n_full_shards} full shards cannot feed "
                    f"{readers} readers ({world_size} ranks x {max(num_workers, 1)} "
                    f"workers) — lower num_workers or use fewer ranks"
                )
            per_reader = -(-self._n_full_shards // readers)   # ceil
            # Every reader gets the same number of equally sized shards, so every
            # reader yields the same number of batches and the ranks stay in step.
            per_reader_batches = (per_reader * self._per_shard) // batch_size
            self._n_batches = per_reader_batches * max(num_workers, 1)
        else:
            self._n_batches = n_samples // batch_size

        # Discover which truth datasets the shards actually carry, then keep
        # only the requested tiers: per-pixel truth in the shards is NOT free
        # to read (unlike the packed .npz, HDF5 datasets present in a shard
        # would be decompressed on every epoch), so it is opt-in like on the
        # other two backends. Event truth (a few scalars per event) is always
        # returned when present.
        with h5py.File(self.shards[0], "r") as f:
            self._has_event_truth = "labels" in f
            available = tuple(k for k in _PIXEL_KEYS if k in f)
        wanted = []
        if self.return_pixel_truth:
            wanted.append("pixel_labels")
        if self.return_extra_truth:
            wanted += ["pixel_energyfrac", "pixel_trackid", "pixel_truth_q"]
        missing = [k for k in wanted if k not in available]
        if missing:
            raise ValueError(
                f"{root_dir}: shards have no {missing}; rebuild with "
                f"`python -m wcfm.data.prep.create_shards` and the matching "
                f"--with_pixel_truth/--with_extra_truth flags."
            )
        self._pixel_keys: tuple[str, ...] = tuple(wanted)

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of batches per epoch (used by DataLoader and schedulers)."""
        return self._n_batches

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed the shared shard shuffle (DDP path).

        Every rank must shuffle the full shard list the same way before splitting it,
        or two readers are handed the same shard while another is never read. The
        training loop calls this once per epoch, like DistributedSampler.set_epoch.
        """
        self.epoch = epoch

    def _ddp_shards(self, worker_id: int, num_workers: int) -> list[Path]:
        """This reader's shards under DDP: an equal, non-overlapping slice.

        Three properties the collective depends on:

        - the shuffle is seeded from (seed, epoch) alone, so every rank permutes the
          full list identically and the split below is a true partition;
        - only full shards take part. A short trailing shard would give its reader
          fewer batches than the others, and the ranks that finish early leave the
          rest blocked in the gradient all-reduce until the job is killed;
        - readers are levelled up to ceil(n_full / readers) shards, not down to
          floor. The full-shard count is rarely divisible by the reader count (here
          it is 199, a prime), and truncating would idle up to readers-1 shards every
          epoch — 39 of 199 at 40 readers. Padding instead repeats a handful of
          shards, so every shard is still read each epoch and only the padding is
          seen twice, which is what DistributedSampler does for the same reason.
        """
        shards = self.shards[: self._n_full_shards]

        g = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(shards), generator=g).tolist()
        shards = [shards[i] for i in order]

        readers = self.world_size * num_workers
        reader_id = self.rank * num_workers + worker_id
        per_reader = -(-len(shards) // readers)               # ceil

        # Pad the permuted list up to readers * per_reader by wrapping around, so the
        # round-robin slice below hands every reader exactly per_reader shards. The
        # repeats land on different shards each epoch because the order is reshuffled.
        padded = shards + shards[: readers * per_reader - len(shards)]
        return padded[reader_id::readers]

    def __iter__(self) -> Iterator[Batch]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if self.world_size > 1:
            shards = self._ddp_shards(worker_id, num_workers)
        else:
            # Single process: split shards round-robin across DataLoader workers.
            shards = self.shards
            if worker_info is not None:
                shards = shards[worker_id::num_workers]

            if self.shuffle:
                # Shuffle shard order for this epoch.
                order  = torch.randperm(len(shards)).tolist()
                shards = [shards[i] for i in order]

        # Buffer: list of (coords, feats, meta_row) tuples, one entry per sample.
        buffer: list[tuple] = []
        shard_idx = 0

        def fill_buffer() -> None:
            nonlocal shard_idx
            while len(buffer) < self.buffer_size and shard_idx < len(shards):
                buffer.extend(self._load_shard(shards[shard_idx]))
                shard_idx += 1

        fill_buffer()

        while len(buffer) >= self.batch_size:
            if self.shuffle:
                # Pick batch_size samples at random from the buffer, keep the rest.
                perm          = torch.randperm(len(buffer)).tolist()
                batch_indices = perm[: self.batch_size]
                keep_indices  = perm[self.batch_size :]
                batch  = [buffer[i] for i in batch_indices]
                buffer[:] = [buffer[i] for i in keep_indices]
            else:
                # Deterministic: consume the buffer front in order.
                batch  = buffer[: self.batch_size]
                buffer[:] = buffer[self.batch_size :]
            fill_buffer()
            yield self._make_batch(batch)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_shard(self, path: Path) -> list[tuple]:
        """Read one shard file and return a list of (coords, feats, meta_row) per sample."""
        with h5py.File(path, "r") as f:
            coords  = torch.from_numpy(f["coords"][()]).to(torch.int32)    # (N_total, 2)
            feats   = torch.from_numpy(f["features"][()]).to(torch.float32) # (N_total, 1)
            offsets = torch.from_numpy(f["offsets"][()]).to(torch.int64)   # (B+1,)

            event = {}
            if self._has_event_truth:
                for k in _EVENT_INT_KEYS + _EVENT_FLOAT_KEYS:
                    event[k] = f[k][()]
                event["vertex_xyz"] = f["vertex_xyz"][()]
                event["event_key"]  = f["event_key"][()]
            pixel = {k: f[k][()] for k in self._pixel_keys}

        samples = []
        for i in range(len(offsets) - 1):
            s, e = int(offsets[i]), int(offsets[i + 1])

            meta_row = None
            if self._has_event_truth or pixel:
                meta_row = {}
                if self._has_event_truth:
                    for k in _EVENT_INT_KEYS:
                        meta_row[k[:-1] if k == "labels" else k] = int(event[k][i])
                    meta_row["nu_energy"]  = float(event["nu_energy"][i])
                    meta_row["vertex_xyz"] = event["vertex_xyz"][i]
                    ev_key = event["event_key"][i]
                    if isinstance(ev_key, (bytes, np.bytes_)):
                        ev_key = ev_key.decode("utf-8")
                    meta_row["event_key"] = ev_key
                for k, arr in pixel.items():
                    meta_row[k] = arr[s:e]

            samples.append((coords[s:e].clone(), feats[s:e].clone(), meta_row))
        return samples

    def _make_batch(self, samples: list[tuple]) -> Batch:
        """Assemble per-sample tuples into one `Batch`.

        Meta goes through `collate_meta`, the same function the map-style backends use, so
        the stacking has one implementation rather than two that can drift apart.
        """
        all_coords, all_feats, meta_rows = zip(*samples, strict=True)
        offsets = offsets_from_counts([c.shape[0] for c in all_coords])
        voxels = voxels_from(
            torch.cat(all_coords, dim=0), torch.cat(all_feats, dim=0), offsets
        )
        if meta_rows[0] is None:
            return Batch(voxels, {})
        return Batch(voxels, collate_meta(meta_rows))
