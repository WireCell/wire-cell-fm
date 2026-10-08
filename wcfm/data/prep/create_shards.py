"""
Create pre-sharded HDF5 files from one or more raw production HDF5 trees.

Each shard file uses a flat layout (not one group per event) so the pixel payload can be
loaded with exactly 3 HDF5 reads regardless of shard size. Event-level truth is always stored
alongside (it is a few scalars per event); per-pixel truth tiers are opt-in flags. The reader
(`ShardedDataset`) detects and returns whatever truth datasets are present automatically.

HDF5 layout per shard file:
    /coords       (N_pix, 2)  int32   -- channel/tick coords (view-rebased)
    /features     (N_pix, 1)  float32 -- pixel ADC values
    /offsets      (N_img+1,)  int64   -- CSR pixel offsets (shared by pixel truth)
    /labels       (N_img,)    int32   -- event class: 0=numuCC 1=nueCC 2=NC -1=unknown
    /nu_pdg       (N_img,)    int32
    /nu_ccnc      (N_img,)    int32
    /nu_intType   (N_img,)    int32
    /nu_energy    (N_img,)    float32
    /vertex_xyz   (N_img, 3)  float32
    /event_key    (N_img,)    bytes   -- UTF-8 "{file}:{group}" traceability string
  with --with_pixel_truth additionally:
    /pixel_labels (N_pix,)    int8    -- per-pixel class label (0=Background/no-truth,
                                         1=Track 2=Shower 3=Michel 4=DeltaRay 5=Blip
                                         6=Other); same CSR as /coords
  with --with_extra_truth additionally:
    /pixel_energyfrac (N_pix,) float32 -- truth-overlap score (frame_energyfrac_1st)
    /pixel_trackid    (N_pix,) int32   -- truth track id, signed (frame_trackid_1st)
    /pixel_truth_q    (N_pix,) float32 -- truth charge (frame_total_numelectrons)
  with --with_rich_truth additionally (`wcfm.data.truth` lists every key):
    /pixel_labels2, /pixel_trackid2, /pixel_energyfrac2 (N_pix,) -- the 2nd contributor
    /pixel_pdg         (N_pix,) int32   -- PDG of the particle pixel_trackid names
    /pixel_vertex_dist (N_pix,) float32 -- distance to the nearest vertex in this view (px)
    /mcpart_offsets (N_img+1,) int64 and /mcpart_<column> -- the event's stored particles
    /simchnl_offsets (N_img+1,) int64 and /simchnl_<column> -- the particles that left charge
    /vertex_offsets (N_img+1,) int64 and /vertex_<column> -- its vertices (`wcfm.data.vertices`)

A metadata.json alongside records the roots, apa, view, n_samples, shard_size, seed, which
truth tiers are present and, with the rich tier, the vertex definition and tick offset.

How the production is read. A cold file on GPFS costs a few hundred milliseconds whatever is
read from it, and the pool tolerates dozens of concurrent readers, so every source file is
read exactly once, whole, by a pool of threads: the `*anode<apa>.h5` files are found by
walking the trees (no file is opened to index it), each is paired with its `_metadata.h5` by
name, and all of its events are parsed in memory with the functions `DirectDataset` reads
with, so a shard set is the production exactly as that reader sees it.

How the events are shuffled. The file list is permuted once from `--seed` and cut into blocks
of `--block_files`; a block is read, its events (plus the remainder carried from the previous
block) are permuted and cut into shards of exactly `--shard_size`, and the remainder carries
on. Only the last shard of the run can be short, which `ShardedDataset` expects. Files are
assigned to blocks at random, so a shard is a uniform draw over every root; what a block
shuffle does not do is separate the events of one file across more than one block's worth of
shards. Memory is one block of events plus `--writers` shards in flight: measured 60 KB per
event on this production, so about 6.5 GB at `--block_files 10000` and 4 writers.

Resume. `files.txt` (the walk) is written before the first block and `state.json` plus
`carry.npz` after every block, all beside the shards; a rerun with the same arguments continues
from them, and refuses different arguments rather than mix two permutations in one directory.
The three files are removed when the run completes.

Usage:
    python -m wcfm.data.prep.create_shards \\
        --datadir /path/to/production [/path/to/another ...] \\
        --apa 0 --view W \\
        --outdir /path/to/shards \\
        [--shard_size 4000] [--seed 42] [--block_files 10000]
        [--threads 32] [--writers 4]
        [--with_pixel_truth] [--with_extra_truth] [--with_rich_truth]
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import io
import json
import shutil
import tempfile
import threading
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path

import h5py
import numpy as np

from wcfm.data import truth, vertices
from wcfm.data.direct import (
    companion_path,
    is_sparse_frame,
    load_geometry,
    read_event,
    view_range,
)
from wcfm.data.wire_geometry import WireGeometry

# One event as read: view-filtered coords (N, 2) int32, features (N, 1) float32, and the meta
# dict from `event_truth` plus any pixel-truth arrays.
Event = tuple[np.ndarray, np.ndarray, dict]

_warned: set[str] = set()
_warn_lock = threading.Lock()


def _warn_once(msg: str) -> None:
    with _warn_lock:
        if msg in _warned:
            return
        _warned.add(msg)
    warnings.warn(msg, stacklevel=2)


def find_files(datadirs: list[str], apa: int) -> list[Path]:
    """Every `*anode<apa>.h5` under the roots, sorted, without opening any of them."""
    files: list[Path] = []
    for d in datadirs:
        found = sorted(p for p in Path(d).rglob(f"*anode{apa}.h5") if p.is_file())
        print(f"Found        : {len(found)} files under {d}")
        files += found
    return files


def _group_key(name: str) -> tuple[int, str]:
    return (int(name), "") if name.isdigit() else (1 << 30, name)


def _open_whole(path: Path, stack: contextlib.ExitStack) -> h5py.File | None:
    """`path` read with one sequential read and opened in memory; None when it is absent."""
    if not path.exists():
        return None
    return stack.enter_context(h5py.File(io.BytesIO(path.read_bytes()), "r"))


def read_file(
    pix_path: Path,
    apa: int,
    view: str,
    keys: dict[str, type],
    rich: bool,
    geom: WireGeometry | None,
) -> list[Event]:
    """Every sparse event of one pixel file, view-filtered, with its truth. The file and its
    companions are each read whole with one sequential read and parsed in memory."""
    ch_start, ch_end = view_range(view)
    events: list[Event] = []
    with contextlib.ExitStack() as stack:
        f = _open_whole(pix_path, stack)
        metadata = _open_whole(companion_path(pix_path, apa, "metadata"), stack)
        trackmap = (
            _open_whole(companion_path(pix_path, apa, "trackid_pid_map"), stack) if rich else None
        )
        for group in sorted(f.keys(), key=_group_key):
            grp = f[group]
            if not isinstance(grp, h5py.Group) or not is_sparse_frame(grp):
                continue
            events.append(
                read_event(
                    grp,
                    group,
                    f"{pix_path.name}:{group}",
                    metadata=metadata,
                    trackmap=trackmap,
                    ch_start=ch_start,
                    ch_end=ch_end,
                    keys=keys,
                    rich=rich,
                    geom=geom,
                    apa=apa,
                    view=view,
                    warn=_warn_once,
                )
            )
    return events


def assemble(events: list[Event], keys: dict[str, type], rich: bool) -> dict[str, np.ndarray]:
    """The flat arrays of one shard, in the layout the module docstring lists."""
    coords = [c for c, _, _ in events]
    offsets = np.zeros(len(events) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(c) for c in coords])
    out = {
        "coords": np.concatenate(coords, axis=0).astype(np.int32),
        "features": np.concatenate([f for _, f, _ in events], axis=0).astype(np.float32),
        "offsets": offsets,
        "labels": np.array([m["label"] for _, _, m in events], dtype=np.int32),
        "nu_pdg": np.array([m["nu_pdg"] for _, _, m in events], dtype=np.int32),
        "nu_ccnc": np.array([m["nu_ccnc"] for _, _, m in events], dtype=np.int32),
        "nu_intType": np.array([m["nu_intType"] for _, _, m in events], dtype=np.int32),
        "nu_energy": np.array([m["nu_energy"] for _, _, m in events], dtype=np.float32),
        "vertex_xyz": np.stack([m["vertex_xyz"] for _, _, m in events]).astype(np.float32),
        "event_key": np.array([m["event_key"].encode("utf-8") for _, _, m in events], dtype=object),
    }
    for k, dtype in keys.items():
        parts = [m[k] for _, _, m in events]
        assert all(len(p) == len(c) for p, c in zip(parts, coords, strict=True)), (
            f"{k} length differs from coords in shard"
        )
        out[k] = np.concatenate(parts, axis=0).astype(dtype)
    if rich:
        out.update(truth.stack_tables([m for _, _, m in events]))
    return out


def write_shard(path: Path, arrays: dict[str, np.ndarray]) -> tuple[str, int, int]:
    """Write one shard; gzip-6 on the per-pixel datasets. Runs in a writer process because
    HDF5 compresses under one process-wide lock, so writers parallelise only as processes.

    The file is built on local scratch (`tempfile.gettempdir()`, which Condor points at the
    job's scratch directory) and moved to `path` in one sequential copy: GPFS charges latency
    per write call, and a shard is a few hundred compressed chunks. Cluster 348 spent 18 min
    per block writing straight to GPFS against about 2 min of compression.
    """
    per_pixel = {"coords", "features", *truth.pixel_keys(True, True, True), *truth.TABLE_COLUMNS}
    with tempfile.TemporaryDirectory(prefix="shard_") as tmpdir:
        local = Path(tmpdir) / path.name
        with h5py.File(local, "w") as hf:
            for k, v in arrays.items():
                if k == "event_key":
                    hf.create_dataset(k, data=v, dtype=h5py.special_dtype(vlen=bytes))
                elif k in per_pixel:
                    # Large chunks: the readers load whole datasets, and every chunk is one
                    # write call on the copy's destination filesystem.
                    chunks = (min(len(v), 1 << 19),) + v.shape[1:]
                    hf.create_dataset(
                        k, data=v, chunks=chunks, compression="gzip", compression_opts=6
                    )
                else:
                    hf.create_dataset(k, data=v)
        staged = path.with_suffix(".h5.tmp")
        shutil.copyfile(local, staged)
    staged.replace(path)
    return path.name, int(arrays["coords"].shape[0]), int(arrays["offsets"].shape[0] - 1)


def _write_text(path: Path, text: str) -> None:
    """Write via a temp name and rename, so a job removed mid-write leaves the old file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _release_heap() -> None:
    """Hand the freed block back to the OS. A block is 100k small arrays, and glibc keeps
    their arenas after `del`, so without this the next block sits on top of the last one's
    footprint (cluster 347 was held at 16 GB for exactly that)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _prefork(pool: ProcessPoolExecutor, n: int) -> None:
    """Start every writer process now, while this process is small. The pool forks a worker at
    the first submit that finds none idle, and a worker forked while a block is loaded carries a
    copy-on-write snapshot of it for the rest of the run."""
    wait([pool.submit(time.sleep, 0.5) for _ in range(n)])


def _save_carry(path: Path, events: list[Event], keys: dict[str, type], rich: bool) -> None:
    if not events:
        path.unlink(missing_ok=True)
        return
    arrays = assemble(events, keys, rich)
    arrays["event_key"] = arrays["event_key"].astype("S")
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fp:
        np.savez(fp, **arrays)
    tmp.replace(path)


def _load_carry(path: Path, keys: dict[str, type], rich: bool) -> list[Event]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    events: list[Event] = []
    for i in range(len(a["offsets"]) - 1):
        s, e = int(a["offsets"][i]), int(a["offsets"][i + 1])
        meta = {
            "label": int(a["labels"][i]),
            "nu_pdg": int(a["nu_pdg"][i]),
            "nu_ccnc": int(a["nu_ccnc"][i]),
            "nu_intType": int(a["nu_intType"][i]),
            "nu_energy": float(a["nu_energy"][i]),
            "vertex_xyz": a["vertex_xyz"][i],
            "event_key": a["event_key"][i].decode("utf-8"),
        }
        meta.update(truth.event_truth_slice(a, i, tuple(keys), rich))
        events.append((a["coords"][s:e], a["features"][s:e], meta))
    return events


def create_shards(
    datadirs: list[str],
    apa: int,
    view: str,
    outdir: str,
    shard_size: int = 4000,
    seed: int = 42,
    block_files: int = 10000,
    threads: int = 32,
    writers: int = 4,
    with_pixel_truth: bool = False,
    with_extra_truth: bool = False,
    with_rich_truth: bool = False,
) -> None:
    keys = truth.pixel_keys(with_pixel_truth, with_extra_truth, with_rich_truth)
    view_range(view)  # an unknown view raises here, not once per file in the reader threads
    # Loaded here, before the reader threads start: a first load writes the geometry cache.
    geom = load_geometry() if with_rich_truth else None
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # The walk is deterministic and costs minutes on GPFS, so its result is kept beside the
    # state for the run's lifetime and a resume reads it back instead of walking again. That
    # also pins the permutation: a file added to a tree mid-run would otherwise shift it.
    files_path = outdir / "files.txt"
    if files_path.exists() and (outdir / "state.json").exists():
        files = [Path(line) for line in files_path.read_text().splitlines()]
        print(f"Files        : {len(files)} from {files_path.name}")
    else:
        files = find_files(datadirs, apa)
        if not files:
            raise RuntimeError(f"no *anode{apa}.h5 under {datadirs}")
        _write_text(files_path, "\n".join(str(f) for f in files) + "\n")
    order = np.random.default_rng(seed).permutation(len(files))
    files = [files[i] for i in order]
    blocks = [files[i : i + block_files] for i in range(0, len(files), block_files)]
    print(f"Walked       : {len(files)} files in {time.time() - t0:.0f} s -> {len(blocks)} blocks")

    # The arguments that fix the permutation. A resume with any of them changed would mix two
    # permutations in one directory, so it is refused rather than continued.
    signature = {
        "datadirs": [str(Path(d).resolve()) for d in datadirs],
        "n_files": len(files),
        "apa": int(apa),
        "view": view.upper(),
        "seed": int(seed),
        "shard_size": int(shard_size),
        "block_files": int(block_files),
        "pixel_truth": "pixel_labels" in keys,
        "extra_truth": "pixel_trackid" in keys,
        "rich_truth": bool(with_rich_truth),
    }
    state_path, carry_path = outdir / "state.json", outdir / "carry.npz"
    for tmp in outdir.glob("shard_*.h5.tmp"):
        tmp.unlink()
    state = {"next_block": 0, "next_shard": 0, "n_written": 0}
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        if saved["signature"] != signature:
            raise RuntimeError(
                f"{outdir} holds a partial run with different arguments "
                f"({saved['signature']}); clear the directory or restore them"
            )
        state = saved["state"]
        print(
            f"  Resuming: {state['next_block']}/{len(blocks)} blocks done, "
            f"{state['next_shard']} shards, {state['n_written']} events written."
        )
    carry = _load_carry(carry_path, keys, with_rich_truth)

    def write_metadata(n_samples: int, n_shards: int) -> None:
        meta_out = {
            **{k: signature[k] for k in ("datadirs", "apa", "view", "seed", "shard_size")},
            "n_samples": n_samples,
            "n_shards": n_shards,
            "pixel_truth": signature["pixel_truth"],
            "extra_truth": signature["extra_truth"],
            "rich_truth": signature["rich_truth"],
            "label_format": truth.LABEL_FORMAT,
            "class_names": truth.CLASS_NAMES,
        }
        if with_rich_truth:
            meta_out["vertex"] = {
                **vertices.PARAMETERS,
                "t0_ticks": vertices.DEFAULT_VERTEX_T0_TICKS,
                "type_names": vertices.VERTEX_TYPE_NAMES,
            }
        _write_text(outdir / "metadata.json", json.dumps(meta_out, indent=2))

    def read_one(fp: Path) -> list[Event]:
        try:
            return read_file(fp, apa, view, keys, with_rich_truth, geom)
        except Exception as e:  # noqa: BLE001 - one unreadable file must not end the run
            _warn_once(f"skipping {fp}: {e}")
            return []

    with ProcessPoolExecutor(max_workers=writers) as pool:
        _prefork(pool, writers)
        for b in range(state["next_block"], len(blocks)):
            t1 = time.time()
            with ThreadPoolExecutor(max_workers=threads) as ex:
                per_file = list(ex.map(read_one, blocks[b]))
            events = carry + [e for lst in per_file for e in lst]
            n_read = len(events) - len(carry)
            dt = time.time() - t1
            print(
                f"  block {b + 1}/{len(blocks)}: {len(blocks[b])} files, {n_read} events "
                f"in {dt:.0f} s ({len(blocks[b]) / max(dt, 1e-9):.0f} files/s)"
            )

            perm = np.random.default_rng([seed, b]).permutation(len(events))
            events = [events[i] for i in perm]
            n_full = len(events) // shard_size
            carry = events[n_full * shard_size :]
            # Shards are assembled one at a time with at most `writers` in flight, and each
            # shard's events are released as soon as its arrays exist: assembling a whole block
            # up front held every event twice and put cluster 345 over its memory limit.
            pending: list = []
            for k in range(n_full):
                lo, hi = k * shard_size, (k + 1) * shard_size
                arrays = assemble(events[lo:hi], keys, with_rich_truth)
                events[lo:hi] = [None] * shard_size
                path = outdir / f"shard_{state['next_shard'] + k:05d}.h5"
                pending.append(pool.submit(write_shard, path, arrays))
                del arrays
                if len(pending) >= writers:
                    name, n_pix, n_img = pending.pop(0).result()
                    print(f"    wrote {name}  ({n_pix} pixels, {n_img} images)")
            for fut in pending:
                name, n_pix, n_img = fut.result()
                print(f"    wrote {name}  ({n_pix} pixels, {n_img} images)")
            del events, per_file
            _release_heap()

            state = {
                "next_block": b + 1,
                "next_shard": state["next_shard"] + n_full,
                "n_written": state["n_written"] + n_full * shard_size,
            }
            _save_carry(carry_path, carry, keys, with_rich_truth)
            write_metadata(state["n_written"], state["next_shard"])
            _write_text(state_path, json.dumps({"signature": signature, "state": state}, indent=2))

        if carry:
            path = outdir / f"shard_{state['next_shard']:05d}.h5"
            name, n_pix, n_img = write_shard(path, assemble(carry, keys, with_rich_truth))
            print(f"    wrote {name}  ({n_pix} pixels, {n_img} images, trailing)")
            state["next_shard"] += 1
            state["n_written"] += len(carry)

    write_metadata(state["n_written"], state["next_shard"])
    carry_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    files_path.unlink(missing_ok=True)
    print(
        f"\nDone. {state['n_written']} events in {state['next_shard']} shards + metadata.json "
        f"under {outdir} ({(time.time() - t0) / 60:.0f} min)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-shard one or more production HDF5 trees into flat HDF5 shard files."
    )
    parser.add_argument(
        "--datadir",
        required=True,
        nargs="+",
        help="Root directory of source HDF5 files; several roots are one "
        "dataset, shuffled together",
    )
    parser.add_argument("--apa", type=int, required=True, help="APA number")
    parser.add_argument("--view", default="W", help="Wire-plane view (U/V/W)")
    parser.add_argument("--outdir", required=True, help="Output directory for shard files")
    parser.add_argument(
        "--shard_size", type=int, default=4000, help="Images per shard (default: 4000)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for the shuffle (default: 42)"
    )
    parser.add_argument(
        "--block_files",
        type=int,
        default=20000,
        help="Source files read and shuffled together; sets the memory "
        "footprint (default: 20000, about 11 GB of events)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Concurrent file reads; IO-bound, so more than the CPUs (default: 32)",
    )
    parser.add_argument(
        "--writers", type=int, default=4, help="Processes compressing shards (default: 4)"
    )
    parser.add_argument(
        "--with_pixel_truth",
        action="store_true",
        help="Also store per-pixel class labels (pixel_labels)",
    )
    parser.add_argument(
        "--with_extra_truth",
        action="store_true",
        help="Also store pixel_energyfrac/pixel_trackid/pixel_truth_q (implies --with_pixel_truth)",
    )
    parser.add_argument(
        "--with_rich_truth",
        action="store_true",
        help="Also store the 2nd contributor, pixel PDG, vertex truth and the mcpart/simchnl "
        "particle tables (implies --with_extra_truth; needs the trackid_pid_map files)",
    )
    args = parser.parse_args()

    create_shards(
        datadirs=args.datadir,
        apa=args.apa,
        view=args.view,
        outdir=args.outdir,
        shard_size=args.shard_size,
        seed=args.seed,
        block_files=args.block_files,
        threads=args.threads,
        writers=args.writers,
        with_pixel_truth=args.with_pixel_truth,
        with_extra_truth=args.with_extra_truth,
        with_rich_truth=args.with_rich_truth,
    )


if __name__ == "__main__":
    main()
