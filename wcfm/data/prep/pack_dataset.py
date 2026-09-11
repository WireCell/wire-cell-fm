"""
Preprocess the raw production HDF5 tree into a single packed .npz for fast,
in-RAM training, read back by `PackedDataset` (wcfm/data/packed.py).

Every event is read through `DirectDataset`, so the same anode is
selected, the same wire-plane view is filtered, and channel coordinates are
rebased identically to what the training loop sees. The per-event sparse
tensors are concatenated into one file using a CSR `offsets` layout, in
dataset index order (no shuffle) — pack sample i == dataset sample i, which
makes exact equivalence tests possible.

Event truth is always packed (a few scalars per event). Per-pixel truth tiers
are on by default: npz members load lazily, so truth baked into the pack costs
training nothing. Disable with --no_pixel_truth /
--no_extra_truth to shrink the file.

Output arrays (CSR over E events):
    coords    (ΣN, 2) int32     rebased (channel, tick) per pixel
    features  (ΣN, 1) float32   ADC charge per pixel
    offsets   (E+1,)  int64     event i = rows offsets[i]:offsets[i+1]
    labels    (E,) int64        event class (0=numuCC,1=nueCC,2=NC,-1=unknown)
    nu_pdg, nu_ccnc, nu_intType (E,) int64
    nu_energy (E,) float32
    vertex_xyz (E, 3) float32
    event_key (E,) str
  with pixel truth (default):
    pixel_labels (ΣN,) int8     class labels 0-6 (0=Background/no-truth),
                                CSR-aligned to coords/offsets
  with extra truth (default):
    pixel_energyfrac (ΣN,) f32  truth-overlap score (frame_energyfrac_1st)
    pixel_trackid    (ΣN,) i32  signed truth track id (frame_trackid_1st)
    pixel_truth_q    (ΣN,) f32  truth charge (frame_total_numelectrons)
  scalars: apa (int), view (str), truth_format ("classes7_v1"), class_names

Usage:
    python -m wcfm.data.prep.pack_dataset \\
        --datadir /path/to/production --apa 0 --view W \\
        --out_path /path/to/packed_apa0_W.npz \\
        --cache_dir /path/to/cache --num_workers 8
    # quick test:
    python -m wcfm.data.prep.pack_dataset --datadir ... --out_path test.npz --n_subset 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from wcfm.data.direct import DirectDataset

TRUTH_FORMAT = "classes7_v1"
CLASS_NAMES = ["Background", "Track", "Shower", "Michel", "DeltaRay", "Blip", "Other"]

# Per-pixel truth: meta key -> output dtype.
PIXEL_SPEC = {"pixel_labels": np.int8}
EXTRA_SPEC = {
    "pixel_energyfrac": np.float32,
    "pixel_trackid":    np.int32,
    "pixel_truth_q":    np.float32,
}


class _Extract(torch.utils.data.Dataset):
    """
    Thin wrapper that returns plain numpy (picklable across DataLoader workers)
    instead of Voxels, so reading the ~10k raw files can be parallelised.
    """

    def __init__(self, ds: DirectDataset, pixel_keys: tuple):
        self.ds = ds
        self.pixel_keys = pixel_keys

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        vox, meta = self.ds[i]
        # Convert to plain numpy / python scalars so NO torch tensors cross the
        # DataLoader worker boundary. torch shares tensors via shared-memory
        # file descriptors; accumulating ~100k of them otherwise exhausts the
        # fd limit ("unable to mmap ...: Cannot allocate memory").
        out_meta = {
            "label":      int(meta["label"]),
            "nu_pdg":     int(meta["nu_pdg"]),
            "nu_ccnc":    int(meta["nu_ccnc"]),
            "nu_intType": int(meta["nu_intType"]),
            "nu_energy":  float(meta["nu_energy"]),
            "vertex_xyz": np.asarray(meta["vertex_xyz"], dtype=np.float32),
            "event_key":  str(meta["event_key"]),
        }
        for k in self.pixel_keys:
            out_meta[k] = np.asarray(meta[k])
        coords = vox.coordinate_tensor.numpy().astype(np.int32)   # (N, 2)
        feats  = vox.feature_tensor.numpy().astype(np.float32)    # (N, 1)
        return coords, feats, out_meta


def _collate_first(batch):
    # batch_size == 1, so just unwrap the single item.
    return batch[0]


def pack_dataset(
    datadir: str,
    out_path: str,
    apa: int = 0,
    view: str = "W",
    cache_dir: str = "./data",
    n_subset: int = -1,
    with_pixel_truth: bool = True,
    with_extra_truth: bool = True,
    num_workers: int = 8,
    log_every: int = 500,
) -> None:
    # Belt-and-suspenders against fd exhaustion when streaming ~100k worker results.
    torch.multiprocessing.set_sharing_strategy("file_system")

    if with_extra_truth:
        with_pixel_truth = True
    pixel_spec = {}
    if with_pixel_truth:
        pixel_spec.update(PIXEL_SPEC)
    if with_extra_truth:
        pixel_spec.update(EXTRA_SPEC)

    ds = DirectDataset(
        datadir=datadir,
        apa=apa,
        view=view,
        use_cache=True,
        cache_dir=cache_dir,
        return_pixel_truth=with_pixel_truth,
        return_extra_truth=with_extra_truth,
    )

    n_total = len(ds)
    n = n_total if n_subset < 0 else min(n_subset, n_total)
    print(f"Packing {n}/{n_total} events  (apa={apa}, view={view}, "
          f"pixel_truth={with_pixel_truth}, extra_truth={with_extra_truth})")

    wrapped = Subset(_Extract(ds, tuple(pixel_spec)), list(range(n)))
    loader = DataLoader(
        wrapped,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_first,
    )

    coords_list, feats_list, sizes = [], [], []
    labels, pdg, ccnc, itype, energy, vtx, keys = [], [], [], [], [], [], []
    pixel_lists = {k: [] for k in pixel_spec}

    for k, (coords, feats, meta) in enumerate(loader):
        coords_list.append(coords)
        feats_list.append(feats)
        sizes.append(coords.shape[0])

        labels.append(meta["label"])
        pdg.append(meta["nu_pdg"])
        ccnc.append(meta["nu_ccnc"])
        itype.append(meta["nu_intType"])
        energy.append(meta["nu_energy"])
        vtx.append(meta["vertex_xyz"])
        keys.append(meta["event_key"])

        for kk, lst in pixel_lists.items():
            arr = meta[kk]
            if arr.shape[0] != coords.shape[0]:
                raise RuntimeError(
                    f"{kk}/coords length mismatch at event {k}: "
                    f"{arr.shape[0]} vs {coords.shape[0]} (key={meta['event_key']})"
                )
            lst.append(arr)

        if (k + 1) % log_every == 0:
            print(f"  {k + 1}/{n}  (pixels so far: {sum(sizes)})")

    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    # Build the output arrays one at a time, freeing each accumulation list
    # right after its concatenate — roughly halves the peak RSS of the job
    # (at ~200k mixed events the per-pixel lists alone are tens of GB).
    out = {
        "offsets":    offsets,
        "apa":        np.int64(apa),
        "view":       str(view),
        "labels":     np.array(labels, np.int64),
        "nu_pdg":     np.array(pdg, np.int64),
        "nu_ccnc":    np.array(ccnc, np.int64),
        "nu_intType": np.array(itype, np.int64),
        "nu_energy":  np.array(energy, np.float32),
        "vertex_xyz": (np.stack(vtx, axis=0).astype(np.float32) if vtx
                       else np.zeros((0, 3), np.float32)),
        "event_key":  np.array(keys),
    }
    out["coords"] = (np.concatenate(coords_list, axis=0).astype(np.int32)
                     if coords_list else np.zeros((0, 2), np.int32))
    coords_list.clear()
    out["features"] = (np.concatenate(feats_list, axis=0).astype(np.float32)
                       if feats_list else np.zeros((0, 1), np.float32))
    feats_list.clear()
    if pixel_spec:
        out["truth_format"] = TRUTH_FORMAT
        out["class_names"]  = np.array(CLASS_NAMES)
        for kk, dt in pixel_spec.items():
            arr = (np.concatenate(pixel_lists[kk], axis=0).astype(dt)
                   if pixel_lists[kk] else np.zeros((0,), dt))
            assert arr.shape[0] == int(offsets[-1]), (
                f"{kk} total {arr.shape[0]} != offsets[-1] {int(offsets[-1])}")
            out[kk] = arr
            pixel_lists[kk].clear()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Saved {n} events, {int(offsets[-1])} pixels -> {out_path}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack the production HDF5 tree into a single .npz for in-RAM training."
    )
    parser.add_argument("--datadir",    required=True,
                        help="Root directory of source HDF5 files")
    parser.add_argument("--out_path",   required=True,           help="Output .npz path")
    parser.add_argument("--apa",        type=int, default=0,     help="APA number (default: 0)")
    parser.add_argument("--view",       default="W",
                        help="Wire-plane view U/V/W (default: W)")
    parser.add_argument("--cache_dir",  required=True,
                        help="Dataset index cache directory")
    parser.add_argument("--n_subset",   type=int, default=-1,
                        help="Pack only the first N events in index order (default: all). "
                             "Note: on a mixed nominal+nueswap root the index is "
                             "flavor-blocked, so a subset is effectively single-flavor "
                             "— use for smoke tests only")
    parser.add_argument("--no_pixel_truth", action="store_true",
                        help="Skip per-pixel class labels (pixel truth is packed by default)")
    parser.add_argument("--no_extra_truth", action="store_true",
                        help="Skip energyfrac/trackid/truth_q (packed by default)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers for the raw read (default: 8)")
    parser.add_argument("--log_every",  type=int, default=500)
    args = parser.parse_args()

    pack_dataset(
        datadir=args.datadir,
        out_path=args.out_path,
        apa=args.apa,
        view=args.view,
        cache_dir=args.cache_dir,
        n_subset=args.n_subset,
        with_pixel_truth=not args.no_pixel_truth,
        with_extra_truth=not args.no_extra_truth,
        num_workers=args.num_workers,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
