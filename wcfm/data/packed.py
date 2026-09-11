"""The "packed" backend: one packed .npz file, held in RAM, sliced per sample.

The packed file is originally built by walking the production HDF5 files through the "direct"
reader and concatenating every event into one file with a CSR offsets layout.

Loading is a single np.load, after which __getitem__ is a pure tensor slice. 
Same per-sample content as sharded; different container and IO pattern.
All dataset is in RAM, then you can batch freely.

Map-style, so it goes through `collate` in the DataLoader (see build.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from wcfm.data.voxels import Batch, voxels_from

# Per-pixel truth arrays optionally present in a pack (CSR-aligned to coords).
_PIXEL_KEYS = ("pixel_labels",)
_EXTRA_KEYS = ("pixel_energyfrac", "pixel_trackid", "pixel_truth_q")


class PackedDataset(Dataset):
    """
    Map-style dataset over a packed .npz

    __getitem__ returns (voxels: Voxels, meta: dict), behaviorally identical
    to DirectDataset — use with wcfm.data.collate. Event truth is always
    present in meta; per-pixel tiers are opt-in flags.

    npz members are lazy: np.load reads a zip member only when accessed, so
    the per-pixel truth baked into the pack costs nothing (RAM or time)
    unless return_pixel_truth / return_extra_truth is set. Members cannot be
    memory-mapped (numpy ignores mmap_mode for zip archives) — a pack is
    loaded fully into RAM, array by array, on first access.

    Expected arrays in the .npz (CSR over E events):
        coords    (ΣN, 2) int32     rebased (channel, tick) per pixel
        features  (ΣN, 1) float32   ADC charge per pixel
        offsets   (E+1,)  int64     event i occupies rows offsets[i]:offsets[i+1]
        labels    (E,) int64        event class (0=numuCC,1=nueCC,2=NC,-1=unknown)
        nu_pdg, nu_ccnc, nu_intType (E,) int64
        nu_energy (E,) float32
        vertex_xyz (E, 3) float32
        event_key (E,) str
      optional per-pixel truth (pack built with --with_pixel_truth / default):
        pixel_labels (ΣN,) int8     class labels 0-6 (0=Background/no-truth)
        pixel_energyfrac (ΣN,) f32 · pixel_trackid (ΣN,) i32 (signed)
        pixel_truth_q (ΣN,) f32
      scalars: apa (int), view (str), truth_format (str)

    Args:
        npz_path:           Path to the packed .npz.
        return_pixel_truth: Add pixel_labels to meta (loads the array to RAM).
        return_extra_truth: Add energyfrac/trackid/truth_q to meta as well.
    """

    def __init__(
        self,
        npz_path: str | Path,
        return_pixel_truth: bool = False,
        return_extra_truth: bool = False,
    ):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Packed dataset not found: {self.npz_path}")
        if return_extra_truth:
            return_pixel_truth = True
        self.return_pixel_truth = return_pixel_truth
        self.return_extra_truth = return_extra_truth

        d = np.load(self.npz_path, allow_pickle=True)
        files = set(d.files)

        # Core sparse payload — pulled fully into RAM as torch tensors.
        self.coords  = torch.from_numpy(np.ascontiguousarray(d["coords"])).to(torch.int32)
        self.feats   = torch.from_numpy(np.ascontiguousarray(d["features"])).to(torch.float32)
        self.offsets = torch.from_numpy(np.ascontiguousarray(d["offsets"])).to(torch.int64)

        # Event truth (always written by `wcfm.data.prep.pack_dataset`).
        missing = {"labels", "nu_pdg", "nu_ccnc", "nu_intType",
                   "nu_energy", "vertex_xyz", "event_key"} - files
        if missing:
            raise ValueError(
                f"{self.npz_path} is missing event-truth arrays {sorted(missing)}; "
                f"re-pack with `python -m wcfm.data.prep.pack_dataset`."
            )
        self.labels     = d["labels"].astype(np.int64)
        self.nu_pdg     = d["nu_pdg"].astype(np.int64)
        self.nu_ccnc    = d["nu_ccnc"].astype(np.int64)
        self.nu_intType = d["nu_intType"].astype(np.int64)
        self.nu_energy  = d["nu_energy"].astype(np.float32)
        self.vertex_xyz = torch.from_numpy(
            np.ascontiguousarray(d["vertex_xyz"])).to(torch.float32)
        self.event_key  = d["event_key"]

        # Optional per-pixel truth — loaded only when requested.
        self.pixel_truth = {}
        if return_pixel_truth:
            wanted = _PIXEL_KEYS + (_EXTRA_KEYS if return_extra_truth else ())
            missing = set(wanted) - files
            if missing:
                raise ValueError(
                    f"{self.npz_path} has no {sorted(missing)}; re-pack with "
                    f"the matching --with_pixel_truth/--with_extra_truth flags."
                )
            for k in wanted:
                self.pixel_truth[k] = d[k]

        # Provenance / metadata.
        self.apa  = int(d["apa"]) if "apa" in files else None
        self.view = str(d["view"]) if "view" in files else None

    def __len__(self) -> int:
        return self.offsets.numel() - 1

    def __getitem__(self, idx: int) -> Batch:
        a = int(self.offsets[idx])
        b = int(self.offsets[idx + 1])

        # .clone() both tensors so downstream in-place ops (e.g. the training
        # loop's FeatureLogTransform) never touch the shared RAM buffer —
        # an un-cloned feats view would silently double-normalize the pack
        # from epoch 2 onward.
        coords = self.coords[a:b].clone()
        feats  = self.feats[a:b].clone()

        n = b - a
        offsets = torch.tensor([0, n], dtype=torch.int64)

        vox = voxels_from(coords, feats, offsets)

        meta = {
            "label":      int(self.labels[idx]),
            "nu_pdg":     int(self.nu_pdg[idx]),
            "nu_ccnc":    int(self.nu_ccnc[idx]),
            "nu_intType": int(self.nu_intType[idx]),
            "nu_energy":  float(self.nu_energy[idx]),
            "vertex_xyz": self.vertex_xyz[idx].clone(),
            "event_key":  str(self.event_key[idx]),
        }
        for k, arr in self.pixel_truth.items():
            meta[k] = arr[a:b]

        return Batch(vox, meta)
