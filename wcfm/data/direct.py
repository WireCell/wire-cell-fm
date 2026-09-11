"""The "direct" backend: read the production HDF5 as it comes off the simulation.
This is the legacy option, and is not recommended for training. 

Events sit on different files in a certain directory.
The dataset scans it and builds an index of all the sparse samples, then reads them on demand.
Because many files need to accessed to build a batch, it is slow and memory-inefficient

"""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from wcfm.data.voxels import Batch, voxels_from

# Default channel ranges for each wire-plane view.
# Channels 0-799 -> U plane, 800-1599 -> V plane, 1600-2649 -> W plane.
DEFAULT_VIEW_RANGES: dict[str, tuple[int, int]] = {
    "U": (0, 800),
    "V": (800, 1600),
    "W": (1600, 2650),
}


@dataclass(frozen=True)
class SampleIndex:
    """One sample: a file path, and a group inside it"""
    path: Path
    group: str


def _classify(nu_pdg: int, nu_ccnc: int) -> int:
    if nu_ccnc == 1:
        return 2          # NC — any flavour
    if abs(nu_pdg) == 14 and nu_ccnc == 0:
        return 0          # numuCC (nu or anti-nu)
    if abs(nu_pdg) == 12 and nu_ccnc == 0:
        return 1          # nueCC (nu or anti-nu)
    return -1             # skip (e.g. nu_tau CC)


class DirectDataset(Dataset):
    """Map-style reader over the production HDF5 tree.

    Scans recursively for `*anode{APA}.h5`, treats each `/<group>/<frame_name>` as one sparse
    sample, keeps a single wire-plane view, and caches the file index to disk keyed by a hash of
    the resolved root so two datasets never share a cache file.

    `__getitem__` returns `(voxels, meta)`. Event-level truth is always present.
    Per-pixel truth are opt-in, because HDF5 decompresses them on every read.
    """

    _EXTRA_TRUTH_FRAMES = {
        "pixel_energyfrac": ("frame_energyfrac_1st", np.float32),
        "pixel_trackid": ("frame_trackid_1st", np.int32),
        "pixel_truth_q": ("frame_total_numelectrons", np.float32),
    }

    def __init__(
        self,
        datadir: str | Path,
        apa: int,
        view: str,
        use_cache: bool = True,
        cache_dir: str | Path = "./data",
        view_ranges: dict[str, tuple[int, int]] | None = None,
        frame_name: str = "frame_rebinned_reco",
        return_pixel_truth: bool = False,
        return_extra_truth: bool = False,
    ):
        self.datadir = Path(datadir)
        self.apa = int(apa)
        self.frame_name = frame_name

        self.view_ranges: dict[str, tuple[int, int]] = (
            view_ranges if view_ranges is not None else DEFAULT_VIEW_RANGES
        )
        self.view = view.upper()
        if self.view not in self.view_ranges:
            raise ValueError(f"view must be one of {list(self.view_ranges)}, got {view!r}")
        self.ch_start, self.ch_end = self.view_ranges[self.view]

        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # A short hash of the resolved datadir, so two datasets with the same APA and view but
        # different roots never share a cache file.
        root_hash = hashlib.md5(str(self.datadir.resolve()).encode()).hexdigest()[:8]
        self.cache_file = (
            self.cache_dir / f"DirectDataset_APA{self.apa}_view{self.view}_{root_hash}_cache.pt"
        )

        self.return_pixel_truth = return_pixel_truth
        self.return_extra_truth = return_extra_truth
        self._warned_missing: set = set()

        self.samples: list[SampleIndex] = self._scan()
        if not self.samples:
            raise RuntimeError(f"No sparse samples found under {self.datadir} for APA {self.apa}")

    # -- cache -------------------------------------------------------------------

    def _save_index_pt(self, cache_file: Path, samples: list[SampleIndex]) -> None:
        data = [(str(s.path), s.group) for s in samples]
        # Atomic write (tmp + rename): concurrent jobs sharing a cache dir can never observe a
        # half-written index.
        tmp = cache_file.with_suffix(f".tmp.{os.getpid()}")
        torch.save(data, tmp)
        os.replace(tmp, cache_file)

    def _load_index_pt(self, cache_file: Path) -> list[SampleIndex]:
        data = torch.load(cache_file, map_location="cpu")
        return [SampleIndex(path=Path(p), group=g) for p, g in data]

    def _scan(self) -> list[SampleIndex]:
        if self.use_cache and self.cache_file.exists():
            return self._load_index_pt(self.cache_file)

        samples: list[SampleIndex] = []
        for fp in self.datadir.rglob(f"*anode{self.apa}.h5"):
            if not fp.is_file():
                continue
            try:
                with h5py.File(fp, "r") as f:
                    for group in f.keys():
                        grp = f[group]
                        if not isinstance(grp, h5py.Group) or self.frame_name not in grp:
                            continue
                        frame = grp[self.frame_name]
                        # Sparse format only: a subgroup carrying coords and features.
                        if (
                            isinstance(frame, h5py.Group)
                            and "coords" in frame
                            and "features" in frame
                        ):
                            samples.append(SampleIndex(path=fp, group=group))
            except OSError as e:
                warnings.warn(f"could not open {fp}: {e}", stacklevel=2)

        samples.sort(key=lambda s: (str(s.path), int(s.group)))
        if self.use_cache:
            self._save_index_pt(self.cache_file, samples)
        return samples

    # -- Dataset API -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Batch:
        s = self.samples[idx]

        with h5py.File(s.path, "r") as f:
            frame = f[s.group][self.frame_name]
            coords = torch.from_numpy(frame["coords"][()]).to(torch.int32)  # (N, 2)
            feats = torch.from_numpy(frame["features"][()]).to(torch.float32)  # (N,)

        # Keep only pixels in this view's channel range, then rebase the channel to 0.
        keep = (coords[:, 0] >= self.ch_start) & (coords[:, 0] < self.ch_end)
        coords = coords[keep].clone()
        feats = feats[keep].unsqueeze(1)  # Voxels wants (N, C)
        coords[:, 0] -= self.ch_start

        offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int64)

        meta = self._read_event_truth(s.path, s.group)
        if self.return_pixel_truth:
            meta.update(self._read_pixel_truth_arrays(s.path, s.group))
        return Batch(voxels_from(coords, feats, offsets), meta)

    # -- truth -------------------------------------------------------------------

    def _metadata_path(self, pixeldata_path: Path) -> Path:
        suffix = f"_pixeldata-anode{self.apa}.h5"
        if not pixeldata_path.name.endswith(suffix):
            # Older naming: "..._anode3.h5" without "_pixeldata" prefix.
            basename = pixeldata_path.stem   # drop .h5
        else:
            basename = pixeldata_path.name[: -len(suffix)]
        return pixeldata_path.parent / f"{basename}_metadata.h5"

    def _warn_once(self, metadata_path: Path, msg: str) -> None:
        key = str(metadata_path)
        if key not in self._warned_missing:
            warnings.warn(msg, stacklevel=2)
            self._warned_missing.add(key)

    # ------------------------------------------------------------------
    # Event-truth reader
    # ------------------------------------------------------------------

    def _unknown_metadata(self, event_key: str) -> dict:
        """Sentinel metadata dict used when the metadata file is missing/unreadable."""
        return {
            "label":      -1,
            "nu_pdg":     0,
            "nu_ccnc":    -1,
            "nu_intType": -1,
            "nu_energy":  0.0,
            "vertex_xyz": torch.zeros(3, dtype=torch.float32),
            "event_key":  event_key,
        }

    def _read_event_truth(self, pixeldata_path: Path, group: str) -> dict:
        event_key = f"{pixeldata_path.name}:{group}"
        metadata_path = self._metadata_path(pixeldata_path)

        if not metadata_path.exists():
            self._warn_once(metadata_path, f"Metadata file not found: {metadata_path}")
            return self._unknown_metadata(event_key)

        try:
            with h5py.File(metadata_path, "r") as f:
                row = f[group]["metadata"][0]
                nu_pdg     = int(row["nu_pdg"])
                nu_ccnc    = int(row["nu_ccnc"])
                nu_intType = int(row["nu_intType"])
                nu_energy  = float(row["nu_energy"])
                vx = float(row["nu_vertex_x"])
                vy = float(row["nu_vertex_y"])
                vz = float(row["nu_vertex_z"])
        except Exception as e:
            self._warn_once(
                metadata_path, f"Could not read metadata from {metadata_path}[{group}]: {e}"
            )
            return self._unknown_metadata(event_key)

        return {
            "label":      _classify(nu_pdg, nu_ccnc),
            "nu_pdg":     nu_pdg,
            "nu_ccnc":    nu_ccnc,
            "nu_intType": nu_intType,
            "nu_energy":  nu_energy,
            "vertex_xyz": torch.tensor([vx, vy, vz], dtype=torch.float32),
            "event_key":  event_key,
        }

    # ------------------------------------------------------------------
    # Pixel-level truth reader
    # ------------------------------------------------------------------

    # Extra per-pixel truth frames: meta key -> (HDF5 frame name, output dtype).
    # Track ids can be positive or negative (negative = G4-dropped secondary
    # of parent abs(id)); both are kept as-is.
    _EXTRA_TRUTH_FRAMES = {
        "pixel_energyfrac": ("frame_energyfrac_1st",     np.float32),
        "pixel_trackid":    ("frame_trackid_1st",        np.int32),
        "pixel_truth_q":    ("frame_total_numelectrons", np.float32),
    }

    def _empty_pixel_truth(self, n: int) -> dict:
        """All-fill pixel-truth dict of length n (label 0 = Background/no-truth)."""
        out = {"pixel_labels": np.zeros(n, dtype=np.int8)}
        if self.return_extra_truth:
            for key, (_, dtype) in self._EXTRA_TRUTH_FRAMES.items():
                out[key] = np.zeros(n, dtype=dtype)
        return out

    def _read_pixel_truth_arrays(self, pixeldata_path: Path, group: str) -> dict:
        """
        Read per-pixel truth frames (frame_label_1st and, with
        return_extra_truth, energyfrac/trackid/total_numelectrons) and
        return arrays aligned to the reco pixel order.

        Reco pixels with no truth hit carry the fill value (0 / 0.0).
        """
        try:
            with h5py.File(pixeldata_path, "r") as f:
                g = f[group]
                reco_coords = g[self.frame_name]["coords"][()]   # (N, 2) int32

                mask_reco = ((reco_coords[:, 0] >= self.ch_start) &
                             (reco_coords[:, 0] < self.ch_end))
                n_view = int(mask_reco.sum())

                if "frame_label_1st" not in g:
                    self._warn_once(
                        pixeldata_path,
                        f"frame_label_1st not found in {pixeldata_path}[{group}]; "
                        f"pixel truth defaults to 0 (Background). Pre-2026-06-11 "
                        f"productions (frame_pid_*) are no longer supported.",
                    )
                    return self._empty_pixel_truth(n_view)

                label_coords = g["frame_label_1st"]["coords"][()]    # (M, 2) int32
                label_feats  = g["frame_label_1st"]["features"][()]  # (M,)   int8

                extra_raw = {}
                if self.return_extra_truth:
                    for key, (frame, _) in self._EXTRA_TRUTH_FRAMES.items():
                        if frame in g:
                            extra_raw[key] = (g[frame]["coords"][()],
                                              g[frame]["features"][()])
                        else:
                            self._warn_once(
                                pixeldata_path,
                                f"{frame} not found in {pixeldata_path}; "
                                f"{key} defaults to 0.",
                            )
        except Exception as e:
            self._warn_once(
                pixeldata_path,
                f"Could not read pixel truth from {pixeldata_path}[{group}]: {e}",
            )
            return self._empty_pixel_truth(0)

        # Filter to this view's channel range (same logic as APASparseDataset)
        reco_view = reco_coords[mask_reco]
        mask_lbl  = ((label_coords[:, 0] >= self.ch_start) &
                     (label_coords[:, 0] < self.ch_end))
        lbl_view_coords = label_coords[mask_lbl]
        lbl_view_feats  = label_feats[mask_lbl]

        # Map each reco pixel to its row in the (view-filtered) truth frame.
        row_lookup = {
            (int(c[0]), int(c[1])): i for i, c in enumerate(lbl_view_coords)
        }
        rows = np.array(
            [row_lookup.get((int(c[0]), int(c[1])), -1) for c in reco_view],
            dtype=np.int64,
        )
        has = rows >= 0

        out = self._empty_pixel_truth(len(reco_view))
        out["pixel_labels"][has] = lbl_view_feats[rows[has]].astype(np.int8)

        for key, (coords_f, feats_f) in extra_raw.items():
            dtype = self._EXTRA_TRUTH_FRAMES[key][1]
            m = ((coords_f[:, 0] >= self.ch_start) &
                 (coords_f[:, 0] < self.ch_end))
            f_view_coords = coords_f[m]
            f_view_feats  = feats_f[m]

            # All truth frames share coords by construction (classify script
            # reuses frame_trackid coords) — reuse the label lookup when true.
            if (f_view_coords.shape == lbl_view_coords.shape
                    and np.array_equal(f_view_coords, lbl_view_coords)):
                f_rows, f_has = rows, has
            else:
                self._warn_once(
                    pixeldata_path,
                    f"{self._EXTRA_TRUTH_FRAMES[key][0]} coords differ from "
                    f"frame_label_1st in {pixeldata_path}; using per-frame lookup.",
                )
                lk = {(int(c[0]), int(c[1])): i
                      for i, c in enumerate(f_view_coords)}
                f_rows = np.array(
                    [lk.get((int(c[0]), int(c[1])), -1) for c in reco_view],
                    dtype=np.int64,
                )
                f_has = f_rows >= 0

            vals = f_view_feats[f_rows[f_has]]
            if np.issubdtype(dtype, np.integer) and vals.dtype.kind == "f":
                vals = np.rint(vals)
            out[key][f_has] = vals.astype(dtype)

        return out
