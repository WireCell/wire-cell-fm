"""The "direct" backend: read the production HDF5 as it comes off the simulation.

Events sit in many files under one directory. `DirectDataset` scans it, indexes every sparse
sample and reads them on demand, one file open per sample, which is slow on GPFS: it is the
legacy option and not recommended for training. The functions above it parse one event out
of an open file and are shared with `wcfm.data.prep.create_shards`, which reads whole files
instead, so a shard set is the production exactly as this reader sees it.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from collections.abc import Callable
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

FRAME_NAME = "frame_rebinned_reco"

# Extra per-pixel truth frames: meta key -> (HDF5 frame name, output dtype). Track ids can be
# positive or negative (negative = G4-dropped secondary of parent abs(id)); both are kept as-is.
EXTRA_TRUTH_FRAMES = {
    "pixel_energyfrac": ("frame_energyfrac_1st", np.float32),
    "pixel_trackid": ("frame_trackid_1st", np.int32),
    "pixel_truth_q": ("frame_total_numelectrons", np.float32),
}


def view_range(view: str, view_ranges: dict[str, tuple[int, int]] | None = None) -> tuple[int, int]:
    """The `[start, end)` channel range of a wire-plane view. Raises on an unknown view."""
    ranges = view_ranges if view_ranges is not None else DEFAULT_VIEW_RANGES
    view = view.upper()
    if view not in ranges:
        raise ValueError(f"view must be one of {list(ranges)}, got {view!r}")
    return ranges[view]


def metadata_path(pixeldata_path: Path, apa: int) -> Path:
    """The `_metadata.h5` beside a `_pixeldata-anode<apa>.h5` (or older `_anode<apa>.h5`)."""
    suffix = f"_pixeldata-anode{apa}.h5"
    if pixeldata_path.name.endswith(suffix):
        basename = pixeldata_path.name[: -len(suffix)]
    else:
        basename = pixeldata_path.stem
    return pixeldata_path.parent / f"{basename}_metadata.h5"


def is_sparse_frame(group: h5py.Group) -> bool:
    """Whether an event group carries the sparse reco frame the readers consume."""
    if FRAME_NAME not in group:
        return False
    frame = group[FRAME_NAME]
    return isinstance(frame, h5py.Group) and "coords" in frame and "features" in frame


def select_view(
    coords: np.ndarray, feats: np.ndarray, ch_start: int, ch_end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the pixels in `[ch_start, ch_end)`, rebase the channel to 0 and give the features a
    column: `(N, 2) int32` and `(N, 1) float32`, the layout `voxels_from` and the shards use."""
    keep = (coords[:, 0] >= ch_start) & (coords[:, 0] < ch_end)
    out = coords[keep].astype(np.int32, copy=True)
    out[:, 0] -= ch_start
    return out, feats[keep].astype(np.float32).reshape(-1, 1)


def classify(nu_pdg: int, nu_ccnc: int) -> int:
    """Event class: 0 numuCC, 1 nueCC, 2 NC of any flavour, -1 anything else (nu_tau CC)."""
    if nu_ccnc == 1:
        return 2
    if abs(nu_pdg) == 14 and nu_ccnc == 0:
        return 0
    if abs(nu_pdg) == 12 and nu_ccnc == 0:
        return 1
    return -1


def unknown_truth(event_key: str) -> dict:
    """The event-truth dict for an event whose metadata is missing or unreadable."""
    return {
        "label": -1,
        "nu_pdg": 0,
        "nu_ccnc": -1,
        "nu_intType": -1,
        "nu_energy": 0.0,
        "vertex_xyz": np.zeros(3, dtype=np.float32),
        "event_key": event_key,
    }


def event_truth(
    metadata: h5py.File | None, group: str, event_key: str, warn: Callable[[str], None]
) -> dict:
    """Event-level truth from an open `_metadata.h5`, `unknown_truth` when it is None or the
    row cannot be read. `warn` receives the reason once per file."""
    if metadata is None:
        warn(f"metadata file not found for {event_key}")
        return unknown_truth(event_key)
    try:
        row = metadata[group]["metadata"][0]
        nu_pdg = int(row["nu_pdg"])
        nu_ccnc = int(row["nu_ccnc"])
        return {
            "label": classify(nu_pdg, nu_ccnc),
            "nu_pdg": nu_pdg,
            "nu_ccnc": nu_ccnc,
            "nu_intType": int(row["nu_intType"]),
            "nu_energy": float(row["nu_energy"]),
            "vertex_xyz": np.array(
                [row["nu_vertex_x"], row["nu_vertex_y"], row["nu_vertex_z"]], dtype=np.float32
            ),
            "event_key": event_key,
        }
    except Exception as e:  # noqa: BLE001 - any unreadable row is "unknown"
        warn(f"could not read metadata for {event_key}: {e}")
        return unknown_truth(event_key)


def empty_pixel_truth(n: int, extra: bool) -> dict:
    """All-fill pixel truth of length n (label 0 = Background/no-truth)."""
    out = {"pixel_labels": np.zeros(n, dtype=np.int8)}
    if extra:
        for key, (_, dtype) in EXTRA_TRUTH_FRAMES.items():
            out[key] = np.zeros(n, dtype=dtype)
    return out


def pixel_truth(
    group: h5py.Group,
    reco_coords: np.ndarray,
    ch_start: int,
    ch_end: int,
    extra: bool,
    warn: Callable[[str], None],
) -> dict:
    """Per-pixel truth aligned to the view-filtered reco pixel order of `reco_coords` (the
    unfiltered `(N, 2)` frame coords). `pixel_labels` from `frame_label_1st` and, with
    `extra`, the `EXTRA_TRUTH_FRAMES`; reco pixels with no truth hit carry the fill value."""
    mask_reco = (reco_coords[:, 0] >= ch_start) & (reco_coords[:, 0] < ch_end)
    reco_view = reco_coords[mask_reco]
    n_view = int(mask_reco.sum())

    if "frame_label_1st" not in group:
        warn(
            f"frame_label_1st not found in {group.file.filename}[{group.name}]; pixel truth "
            "defaults to 0 (Background). Pre-2026-06-11 productions (frame_pid_*) are not "
            "supported."
        )
        return empty_pixel_truth(n_view, extra)

    label_coords = group["frame_label_1st"]["coords"][()]
    label_feats = group["frame_label_1st"]["features"][()]
    mask_lbl = (label_coords[:, 0] >= ch_start) & (label_coords[:, 0] < ch_end)
    lbl_view_coords = label_coords[mask_lbl]
    lbl_view_feats = label_feats[mask_lbl]

    # Map each reco pixel to its row in the view-filtered truth frame.
    row_lookup = {(int(c[0]), int(c[1])): i for i, c in enumerate(lbl_view_coords)}
    rows = np.array([row_lookup.get((int(c[0]), int(c[1])), -1) for c in reco_view], dtype=np.int64)
    has = rows >= 0

    out = empty_pixel_truth(n_view, extra)
    out["pixel_labels"][has] = lbl_view_feats[rows[has]].astype(np.int8)
    if not extra:
        return out

    for key, (frame, dtype) in EXTRA_TRUTH_FRAMES.items():
        if frame not in group:
            warn(f"{frame} not found in {group.file.filename}; {key} defaults to 0.")
            continue
        coords_f = group[frame]["coords"][()]
        feats_f = group[frame]["features"][()]
        m = (coords_f[:, 0] >= ch_start) & (coords_f[:, 0] < ch_end)
        f_view_coords, f_view_feats = coords_f[m], feats_f[m]
        # All truth frames share coords by construction (the classify script reuses the
        # frame_trackid coords), so the label lookup is reused when that holds.
        if f_view_coords.shape == lbl_view_coords.shape and np.array_equal(
            f_view_coords, lbl_view_coords
        ):
            f_rows, f_has = rows, has
        else:
            warn(
                f"{frame} coords differ from frame_label_1st in {group.file.filename}; "
                "using a per-frame lookup."
            )
            lk = {(int(c[0]), int(c[1])): i for i, c in enumerate(f_view_coords)}
            f_rows = np.array(
                [lk.get((int(c[0]), int(c[1])), -1) for c in reco_view], dtype=np.int64
            )
            f_has = f_rows >= 0
        vals = f_view_feats[f_rows[f_has]]
        if np.issubdtype(dtype, np.integer) and vals.dtype.kind == "f":
            vals = np.rint(vals)
        out[key][f_has] = vals.astype(dtype)
    return out


@dataclass(frozen=True)
class SampleIndex:
    """One sample: a file path, and a group inside it"""

    path: Path
    group: str


class DirectDataset(Dataset):
    """Map-style reader over the production HDF5 tree.

    Scans recursively for `*anode{APA}.h5`, treats each `/<group>/frame_rebinned_reco` as one
    sparse sample, keeps a single wire-plane view, and caches the file index to disk keyed by a
    hash of the resolved root so two datasets never share a cache file.

    `__getitem__` returns `Batch(voxels, meta)`. Event-level truth is always present. Per-pixel
    truth is opt-in, because HDF5 decompresses it on every read.
    """

    def __init__(
        self,
        datadir: str | Path,
        apa: int,
        view: str,
        use_cache: bool = True,
        cache_dir: str | Path = "./data",
        view_ranges: dict[str, tuple[int, int]] | None = None,
        frame_name: str = FRAME_NAME,
        return_pixel_truth: bool = False,
        return_extra_truth: bool = False,
    ):
        self.datadir = Path(datadir)
        self.apa = int(apa)
        self.frame_name = frame_name
        self.view = view.upper()
        self.ch_start, self.ch_end = view_range(view, view_ranges)

        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # A short hash of the resolved datadir, so two datasets with the same APA and view but
        # different roots never share a cache file.
        root_hash = hashlib.md5(str(self.datadir.resolve()).encode()).hexdigest()[:8]
        self.cache_file = (
            self.cache_dir / f"DirectDataset_APA{self.apa}_view{self.view}_{root_hash}_cache.pt"
        )

        self.return_pixel_truth = return_pixel_truth or return_extra_truth
        self.return_extra_truth = return_extra_truth
        self._warned: set[str] = set()

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
                        if isinstance(grp, h5py.Group) and is_sparse_frame(grp):
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

    def _warn_once(self, msg: str) -> None:
        if msg not in self._warned:
            warnings.warn(msg, stacklevel=2)
            self._warned.add(msg)

    def __getitem__(self, idx: int) -> Batch:
        s = self.samples[idx]
        event_key = f"{s.path.name}:{s.group}"

        with h5py.File(s.path, "r") as f:
            grp = f[s.group]
            raw_coords = grp[self.frame_name]["coords"][()]
            raw_feats = grp[self.frame_name]["features"][()]
            coords, feats = select_view(raw_coords, raw_feats, self.ch_start, self.ch_end)
            pixel = (
                pixel_truth(
                    grp,
                    raw_coords,
                    self.ch_start,
                    self.ch_end,
                    self.return_extra_truth,
                    self._warn_once,
                )
                if self.return_pixel_truth
                else {}
            )

        mpath = metadata_path(s.path, self.apa)
        if mpath.exists():
            with h5py.File(mpath, "r") as m:
                meta = event_truth(m, s.group, event_key, self._warn_once)
        else:
            meta = event_truth(None, s.group, event_key, self._warn_once)
        meta["vertex_xyz"] = torch.from_numpy(meta["vertex_xyz"])
        meta.update(pixel)

        offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int64)
        return Batch(voxels_from(torch.from_numpy(coords), torch.from_numpy(feats), offsets), meta)
