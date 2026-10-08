"""The "direct" backend: read the production HDF5 as it comes off the simulation.

Events sit in many files under one directory. `DirectDataset` scans it, indexes every sparse
sample and reads them on demand, one file open per sample, which is slow on GPFS: it is the
legacy option and not recommended for training. The functions above it parse one event out
of an open file and are shared with `wcfm.data.prep.create_shards`, which reads whole files
instead, so a shard set is the production exactly as this reader sees it.

An event is one group of a `*_pixeldata-anode<apa>.h5`. Its event truth is in the
`*_metadata.h5` beside it, and the particle lists the rich tier stores are in the
`*_trackid_pid_map.h5` beside it. Per-pixel labels (`frame_label_*`, `<event>/mcpart/labels`)
are not simulation output: a labeller writes them into the files before they are read here.
"""

from __future__ import annotations

import contextlib
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

from wcfm.data import truth
from wcfm.data.vertices import DEFAULT_VERTEX_T0_TICKS, DIST_CLIP, vertex_truth
from wcfm.data.voxels import Batch, voxels_from
from wcfm.data.wire_geometry import WireGeometry

# Default channel ranges for each wire-plane view.
# Channels 0-799 -> U plane, 800-1599 -> V plane, 1600-2649 -> W plane.
DEFAULT_VIEW_RANGES: dict[str, tuple[int, int]] = {
    "U": (0, 800),
    "V": (800, 1600),
    "W": (1600, 2650),
}

FRAME_NAME = "frame_rebinned_reco"

# Per-pixel truth: meta key -> sparse HDF5 frame. Every frame is stored as float; integer keys
# are rounded on read. Track ids can be negative (a G4-dropped particle; its parent is abs(id))
# and are kept as stored.
TRUTH_FRAMES = {
    "pixel_labels": "frame_label_1st",
    "pixel_energyfrac": "frame_energyfrac_1st",
    "pixel_trackid": "frame_trackid_1st",
    "pixel_truth_q": "frame_total_numelectrons",
    "pixel_labels2": "frame_label_2nd",
    "pixel_trackid2": "frame_trackid_2nd",
    "pixel_energyfrac2": "frame_energyfrac_2nd",
}

# Table column -> dataset under `<event>/<table>` in the trackid_pid_map file.
TABLE_SOURCES = {
    "mcpart": {
        "trackid": "track_ids",
        "pid": "pids",
        "motherid": "mother_ids",
        "mother_pid": "mother_pids",
        "proc": "processes",
        "endproc": "end_processes",
        "status": "statuses",
        "ndaughters": "ndaughters",
        "ntrajpts": "ntrajpts",
        "label": "labels",
        "mass": "masses",
        "start_xyzt": "start_xyzts",
        "end_xyzt": "end_xyzts",
        "start_mom": "start_moms",
        "end_mom": "end_moms",
    },
    "simchnl": {
        "trackid": "track_ids",
        "pid": "pids",
        "motherid": "mother_ids",
        "mother_pid": "mother_pids",
        "proc": "processes",
        "energy": "energies",
    },
}


def view_range(view: str, view_ranges: dict[str, tuple[int, int]] | None = None) -> tuple[int, int]:
    """The `[start, end)` channel range of a wire-plane view. Raises on an unknown view."""
    ranges = view_ranges if view_ranges is not None else DEFAULT_VIEW_RANGES
    view = view.upper()
    if view not in ranges:
        raise ValueError(f"view must be one of {list(ranges)}, got {view!r}")
    return ranges[view]


def companion_path(pixeldata_path: Path, apa: int, kind: str) -> Path:
    """The `_<kind>.h5` beside a `_pixeldata-anode<apa>.h5` (or older `_anode<apa>.h5`):
    `kind` is `metadata` or `trackid_pid_map`."""
    suffix = f"_pixeldata-anode{apa}.h5"
    if pixeldata_path.name.endswith(suffix):
        basename = pixeldata_path.name[: -len(suffix)]
    else:
        basename = pixeldata_path.stem
    return pixeldata_path.parent / f"{basename}_{kind}.h5"


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


def pixel_truth(
    group: h5py.Group,
    reco_coords: np.ndarray,
    ch_start: int,
    ch_end: int,
    keys: dict[str, type],
    warn: Callable[[str], None],
) -> dict:
    """The `TRUTH_FRAMES` among `keys`, aligned to the view-filtered reco pixel order of
    `reco_coords` (the unfiltered `(N, 2)` frame coords). A reco pixel a frame has no hit on
    carries 0. Truth hits sit on reco pixels only, so the alignment drops none."""
    mask_reco = (reco_coords[:, 0] >= ch_start) & (reco_coords[:, 0] < ch_end)
    reco_view = reco_coords[mask_reco]
    out = {k: np.zeros(len(reco_view), dtype=dt) for k, dt in keys.items() if k in TRUTH_FRAMES}
    if "frame_label_1st" not in group:
        warn(
            f"frame_label_1st not found in {group.file.filename}[{group.name}]; pixel truth "
            "defaults to 0 (Background). Label the production before reading it."
        )
        return out

    rows_by_coords: dict[bytes, np.ndarray] = {}
    for key in out:
        frame = TRUTH_FRAMES[key]
        if frame not in group:
            warn(f"{frame} not found in {group.file.filename}; {key} defaults to 0.")
            continue
        coords_f = group[frame]["coords"][()]
        feats_f = group[frame]["features"][()]
        m = (coords_f[:, 0] >= ch_start) & (coords_f[:, 0] < ch_end)
        f_coords, f_feats = coords_f[m], feats_f[m]
        # The frames of one contributor slot share coords, so their row lookup is built once.
        rows = rows_by_coords.get(f_coords.tobytes())
        if rows is None:
            lookup = {(int(c[0]), int(c[1])): i for i, c in enumerate(f_coords)}
            rows = np.array(
                [lookup.get((int(c[0]), int(c[1])), -1) for c in reco_view], dtype=np.int64
            )
            rows_by_coords[f_coords.tobytes()] = rows
        has = rows >= 0
        vals = f_feats[rows[has]]
        if np.issubdtype(out[key].dtype, np.integer) and vals.dtype.kind == "f":
            vals = np.rint(vals)
        out[key][has] = vals.astype(out[key].dtype)
    return out


def read_tables(trackmap: h5py.Group | None, where: str, warn: Callable[[str], None]) -> dict:
    """The event's `mcpart` and `simchnl` table columns from its trackid_pid_map group. A
    missing table or column is empty or zero-filled, with a warning."""
    out: dict[str, np.ndarray] = {}
    for t, sources in TABLE_SOURCES.items():
        grp = trackmap[t] if trackmap is not None and t in trackmap else None
        if grp is None:
            warn(f"no {t} table for {where}; it is stored empty.")
        n = len(grp[sources["trackid"]]) if grp is not None else 0
        for c, src in sources.items():
            name = f"{t}_{c}"
            if grp is not None and src in grp:
                out[name] = grp[src][()].astype(truth.TABLE_COLUMNS[name][1])
            else:
                if grp is not None:
                    warn(f"{t}/{src} not found for {where}; {name} defaults to 0.")
                out[name] = truth.empty_table_column(name, n)
    return out


def rich_truth(
    tables: dict[str, np.ndarray],
    vertex_xyz: np.ndarray,
    coords: np.ndarray,
    pixel_trackid: np.ndarray,
    *,
    geom: WireGeometry,
    apa: int,
    view: str,
) -> dict:
    """`pixel_pdg`, `pixel_vertex_dist` and the `vertex_*` columns of one event, from its
    tables and this view's reco pixels (`coords`, channel rebased) and their track ids.

    `pixel_pdg` looks a track id up in `simchnl`, then `mcpart`, then takes the PDG of its
    parent `abs(id)`: `simchnl` is the only list holding the G4-dropped particles."""
    pdg = dict(zip(tables["mcpart_trackid"].tolist(), tables["mcpart_pid"].tolist(), strict=True))
    pdg.update(zip(tables["simchnl_trackid"].tolist(), tables["simchnl_pid"].tolist(), strict=True))
    pixel_pdg = np.array(
        [pdg.get(t, pdg.get(abs(t), 0)) if t else 0 for t in pixel_trackid.tolist()], np.int32
    )
    if len(tables["mcpart_trackid"]) == 0:
        empty = {k: truth.empty_table_column(k) for k in truth.TABLE_COLUMNS if k[:7] == "vertex_"}
        return {
            "pixel_pdg": pixel_pdg,
            "pixel_vertex_dist": np.full(len(coords), DIST_CLIP, np.float32),
            **empty,
        }
    mcpart = {c: tables[f"mcpart_{c}"] for c in truth.TABLES["mcpart"]}
    vertex, dist = vertex_truth(
        mcpart, vertex_xyz, coords, pixel_trackid, geom=geom, apa=apa, view=view
    )
    return {"pixel_pdg": pixel_pdg, "pixel_vertex_dist": dist, **vertex}


def read_event(
    grp: h5py.Group,
    group: str,
    key: str,
    *,
    metadata: h5py.File | None,
    trackmap: h5py.File | None,
    ch_start: int,
    ch_end: int,
    keys: dict[str, type],
    rich: bool,
    geom: WireGeometry | None,
    apa: int,
    view: str,
    warn: Callable[[str], None],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One event: its view-filtered `(N, 2)` coords and `(N, 1)` features, and a meta dict of
    event truth plus the per-pixel `keys` and, with `rich`, the table columns.

    `metadata` and `trackmap` are the open companion files, None when absent. `geom` is
    needed with `rich` only."""
    raw_coords = grp[FRAME_NAME]["coords"][()]
    raw_feats = grp[FRAME_NAME]["features"][()]
    coords, feats = select_view(raw_coords, raw_feats, ch_start, ch_end)
    meta = event_truth(metadata, group, key, warn)
    if keys:
        meta.update(pixel_truth(grp, raw_coords, ch_start, ch_end, keys, warn))
    if rich:
        tables = read_tables(
            trackmap[group] if trackmap is not None and group in trackmap else None, key, warn
        )
        meta.update(tables)
        meta.update(
            rich_truth(
                tables,
                meta["vertex_xyz"],
                coords,
                meta["pixel_trackid"],
                geom=geom,
                apa=apa,
                view=view,
            )
        )
    return coords, feats, meta


def load_geometry() -> WireGeometry:
    """The wire geometry the rich tier projects vertices with, at the measured tick offset."""
    return WireGeometry.load(t0_ticks=DEFAULT_VERTEX_T0_TICKS)


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

    `__getitem__` returns `Batch(voxels, meta)`. Event-level truth is always present. The
    `wcfm.data.truth` tiers are opt-in, because HDF5 decompresses them on every read; each
    implies the ones before it.
    """

    def __init__(
        self,
        datadir: str | Path,
        apa: int,
        view: str,
        use_cache: bool = True,
        cache_dir: str | Path = "./data",
        view_ranges: dict[str, tuple[int, int]] | None = None,
        return_pixel_truth: bool = False,
        return_extra_truth: bool = False,
        return_rich_truth: bool = False,
    ):
        self.datadir = Path(datadir)
        self.apa = int(apa)
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

        self.truth_keys = truth.pixel_keys(
            return_pixel_truth, return_extra_truth, return_rich_truth
        )
        self.return_rich_truth = return_rich_truth
        self.geom = load_geometry() if return_rich_truth else None
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
        mpath = companion_path(s.path, self.apa, "metadata")
        tpath = companion_path(s.path, self.apa, "trackid_pid_map")
        with contextlib.ExitStack() as stack:
            f = stack.enter_context(h5py.File(s.path, "r"))
            metadata = stack.enter_context(h5py.File(mpath, "r")) if mpath.exists() else None
            trackmap = (
                stack.enter_context(h5py.File(tpath, "r"))
                if self.return_rich_truth and tpath.exists()
                else None
            )
            coords, feats, meta = read_event(
                f[s.group],
                s.group,
                event_key,
                metadata=metadata,
                trackmap=trackmap,
                ch_start=self.ch_start,
                ch_end=self.ch_end,
                keys=self.truth_keys,
                rich=self.return_rich_truth,
                geom=self.geom,
                apa=self.apa,
                view=self.view,
                warn=self._warn_once,
            )
        meta["vertex_xyz"] = torch.from_numpy(meta["vertex_xyz"])
        offsets = torch.tensor([0, coords.shape[0]], dtype=torch.int64)
        return Batch(voxels_from(torch.from_numpy(coords), torch.from_numpy(feats), offsets), meta)
