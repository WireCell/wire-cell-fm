"""The truth a production carries, by tier, and how one event's share is cut out of a CSR store.

Every reader and writer takes its key lists from here: `direct` fills them, `create_shards` and
`pack_dataset` store them, `sharded` and `packed` slice them back, `collate` stacks them.

Tiers, each implying the one before:

- event truth, always present: `labels`, `nu_*`, `vertex_xyz`, `event_key`.
- `pixel`: `pixel_labels`, the 7-class label of the 1st contributor (0 = Background, no truth).
- `extra`: the 1st contributor's energy fraction, signed track id and true charge.
- `rich`: everything else the truth files hold, per pixel (`RICH_PIXEL`) and per event as
  variable-length tables (`TABLES`).

Per-pixel arrays are CSR-aligned to the reco pixels through `offsets`. A table `t` is
CSR-aligned to the events through its own `<t>_offsets`; its columns are named `<t>_<column>`.

Quantities that are one expression of stored fields are not stored:
- overlap score = `1 - pixel_energyfrac` where `pixel_labels != 0`, else 0;
- instance id = `abs(pixel_trackid)`;
- 2nd contributor of a different class = `(pixel_labels2 != 0) & (pixel_labels2 != pixel_labels)`;
- vertex flag = `pixel_vertex_dist <= r`;
- kinetic energy of a particle = `mcpart_start_mom[:, 3] - mcpart_mass` (GeV), likewise at its end.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

LABEL_FORMAT = "classes7_v1"
CLASS_NAMES = ["Background", "Track", "Shower", "Michel", "DeltaRay", "Blip", "Other"]

PIXEL: dict[str, type] = {"pixel_labels": np.int8}
EXTRA: dict[str, type] = {
    "pixel_energyfrac": np.float32,
    "pixel_trackid": np.int32,
    "pixel_truth_q": np.float32,
}
# 0 in `pixel_labels2` / `pixel_trackid2` means the pixel has no 2nd contributor. `pixel_pdg` is
# the PDG of the particle `pixel_trackid` names, negative ids included (the G4-dropped particle
# itself, from `simchnl`), 0 where there is no truth. `pixel_vertex_dist` is the pixel distance
# in this view to the nearest vertex of `vertex_*`, clipped at `wcfm.data.vertices.DIST_CLIP`.
RICH_PIXEL: dict[str, type] = {
    "pixel_labels2": np.int8,
    "pixel_trackid2": np.int32,
    "pixel_energyfrac2": np.float32,
    "pixel_pdg": np.int32,
    "pixel_vertex_dist": np.float32,
}

# Table -> column -> (dtype, row width; 0 for a 1-D column).
# `mcpart` is the stored particle list of the event: every column of `<event>/mcpart` in the
# trackid_pid_map file, `label` being the 7-class label the labeller gave each particle.
# `simchnl` lists the particles whose charge reached a channel, including the G4-dropped ones
# (negative track id) that `mcpart` does not store. `vertex` is derived, see `wcfm.data.vertices`.
TABLES: dict[str, dict[str, tuple[type, int]]] = {
    "mcpart": {
        "trackid": (np.int32, 0),
        "pid": (np.int32, 0),
        "motherid": (np.int32, 0),
        "mother_pid": (np.int32, 0),
        "proc": (np.int32, 0),
        "endproc": (np.int32, 0),
        "status": (np.int32, 0),
        "ndaughters": (np.int32, 0),
        "ntrajpts": (np.int32, 0),
        "label": (np.int8, 0),
        "mass": (np.float32, 0),
        "start_xyzt": (np.float32, 4),
        "end_xyzt": (np.float32, 4),
        "start_mom": (np.float32, 4),
        "end_mom": (np.float32, 4),
    },
    "simchnl": {
        "trackid": (np.int32, 0),
        "pid": (np.int32, 0),
        "motherid": (np.int32, 0),
        "mother_pid": (np.int32, 0),
        "proc": (np.int32, 0),
        "energy": (np.float32, 0),
    },
    "vertex": {
        "xyz3d": (np.float32, 3),
        "type": (np.int32, 0),
        "parent_tid": (np.int32, 0),
        "face": (np.int32, 0),
        "uvwt": (np.int32, 4),
    },
}

# Every table column under its stored name, and the offsets dataset of each table.
TABLE_COLUMNS: dict[str, tuple[str, type, int]] = {
    f"{t}_{c}": (t, dtype, width)
    for t, cols in TABLES.items()
    for c, (dtype, width) in cols.items()
}
TABLE_OFFSETS: tuple[str, ...] = tuple(f"{t}_offsets" for t in TABLES)


def pixel_keys(pixel: bool, extra: bool, rich: bool) -> dict[str, type]:
    """The per-pixel keys of the requested tiers, each tier implying the ones before it."""
    out: dict[str, type] = {}
    if pixel or extra or rich:
        out.update(PIXEL)
    if extra or rich:
        out.update(EXTRA)
    if rich:
        out.update(RICH_PIXEL)
    return out


def empty_table_column(name: str, n: int = 0) -> np.ndarray:
    _, dtype, width = TABLE_COLUMNS[name]
    return np.zeros((n, width) if width else (n,), dtype=dtype)


def stack_tables(rows: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate per-event table columns into the CSR layout: every column plus one
    `<table>_offsets` per table. Each row is one event's dict holding every `TABLE_COLUMNS` key."""
    out: dict[str, np.ndarray] = {}
    for t, cols in TABLES.items():
        first = f"{t}_{next(iter(cols))}"
        offsets = np.zeros(len(rows) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(r[first]) for r in rows])
        out[f"{t}_offsets"] = offsets
        for c in cols:
            name = f"{t}_{c}"
            parts = [r[name] for r in rows]
            assert all(len(p) == len(r[first]) for p, r in zip(parts, rows, strict=True)), (
                f"{name} length differs from {first} within an event"
            )
            out[name] = (
                np.concatenate(parts, axis=0).astype(TABLE_COLUMNS[name][1])
                if parts
                else empty_table_column(name)
            )
    return out


def event_truth_slice(
    store: Mapping[str, np.ndarray], i: int, pixel: tuple[str, ...], tables: bool
) -> dict[str, np.ndarray]:
    """Event `i`'s per-pixel arrays (`pixel` keys, sliced by `offsets`) and, with `tables`, its
    table columns (sliced by each table's own offsets), out of a CSR store: an open shard, a
    loaded pack or a carry file."""
    out: dict[str, np.ndarray] = {}
    if pixel:
        s, e = int(store["offsets"][i]), int(store["offsets"][i + 1])
        for k in pixel:
            out[k] = store[k][s:e]
    if tables:
        for t, cols in TABLES.items():
            off = store[f"{t}_offsets"]
            s, e = int(off[i]), int(off[i + 1])
            for c in cols:
                out[f"{t}_{c}"] = store[f"{t}_{c}"][s:e]
    return out
