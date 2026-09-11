#!/usr/bin/env python3
"""Exact 3D->(channel, tick) projection for the DUNE FD-HD 1x2x6 detector,
replicating what WireCell-Toolkit's DepoFluxWriter does at
simulation time, but offline in pure numpy so it can run on the host (no LArSoft
container) against the same authoritative wire geometry.

The per-pixel truth pack carries 3D mcpart vertices (start/end_xyz, cm), 
but a per-pixel 2D vertex target needs those points in (channel, tick). 

The simulation performs the projection inside WCT 
(`DepoFluxWriter.cxx`: `Pimpos::distance()` for wire/channel, drift->time for tick),
driven by the wires file `dune10kt-1x2x6-wires-larsoft-v1.json.bz2`. That file is a
plain bz2 JSON (schema Store -> anodes(12), faces(24=12 APA x 2), planes, wires,
points) and is readable on the host, so we replicate the projection here.

- DUNE FD-HD induction wires WRAP around the APA (one readout channel = several 
  angled wire segments at different y,z), so a global linear channel(y,z) map 
  is wrong for U/V. Instead we store every wire segment's (y,z) endpoints + its channel 
  and, for a query point, return the channel of the nearest segment. 
  Collection (W) wires are vertical (channel monotonic in z) and fall out of the same machinery.

- APA0 tiles y in [-600,0] cm, z in [0,230] cm; its two faces are the two drift volumes 
  (face0 x>0, face1 x<0). A point's (y,z) picks the APA tile and x-sign picks 
  the face (= which drift side, hence which collection
  channel block 1600-2079 vs 2080-2559).

- Raw tick 0.5us x rebin 4 = 2.0us image tick; drift speed 1.60563 mm/us => 0.321126 cm per image tick.
  tick = drift_cm / 0.321126 + t0. The slope is geometry;
  the constant t0 comes from response-plane + G4 reference-time offset.
  t0 is a caller-supplied parameter here and is NOT set by this module,
  which is pure geometry; the default 0.0 means "no offset applied".

Channel numbering matches the pack's view convention (U 0-799, V 800-1599,
W 1600-2559 within a 2560-channel APA.

Usage:
    geom = WireGeometry.load()                 # parses + caches the wires json
    face, u, v, w, tick = geom.project(xyz_cm, apa=0)   # single point
    faces, uvw, ticks   = geom.project_many(xyz_cm_arr, apa=0)  # (M,3) -> arrays
"""

from __future__ import annotations

import bz2
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Authoritative wires file (bz2 JSON) -- located on cvmfs; overridable.
DEFAULT_WIRES_JSON = (
    "/cvmfs/dune.opensciencegrid.org/products/dune/dunetpc/v09_09_00/"
    "wire-cell-cfg/dune10kt-1x2x6-wires-larsoft-v1.json.bz2"
)
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "wire_geometry_dune10kt_1x2x6.npz"

MM_TO_CM = 0.1

# Drift / tick constants (WCT dune10kt-1x2x6).
DRIFT_SPEED_MM_PER_US = 1.60563
RAW_TICK_US = 0.5
REBIN = 4
IMAGE_TICK_US = RAW_TICK_US * REBIN                       # 2.0 us
CM_PER_TICK = DRIFT_SPEED_MM_PER_US * IMAGE_TICK_US * MM_TO_CM   # 0.321126 cm/tick

# Per-view channel ranges within one 2560-channel APA (pack convention).
VIEW_CH_RANGES = {"U": (0, 800), "V": (800, 1600), "W": (1600, 2560)}
CHANNELS_PER_APA = 2560   # global wire channel of anode a is based at a*CHANNELS_PER_APA
# plane-index-within-face (0,1,2) -> view name
_PLANE_VIEW = {0: "U", 1: "V", 2: "W"}


class WireGeometry:
    """Holds per (anode, face, plane) wire-segment tables and projects 3D->2D.

    Internally, for each (anode, face, plane) it stores:
        seg_a  (S,2)  segment endpoint A in (y,z) cm
        seg_b  (S,2)  segment endpoint B in (y,z) cm
        seg_ch (S,)   readout channel of each segment
    plus per (anode, face) the collection-plane x (cm) used as the drift reference,
    and per anode the (y,z) bounding box for APA assignment.
    """

    def __init__(self, tables: Dict, t0_ticks: float = 0.0):
        self._t = tables
        self.t0_ticks = float(t0_ticks)

    # ------------------------------------------------------------------ build
    @classmethod
    def load(cls, wires_json: str = DEFAULT_WIRES_JSON,
             cache_path: Optional[Path] = DEFAULT_CACHE,
             t0_ticks: float = 0.0, rebuild: bool = False) -> "WireGeometry":
        cache_path = Path(cache_path) if cache_path is not None else None
        if cache_path is not None and cache_path.exists() and not rebuild:
            return cls(_load_cache(cache_path), t0_ticks=t0_ticks)
        tables = _parse_wires_json(wires_json)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _save_cache(cache_path, tables)
        return cls(tables, t0_ticks=t0_ticks)

    # ------------------------------------------------------------------ query
    def _face_of(self, anode: int, x_cm: float) -> int:
        """Face 0 (x>0 drift volume) or 1 (x<0), by the sign of x relative to the
        anode collection-plane x. Uses the stored per-face collection-plane x."""
        # face k collection-plane x sign tells the drift side; pick the face whose
        # collection x has the same sign as the point's x.
        x0 = self._t[(anode, 0, 2)]["plane_x"]   # face0 W-plane x (cm)
        return 0 if (x_cm * x0) >= 0 else 1

    def _nearest_channel(self, anode: int, face: int, plane: int,
                         y: float, z: float) -> int:
        t = self._t[(anode, face, plane)]
        a = t["seg_a"]; b = t["seg_b"]; ch = t["seg_ch"]
        d = _point_seg_dist2(np.array([y, z], np.float64), a, b)
        return int(ch[int(np.argmin(d))])

    def tick_of(self, anode: int, face: int, x_cm: float) -> float:
        """Image tick from drift x. drift distance = |x - x_collection_plane|."""
        xw = self._t[(anode, face, 2)]["plane_x"]
        drift_cm = abs(float(x_cm) - xw)
        return drift_cm / CM_PER_TICK + self.t0_ticks

    def project(self, xyz_cm, apa: int = 0) -> Tuple[int, int, int, int, float]:
        """Project one 3D point (cm) -> (face, u_ch, v_ch, w_ch, tick).

        Channels are returned in the pack's APA-LOCAL 2560-scheme (U 0-799,
        V 800-1599, W 1600-2559), matching a single anode's pixeldata frame.
        The wires-file channels are GLOBAL (anode a is based at a*CHANNELS_PER_APA:
        anode2->5120, anode4->10240), so we subtract that base. For apa=0 the base
        is 0 (a no-op), which is why apa=0 callers -- the 10k packs and
        build_vertex_labels -- were always correct. tick is the image (rebinned) tick.
        """
        x, y, z = float(xyz_cm[0]), float(xyz_cm[1]), float(xyz_cm[2])
        face = self._face_of(apa, x)
        base = apa * CHANNELS_PER_APA
        u = self._nearest_channel(apa, face, 0, y, z) - base
        v = self._nearest_channel(apa, face, 1, y, z) - base
        w = self._nearest_channel(apa, face, 2, y, z) - base
        tick = self.tick_of(apa, face, x)
        return face, u, v, w, tick

    def project_many(self, xyz_cm, apa: int = 0):
        """Vectorized over points. xyz_cm (M,3) -> (faces (M,), uvw (M,3), ticks (M,))."""
        xyz = np.asarray(xyz_cm, np.float64).reshape(-1, 3)
        M = len(xyz)
        faces = np.zeros(M, np.int32)
        uvw = np.zeros((M, 3), np.int32)
        ticks = np.zeros(M, np.float64)
        for i in range(M):
            f, u, v, w, tk = self.project(xyz[i], apa=apa)
            faces[i] = f; uvw[i] = (u, v, w); ticks[i] = tk
        return faces, uvw, ticks

    def channel_for_view(self, view: str, uvw: np.ndarray) -> np.ndarray:
        """Pick the view's channel column and rebase to view-local (0-based),
        matching the pack coords (channel -= ch_start)."""
        col = {"U": 0, "V": 1, "W": 2}[view.upper()]
        ch_start = VIEW_CH_RANGES[view.upper()][0]
        return uvw[..., col] - ch_start

    def apa_bbox(self, anode: int):
        return self._t[("bbox", anode)]


# ---------------------------------------------------------------- json parsing

def _parse_wires_json(path: str) -> Dict:
    S = json.loads(bz2.open(path).read())["Store"]
    anodes = [a["Anode"] for a in S["anodes"]]
    faces = [f["Face"] for f in S["faces"]]
    planes = [p["Plane"] for p in S["planes"]]
    wires = [w["Wire"] for w in S["wires"]]
    pts = np.array([[q["Point"]["x"], q["Point"]["y"], q["Point"]["z"]]
                    for q in S["points"]], np.float64) * MM_TO_CM   # -> cm

    tables: Dict = {}
    for ai, anode in enumerate(anodes):
        ymin = zmin = np.inf
        ymax = zmax = -np.inf
        for fi_local, fi in enumerate(anode["faces"]):
            f = faces[fi]
            for pi_local, pi in enumerate(f["planes"]):
                pl = planes[pi]
                ws = [wires[i] for i in pl["wires"]]
                seg_ch = np.array([w["channel"] for w in ws], np.int32)
                a = pts[[w["tail"] for w in ws]][:, 1:3]   # (S,2) = (y,z) cm
                b = pts[[w["head"] for w in ws]][:, 1:3]
                plane_x = float(np.median(pts[[w["tail"] for w in ws]][:, 0]))
                tables[(ai, fi_local, pi_local)] = {
                    "seg_a": a, "seg_b": b, "seg_ch": seg_ch,
                    "plane_x": np.float64(plane_x),
                    "view": _PLANE_VIEW[pi_local],
                }
                mids = 0.5 * (a + b)
                ymin = min(ymin, mids[:, 0].min()); ymax = max(ymax, mids[:, 0].max())
                zmin = min(zmin, mids[:, 1].min()); zmax = max(zmax, mids[:, 1].max())
        tables[("bbox", ai)] = np.array([ymin, ymax, zmin, zmax], np.float64)
    return tables


def _save_cache(path: Path, tables: Dict) -> None:
    flat = {}
    for k, v in tables.items():
        if k[0] == "bbox":
            flat[f"bbox|{k[1]}"] = v
        else:
            ai, fi, pi = k
            flat[f"seg|{ai}|{fi}|{pi}|a"] = v["seg_a"]
            flat[f"seg|{ai}|{fi}|{pi}|b"] = v["seg_b"]
            flat[f"seg|{ai}|{fi}|{pi}|ch"] = v["seg_ch"]
            flat[f"seg|{ai}|{fi}|{pi}|x"] = np.float64(v["plane_x"])
    np.savez_compressed(path, **flat)


def _load_cache(path: Path) -> Dict:
    z = np.load(path, allow_pickle=False)
    tables: Dict = {}
    for key in z.files:
        parts = key.split("|")
        if parts[0] == "bbox":
            tables[("bbox", int(parts[1]))] = z[key]
        elif parts[0] == "seg":
            ai, fi, pi, field = int(parts[1]), int(parts[2]), int(parts[3]), parts[4]
            t = tables.setdefault((ai, fi, pi), {"view": _PLANE_VIEW[int(parts[3])]})
            if field == "a": t["seg_a"] = z[key]
            elif field == "b": t["seg_b"] = z[key]
            elif field == "ch": t["seg_ch"] = z[key]
            elif field == "x": t["plane_x"] = np.float64(z[key])
    return tables


# ---------------------------------------------------------------- geometry math

def _point_seg_dist2(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared distance from point p (2,) to each segment [a_i, b_i] ((S,2) each)."""
    ab = b - a                                   # (S,2)
    ap = p[None, :] - a                          # (S,2)
    denom = np.einsum("ij,ij->i", ab, ab)        # (S,)
    t = np.where(denom > 0, np.einsum("ij,ij->i", ap, ab) / np.maximum(denom, 1e-12), 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab                   # (S,2)
    d = p[None, :] - proj
    return np.einsum("ij,ij->i", d, d)


if __name__ == "__main__":
    # Smoke: build geometry, print APA0 layout, project the ν vertex of a known event.
    g = WireGeometry.load(rebuild=True)
    print("APA0 bbox (y,z cm):", g.apa_bbox(0))
    for f in (0, 1):
        print(f"  face{f} plane_x: U={g._t[(0,f,0)]['plane_x']:.3f} "
              f"V={g._t[(0,f,1)]['plane_x']:.3f} W={g._t[(0,f,2)]['plane_x']:.3f} cm")
    vtx = np.array([-180.35, -311.23, 41.25])   # numu ev0 ν vertex (cm)
    print("project ν vertex", vtx, "->", g.project(vtx, apa=0))
    print("CM_PER_TICK =", CM_PER_TICK)
