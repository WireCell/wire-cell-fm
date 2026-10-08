"""Vertex truth of one event: the 3D vertices its particle list implies, their projection into
the anode, and every reco pixel's distance to the nearest one in this view.

A vertex is a point where distinct particle instances meet:

- the interaction vertex, the median start of the primaries (`vertex_xyz` when there are none);
- a decay, inelastic or capture vertex, where a particle ends by that process and a daughter
  starts within `COINC_CM` of its end, kept when `MIN_VISIBLE` distinct instances leaving
  pixels in this view meet there. An inelastic vertex also needs `INELASTIC_MIN_CHARGED_KE_GEV`
  of kinetic energy in its coincident charged daughters;
- a stop vertex, the end of a muon, pion, kaon or proton longer than `STOP_MIN_LEN_CM` that
  stops inside (end kinetic energy at most `STOP_MAX_END_KE_GEV`) and leaves pixels here;
- a conversion vertex, the end point of a photon of at least `SHOWER_SEED_MIN_GAMMA_KE_GEV`
  further than `SHOWER_SEED_MIN_DISP_CM` from the interaction vertex whose own and daughters'
  pixels here number at least `SHOWER_SEED_MIN_PIX`; such seeds closer than
  `SHOWER_SEED_MERGE_CM` merge, the one with more pixels kept.

Vertices closer than `MERGE_CM` merge into the one with the lower type code. The definitions
and thresholds are those of WC_FM_DINO's `loader/build_vertex_labels.py`, so a pack built here
and one built there agree vertex for vertex. Visibility is per view, so the U, V and W shard
sets of one production can list different vertices for the same event.

A vertex outside the anode's volume (`in_volume`) has no projection: `face` and `uvwt` are -1
and it does not enter `pixel_vertex_dist`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from wcfm.data.wire_geometry import VIEW_CH_RANGES, WireGeometry

VTX_INTERACTION, VTX_DECAY, VTX_INELASTIC, VTX_CONVERSION, VTX_CAPTURE, VTX_STOP = range(6)
VERTEX_TYPE_NAMES = ["interaction", "decay", "inelastic", "conversion", "capture", "stop"]

# G4 end-process code -> vertex type. Continuous losses (eIoni, muIoni, eBrem, hIoni,
# hadElastic) end no particle at a branch point and are absent.
END_PROC_VERTEX_TYPE = {
    1: VTX_DECAY,
    7: VTX_CONVERSION,
    13: VTX_CONVERSION,
    22: VTX_CONVERSION,
    9: VTX_CAPTURE,
    17: VTX_CAPTURE,
    18: VTX_CAPTURE,
    23: VTX_CAPTURE,
    15: VTX_INELASTIC,
    19: VTX_INELASTIC,
    20: VTX_INELASTIC,
    21: VTX_INELASTIC,
}
# Secondary vertex types listed. Conversions along a shower are left out; the displaced
# conversions that seed a shower come in through the shower-seed rule instead.
ALLOWED_TYPES = frozenset({VTX_DECAY, VTX_INELASTIC, VTX_CAPTURE, VTX_STOP})

COINC_CM = 1.0
MERGE_CM = 1.0
MIN_VISIBLE = 2
DIST_CLIP = 200.0
INELASTIC_MIN_CHARGED_KE_GEV = 0.02
STOP_PDGS = frozenset({13, 211, 321, 2212})
STOP_MAX_END_KE_GEV = 0.002
STOP_MIN_LEN_CM = 3.0
SHOWER_SEED_MIN_PIX = 50
SHOWER_SEED_MIN_DISP_CM = 2.0
SHOWER_SEED_MIN_GAMMA_KE_GEV = 0.010
SHOWER_SEED_MERGE_CM = 5.0
# The drift-to-tick offset: a measured constant, not zero and not a fit performed at run time.
#
# The projection is `tick = drift_cm / 0.321126 + t0`; the channel half needs no parameter. t0
# absorbs the frame reference time, WireCell's response-plane offset, any tick cropping, and
# the pixel-bin indexing convention, so it is a property of how the images were made rather
# than a physical constant -- which is why it is named here, overridable, and recorded in every
# result.
#
# Measured, not assumed. Projecting charged-track 3D endpoints (mcpart start/end_xyzts joined
# to their footprint pixels through track_ids) against the pixels' actual tick, over 2116
# measurements in 282 events of prod-jay-100k-truth-2026-06-11, gives -0.649 ticks, 95% CI
# [-0.729, -0.596] (event-clustered bootstrap). That is what rules out the earlier t0 = 0.
#
# Most of the offset is a BINNING convention, not timing: a stored pixel tick is an integer bin
# index whose centre sits at index+0.5, while the projection is continuous, so
# index-minus-continuous earns -0.5 for free. Refitting against bin centres leaves -0.149
# [-0.229, -0.096], so the genuine frame/response-plane term is only ~0.15 tick. The value to
# USE is the index one, because the metric compares projected ticks against integer pixel
# coordinates.
#
# Moving t0 by 0.8 tick -- five times the width of the CI -- shifts only 0.089% of pixels
# across the 20 px radius, so the metric is insensitive to this at the level it is known. The
# reason to get it right is to rule out a multi-tick error, not to chase the third decimal.
#
# Vertex numbers recorded before 2026-08-05 were taken at t0 = 0.
DEFAULT_VERTEX_T0_TICKS = -0.649

MAX_DRIFT_CM = 360.0

CHARGED_PDGS = frozenset(
    {11, 13, 15, 211, 321, 2212, 213, 323, 3222, 3112, 3312, 411, 431, 521, 4122}
)

# The definition, as recorded beside a shard set or pack that carries vertex truth.
PARAMETERS = {
    "allowed_types": sorted(VERTEX_TYPE_NAMES[t] for t in ALLOWED_TYPES),
    "coinc_cm": COINC_CM,
    "merge_cm": MERGE_CM,
    "min_visible": MIN_VISIBLE,
    "dist_clip_px": DIST_CLIP,
    "inelastic_min_charged_ke_gev": INELASTIC_MIN_CHARGED_KE_GEV,
    "stop_max_end_ke_gev": STOP_MAX_END_KE_GEV,
    "stop_min_len_cm": STOP_MIN_LEN_CM,
    "shower_seed_min_pix": SHOWER_SEED_MIN_PIX,
    "shower_seed_min_disp_cm": SHOWER_SEED_MIN_DISP_CM,
    "shower_seed_min_gamma_ke_gev": SHOWER_SEED_MIN_GAMMA_KE_GEV,
    "shower_seed_merge_cm": SHOWER_SEED_MERGE_CM,
}


def in_volume(geom: WireGeometry, xyz_cm, apa: int) -> bool:
    """Whether a 3D point lies within the anode's wire volume (5 cm margin in y and z) and its
    drift range. `WireGeometry.project` returns the nearest wire for any point, so a point
    outside lands on the edge of the image; callers exclude such points rather than use it."""
    ymin, ymax, zmin, zmax = geom.apa_bbox(int(apa))
    return bool(
        ymin - 5 <= xyz_cm[1] <= ymax + 5
        and zmin - 5 <= xyz_cm[2] <= zmax + 5
        and abs(xyz_cm[0]) < MAX_DRIFT_CM
    )


def _merge_close(verts: list[dict], merge_cm: float = MERGE_CM) -> list[dict]:
    """Greedy merge in list order: a vertex within `merge_cm` of a kept one folds into it, and
    the lower type code (with its parent) wins."""
    kept: list[dict] = []
    for v in verts:
        for k in kept:
            if np.linalg.norm(v["xyz"] - k["xyz"]) <= merge_cm:
                if v["type"] < k["type"]:
                    k["type"], k["parent"] = v["type"], v["parent"]
                break
        else:
            kept.append(dict(v))
    return kept


def _base_vertices(gen: Mapping[str, np.ndarray], vertex_xyz, visible: set[int]) -> list[dict]:
    tid, moth, proc, endproc = gen["trackid"], gen["motherid"], gen["proc"], gen["endproc"]
    pid, sx, ex, ke, end_ke = gen["pid"], gen["start_xyz"], gen["end_xyz"], gen["ke"], gen["end_ke"]
    m = len(tid)

    prim = np.where(proc == 0)[0]
    vx = np.median(sx[prim], axis=0) if len(prim) else np.asarray(vertex_xyz, np.float64)
    verts = [{"xyz": vx.astype(np.float64), "type": VTX_INTERACTION, "parent": 0}]
    if not m:
        return verts

    ke_by_tid = {int(t): float(k) for t, k in zip(tid, ke, strict=True)}
    pid_by_tid = {int(t): int(p) for t, p in zip(tid, pid, strict=True)}
    daughters: dict[int, list] = {}
    for i in range(m):
        if int(moth[i]) != 0:
            daughters.setdefault(int(moth[i]), []).append((sx[i], int(tid[i])))

    for i in range(m):
        vtype = END_PROC_VERTEX_TYPE.get(int(endproc[i]))
        if vtype is None or vtype not in ALLOWED_TYPES:
            continue
        ds = daughters.get(int(tid[i]))
        if ds is None:
            continue
        coincident = [dt for dpos, dt in ds if np.linalg.norm(dpos - ex[i]) <= COINC_CM]
        if not coincident:
            continue
        incident = {abs(dt) for dt in coincident if abs(dt) in visible}
        if abs(int(tid[i])) in visible:
            incident.add(abs(int(tid[i])))
        if len(incident) < MIN_VISIBLE:
            continue
        if vtype == VTX_INELASTIC:
            charged_ke = sum(
                ke_by_tid.get(dt, 0.0)
                for dt in coincident
                if abs(pid_by_tid.get(dt, 0)) in CHARGED_PDGS
            )
            if charged_ke < INELASTIC_MIN_CHARGED_KE_GEV:
                continue
        verts.append({"xyz": ex[i].astype(np.float64), "type": vtype, "parent": int(tid[i])})

    # A stop is listed even where a decay, inelastic or capture vertex already sits: the merge
    # below keeps the specific type, so only stops no other vertex covers survive as stops.
    for i in range(m):
        if abs(int(pid[i])) not in STOP_PDGS or float(end_ke[i]) > STOP_MAX_END_KE_GEV:
            continue
        if np.linalg.norm(ex[i] - sx[i]) < STOP_MIN_LEN_CM or abs(int(tid[i])) not in visible:
            continue
        verts.append({"xyz": ex[i].astype(np.float64), "type": VTX_STOP, "parent": int(tid[i])})
    return _merge_close(verts)


def _shower_seeds(gen: Mapping[str, np.ndarray], vertex_xyz, visible_pix: dict[int, int]):
    tid, pid, moth, ex, ke = gen["trackid"], gen["pid"], gen["motherid"], gen["end_xyz"], gen["ke"]
    vx = np.asarray(vertex_xyz, np.float64)
    dau_by_moth: dict[int, list[int]] = {}
    for j in range(len(tid)):
        dau_by_moth.setdefault(int(moth[j]), []).append(j)
    cand: list[tuple[int, dict]] = []
    for i in range(len(tid)):
        if int(pid[i]) != 22 or float(ke[i]) < SHOWER_SEED_MIN_GAMMA_KE_GEV:
            continue
        conv = ex[i].astype(np.float64)
        if np.linalg.norm(conv - vx) <= SHOWER_SEED_MIN_DISP_CM:
            continue
        npx = visible_pix.get(abs(int(tid[i])), 0)
        for j in dau_by_moth.get(int(tid[i]), ()):
            npx += visible_pix.get(abs(int(tid[j])), 0)
        if npx >= SHOWER_SEED_MIN_PIX:
            cand.append((npx, {"xyz": conv, "type": VTX_CONVERSION, "parent": int(tid[i])}))
    cand.sort(key=lambda c: -c[0])
    kept: list[dict] = []
    for _, s in cand:
        if all(np.linalg.norm(s["xyz"] - k["xyz"]) > SHOWER_SEED_MERGE_CM for k in kept):
            kept.append(s)
    return kept


def derive_vertices(
    mcpart: Mapping[str, np.ndarray], vertex_xyz, pixel_trackid: np.ndarray
) -> list[dict]:
    """The 3D vertices of one event, as `{"xyz", "type", "parent"}` dicts, interaction first.

    `mcpart` holds the event's `TABLES["mcpart"]` columns under their bare names
    (`trackid`, `start_xyzt`, ...); `pixel_trackid` is the 1st-contributor track id of this
    view's reco pixels, which decides what is visible.

    The arithmetic stays in the stored float32, so a particle on a threshold falls the same
    side of it as in the WC_FM_DINO packs."""
    gen = {
        "trackid": mcpart["trackid"],
        "pid": mcpart["pid"],
        "motherid": mcpart["motherid"],
        "proc": mcpart["proc"],
        "endproc": mcpart["endproc"],
        "start_xyz": mcpart["start_xyzt"][:, :3],
        "end_xyz": mcpart["end_xyzt"][:, :3],
        "ke": mcpart["start_mom"][:, 3] - mcpart["mass"],
        "end_ke": mcpart["end_mom"][:, 3] - mcpart["mass"],
    }
    ids, counts = np.unique(np.abs(np.asarray(pixel_trackid, np.int64)), return_counts=True)
    visible_pix = {int(t): int(c) for t, c in zip(ids, counts, strict=True) if t != 0}
    verts = _base_vertices(gen, vertex_xyz, set(visible_pix))
    seeds = _shower_seeds(gen, vertex_xyz, visible_pix) if visible_pix else []
    return _merge_close(verts + seeds) if seeds else verts


def vertex_truth(
    mcpart: Mapping[str, np.ndarray],
    vertex_xyz,
    coords: np.ndarray,
    pixel_trackid: np.ndarray,
    *,
    geom: WireGeometry,
    apa: int,
    view: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One event's `vertex_*` table columns and its `pixel_vertex_dist`.

    `coords` are this view's reco pixels, channel rebased to the view (the stored layout), and
    `pixel_trackid` is aligned to them. The tick of a vertex is rounded before the distance is
    taken, as the stored `uvwt` holds it."""
    verts = derive_vertices(mcpart, vertex_xyz, pixel_trackid)
    view = view.upper()
    col = "UVW".index(view)
    ch_start = VIEW_CH_RANGES[view][0]

    faces = np.full(len(verts), -1, np.int32)
    uvwt = np.full((len(verts), 4), -1, np.int32)
    for i, v in enumerate(verts):
        if in_volume(geom, v["xyz"], apa):
            face, u, vv, w, tick = geom.project(v["xyz"], apa=apa)
            faces[i] = face
            uvwt[i] = (u, vv, w, int(round(tick)))

    projected = faces >= 0
    dist = np.full(len(coords), np.inf)
    pts = coords.astype(np.float64)
    for ch, tick in zip(uvwt[projected, col] - ch_start, uvwt[projected, 3], strict=True):
        np.minimum(dist, np.hypot(pts[:, 0] - ch, pts[:, 1] - tick), out=dist)
    table = {
        "vertex_xyz3d": np.array([v["xyz"] for v in verts], np.float32).reshape(-1, 3),
        "vertex_type": np.array([v["type"] for v in verts], np.int32),
        "vertex_parent_tid": np.array([v["parent"] for v in verts], np.int32),
        "vertex_face": faces,
        "vertex_uvwt": uvwt,
    }
    return table, np.minimum(dist, DIST_CLIP).astype(np.float32)
