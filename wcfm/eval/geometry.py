"""Where the true vertex lands in this view, in pixels.

Split out of `probe_vertex` for the same reason `taxonomy` is split out of `probe_pid`: the pool
of pixels the vertex probe scores is drawn at extraction time, and the draw needs the per-pixel
distance in order to know which pixels have one at all. Extraction may not import a probe, so
the computation lives on this side and both callers share it.

It reads truth and geometry only, no features and no weights, which is what makes the pool
pre-drawable.
"""

from __future__ import annotations

import numpy as np

from wcfm.data.vertices import DEFAULT_VERTEX_T0_TICKS, in_volume
from wcfm.data.wire_geometry import WireGeometry

__all__ = ["DEFAULT_VERTEX_T0_TICKS", "vertex_distance"]


def vertex_distance(
    *,
    positions: np.ndarray,
    offsets: np.ndarray,
    vertex_xyz: np.ndarray,
    apa: int,
    view: str,
    t0_ticks: float = DEFAULT_VERTEX_T0_TICKS,
):
    """Per-pixel distance to the projected true vertex, and which pixels have one.

    Projects each event's `vertex_xyz` into this view's (channel, tick) and measures
    pixel-space distance. Events whose vertex falls outside the wire volume are excluded
    rather than clamped**, so a mis-projected vertex drops out instead of piling every pixel of
    its event at one edge of the image.

    Takes plain arrays rather than a feature object, so extraction can call it with the eval
    set's truth before any features exist.

    Returns `(dist, valid, info)`: `dist` is NaN where there is no projection and `valid` is
    `isfinite(dist)`.
    """

    positions = np.asarray(positions)
    offsets = np.asarray(offsets)
    vertex_xyz = np.asarray(vertex_xyz)
    n_pixels = int(offsets[-1])
    n_events = len(offsets) - 1

    geom = WireGeometry.load(t0_ticks=t0_ticks)

    dist = np.full(n_pixels, np.nan, dtype=np.float32)
    n_ok = n_outside = 0
    for ev in range(n_events):
        a, b = int(offsets[ev]), int(offsets[ev + 1])
        if b == a:
            continue
        xyz = vertex_xyz[ev]
        if not in_volume(geom, xyz, int(apa)):
            n_outside += 1
            continue
        _, u, v, w, tick = geom.project(xyz, apa=int(apa))
        ch = float(geom.channel_for_view(str(view), np.array([u, v, w])))
        pos = positions[a:b]
        dist[a:b] = np.hypot(
            pos[:, 0].astype(np.float64) - ch, pos[:, 1].astype(np.float64) - tick
        ).astype(np.float32)
        n_ok += 1

    return (
        dist,
        np.isfinite(dist),
        {"n_events_projected": n_ok, "n_events_vertex_outside_volume": n_outside},
    )
