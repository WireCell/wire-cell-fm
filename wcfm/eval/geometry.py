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

__all__ = ["DEFAULT_VERTEX_T0_TICKS", "vertex_distance"]

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
    from wcfm.data.wire_geometry import WireGeometry  # pure numpy, no warpconvnet

    positions = np.asarray(positions)
    offsets = np.asarray(offsets)
    vertex_xyz = np.asarray(vertex_xyz)
    n_pixels = int(offsets[-1])
    n_events = len(offsets) - 1

    geom = WireGeometry.load(t0_ticks=t0_ticks)
    ymin, ymax, zmin, zmax = geom.apa_bbox(int(apa))

    dist = np.full(n_pixels, np.nan, dtype=np.float32)
    n_ok = n_outside = 0
    for ev in range(n_events):
        a, b = int(offsets[ev]), int(offsets[ev + 1])
        if b == a:
            continue
        xyz = vertex_xyz[ev]
        if not (
            ymin - 5 <= xyz[1] <= ymax + 5
            and zmin - 5 <= xyz[2] <= zmax + 5
            and abs(xyz[0]) < 360.0
        ):
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
