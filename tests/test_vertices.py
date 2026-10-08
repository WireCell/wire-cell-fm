"""``wcfm.data.vertices``: which particle-list configurations make a vertex, and how a vertex
reaches the pixels. Hand-built particle lists and a linear stand-in geometry, so nothing here
needs the wires file."""

from __future__ import annotations

import numpy as np

from wcfm.data import truth
from wcfm.data.vertices import (
    DIST_CLIP,
    VTX_DECAY,
    VTX_INELASTIC,
    VTX_INTERACTION,
    VTX_STOP,
    derive_vertices,
    vertex_truth,
)

MU_MASS, P_MASS = 0.1057, 0.9383


class LinearGeometry:
    """Anode 0 spans y, z in [0, 100] cm; a point projects to channel 1600 + z, tick x."""

    def apa_bbox(self, anode):
        return np.array([0.0, 100.0, 0.0, 100.0])

    def project(self, xyz, apa=0):
        return 0, 0, 800, 1600 + int(round(xyz[2])), float(xyz[0])


def particles(*rows):
    """`mcpart` columns from (tid, pid, mother, proc, endproc, start, end, start_ke, end_ke)."""
    cols = {c: [] for c in truth.TABLES["mcpart"]}
    for tid, pid, mother, proc, endproc, start, end, ke, end_ke in rows:
        mass = {13: MU_MASS, 2212: P_MASS}.get(abs(pid), 0.0)
        cols["trackid"].append(tid)
        cols["pid"].append(pid)
        cols["motherid"].append(mother)
        cols["proc"].append(proc)
        cols["endproc"].append(endproc)
        cols["mass"].append(mass)
        cols["start_xyzt"].append([*start, 0.0])
        cols["end_xyzt"].append([*end, 0.0])
        cols["start_mom"].append([0, 0, 0, ke + mass])
        cols["end_mom"].append([0, 0, 0, end_ke + mass])
    out = {}
    for c, (dtype, width) in truth.TABLES["mcpart"].items():
        vals = cols[c] if cols[c] else [0] * len(rows)
        out[c] = np.asarray(vals, dtype=dtype).reshape((len(rows), width) if width else -1)
    return out


VTX = (20.0, 50.0, 50.0)
MU_END = (40.0, 50.0, 60.0)
# A stopped mu- that decays (endproc 1) to a Michel electron starting at its end.
MUON_DECAY = particles(
    (1, 13, 0, 0, 1, VTX, MU_END, 0.3, 0.0),
    (2, 11, 1, 1, 2, MU_END, (45.0, 50.0, 62.0), 0.03, 0.0),
)


def test_a_decay_needs_two_visible_instances():
    both = derive_vertices(MUON_DECAY, VTX, np.array([1, 1, 2]))
    assert [v["type"] for v in both] == [VTX_INTERACTION, VTX_DECAY]
    assert both[1]["parent"] == 1 and np.allclose(both[1]["xyz"], MU_END)

    # Michel invisible in this view: the muon end is only a stop, which the decay rule
    # does not cover and the stop rule (a stopped, visible muon) does.
    muon_only = derive_vertices(MUON_DECAY, VTX, np.array([1, 1]))
    assert [v["type"] for v in muon_only] == [VTX_INTERACTION, VTX_STOP]


def test_a_stop_at_a_decay_merges_into_the_decay():
    types = [v["type"] for v in derive_vertices(MUON_DECAY, VTX, np.array([1, 2]))]
    assert types.count(VTX_DECAY) == 1 and VTX_STOP not in types


def test_a_soft_inelastic_vertex_is_dropped():
    def scatter(daughter_ke):
        return particles(
            (1, 2212, 0, 0, 19, VTX, (30.0, 50.0, 55.0), 0.2, 0.1),
            (2, 2212, 1, 19, 8, (30.0, 50.0, 55.0), (31.0, 50.0, 56.0), daughter_ke, 0.1),
        )

    hard = derive_vertices(scatter(0.05), VTX, np.array([1, 2]))
    soft = derive_vertices(scatter(0.005), VTX, np.array([1, 2]))
    assert VTX_INELASTIC in [v["type"] for v in hard]
    assert VTX_INELASTIC not in [v["type"] for v in soft]


def test_a_vertex_outside_the_anode_has_no_projection():
    coords = np.array([[50, 20], [60, 40], [0, 0]])
    table, dist = vertex_truth(
        MUON_DECAY, VTX, coords, np.array([1, 1, 2]), geom=LinearGeometry(), apa=0, view="W"
    )
    assert table["vertex_face"].tolist() == [0, 0]
    assert table["vertex_uvwt"][:, 2].tolist() == [1650, 1660]
    np.testing.assert_allclose(dist[:2], [0.0, 0.0])

    outside = particles((1, 13, 0, 0, 8, (20.0, 500.0, 50.0), (40.0, 500.0, 60.0), 0.3, 0.0))
    table, dist = vertex_truth(
        outside, VTX, coords, np.array([1, 0, 0]), geom=LinearGeometry(), apa=0, view="W"
    )
    assert (table["vertex_face"] == -1).all() and (table["vertex_uvwt"] == -1).all()
    assert (dist == DIST_CLIP).all(), "an unprojected vertex is no pixel's nearest vertex"
