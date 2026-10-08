"""``wcfm.data.prep.create_shards``: several production roots, read whole, shuffled in blocks.

The raw trees are synthetic: a few ``*_pixeldata-anode0.h5`` files with the production's group
layout and the matching ``*_metadata.h5``, so nothing here needs ``/gpfs01``. What is pinned is
the contract the 2M training set relies on: every event lands in exactly one shard, the roots
are permuted together, the shards match what ``DirectDataset`` reads event by event, and a run
killed between blocks resumes to the same output.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import h5py
import numpy as np
import pytest

pytest.importorskip("torch")

from wcfm.data.direct import DirectDataset  # noqa: E402
from wcfm.data.prep import create_shards as cs  # noqa: E402

pytestmark = pytest.mark.stack

META_DTYPE = np.dtype(
    [
        ("nu_pdg", "<i4"),
        ("nu_ccnc", "<i4"),
        ("nu_intType", "<i4"),
        ("nu_energy", "<f8"),
        ("nu_vertex_x", "<f8"),
        ("nu_vertex_y", "<f8"),
        ("nu_vertex_z", "<f8"),
    ]
)


MCPART = {  # trackid_pid_map dataset -> per-particle values for tracks 1..3
    "track_ids": [1, 2, 3],
    "pids": [13, 11, 2212],
    "mother_ids": [0, 1, 0],
    "mother_pids": [0, 13, 0],
    "processes": [0, 1, 0],
    "end_processes": [1, 2, 8],
    "statuses": [1, 1, 1],
    "ndaughters": [1, 0, 0],
    "ntrajpts": [9, 4, 6],
    "labels": [1, 3, 1],
    "masses": [0.1057, 0.000511, 0.9383],
}


def add_truth(pix, tmap, g: int, coords: np.ndarray, rng) -> None:
    """Label-pass truth for one event: 1st-contributor frames on a subset of the reco pixels,
    2nd-contributor frames on a subset of those, and the event's particle lists. Track -7 is a
    G4-dropped photon that only `simchnl` lists."""
    first = coords[rng.random(len(coords)) < 0.7]
    second = first[: len(first) // 3]
    tids = rng.choice([1, 2, 3, -7], len(first)).astype(np.float32)
    label_of = {1: 1, 2: 3, 3: 1, -7: 3}
    frames = {
        "frame_trackid_1st": (first, tids),
        "frame_label_1st": (first, np.array([label_of[int(t)] for t in tids], np.float32)),
        "frame_energyfrac_1st": (first, rng.random(len(first)).astype(np.float32)),
        "frame_total_numelectrons": (first, rng.random(len(first)).astype(np.float32) * 1e4),
        "frame_trackid_2nd": (second, np.full(len(second), 3, np.float32)),
        "frame_label_2nd": (second, np.ones(len(second), np.float32)),
        "frame_energyfrac_2nd": (second, rng.random(len(second)).astype(np.float32)),
    }
    for name, (c, f) in frames.items():
        grp = pix.create_group(f"{g}/{name}")
        grp.create_dataset("coords", data=c)
        grp.create_dataset("features", data=f)
    mc = tmap.create_group(f"{g}/mcpart")
    for k, v in MCPART.items():
        mc.create_dataset(k, data=np.asarray(v, np.float32 if k == "masses" else np.int32))
    xyz = rng.uniform(-50, 50, (3, 4)).astype(np.float32)
    for k in ("start_xyzts", "end_xyzts", "start_moms", "end_moms"):
        mc.create_dataset(k, data=xyz + rng.random((3, 4)).astype(np.float32))
    sc = tmap.create_group(f"{g}/simchnl")
    for k, v in {
        "track_ids": [-7, 1],
        "pids": [22, 13],
        "mother_ids": [3, 0],
        "mother_pids": [2212, 0],
        "processes": [15, 0],
    }.items():
        sc.create_dataset(k, data=np.asarray(v, np.int32))
    sc.create_dataset("energies", data=np.array([1.5, 30.0], np.float32))


def write_production(
    root, run: int, n_files: int, groups: int, pdg: int, rng, with_truth: bool = False
) -> list[str]:
    """One raw tree in the production's layout; returns the event keys it should yield."""
    keys = []
    for seq in range(1, n_files + 1):
        stem = f"monte-carlo-{run:06d}-{seq:06d}_1_1_1_20260101T000000Z"
        d = root / f"{run:06d}" / "001" / f"out_{stem}"
        d.mkdir(parents=True)
        with (
            h5py.File(d / f"{stem}_pixeldata-anode0.h5", "w") as pix,
            h5py.File(d / f"{stem}_metadata.h5", "w") as meta,
            h5py.File(d / f"{stem}_trackid_pid_map.h5", "w") as tmap,
        ):
            for g in range(1, groups + 1):
                n = int(rng.integers(20, 60))
                # Channels straddle the V/W boundary so the view filter has something to drop.
                coords = np.stack(
                    [rng.integers(1500, 2650, n), rng.integers(0, 1500, n)], axis=1
                ).astype(np.int32)
                coords = np.unique(coords, axis=0)
                frame = pix.create_group(f"{g}/frame_rebinned_reco")
                frame.create_dataset("coords", data=coords)
                frame.create_dataset("features", data=rng.random(len(coords)).astype(np.float32))
                if with_truth:
                    add_truth(pix, tmap, g, coords, rng)
                row = np.zeros(1, dtype=META_DTYPE)
                row["nu_pdg"], row["nu_ccnc"], row["nu_energy"] = pdg, g % 2, 2.5
                row["nu_vertex_x"] = seq
                meta.create_dataset(f"{g}/metadata", data=row)
                keys.append(f"{stem}_pixeldata-anode0.h5:{g}")
    return keys


@pytest.fixture
def two_roots(tmp_path):
    rng = np.random.default_rng(0)
    numu = write_production(tmp_path / "numu", run=13825, n_files=3, groups=3, pdg=14, rng=rng)
    nue = write_production(tmp_path / "nue", run=16898, n_files=2, groups=3, pdg=12, rng=rng)
    return tmp_path, numu, nue


def _read_back(out):
    """(event_key -> (coords, feats, label, vertex_x)) over every shard, plus images per shard."""
    events, sizes = {}, []
    for shard in sorted(out.glob("shard_*.h5")):
        with h5py.File(shard) as f:
            off = f["offsets"][:]
            keys = [k.decode() for k in f["event_key"][:]]
            sizes.append(len(keys))
            for i, k in enumerate(keys):
                events[k] = (
                    f["coords"][off[i] : off[i + 1]],
                    f["features"][off[i] : off[i + 1]],
                    int(f["labels"][i]),
                    float(f["vertex_xyz"][i, 0]),
                )
    return events, sizes


def test_several_roots_are_one_shuffled_dataset(two_roots):
    tmp_path, numu, nue = two_roots
    out = tmp_path / "shards"
    cs.create_shards(
        datadirs=[str(tmp_path / "numu"), str(tmp_path / "nue")],
        apa=0,
        view="W",
        outdir=str(out),
        shard_size=4,
        seed=42,
        block_files=2,
        threads=2,
        writers=1,
    )

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["n_samples"] == len(numu) + len(nue) == 15
    assert meta["n_shards"] == 4
    assert meta["datadirs"] == [str((tmp_path / d).resolve()) for d in ("numu", "nue")]
    assert not (out / "state.json").exists() and not (out / "carry.npz").exists()

    events, sizes = _read_back(out)
    assert sizes == [4, 4, 4, 3], "exact shards, one short trailing shard"
    assert sorted(events) == sorted(numu + nue), "every event lands in exactly one shard"
    with h5py.File(sorted(out.glob("shard_*.h5"))[0]) as f:
        first = [k.decode() for k in f["event_key"][:]]
        assert "pixel_labels" not in f, "no per-pixel truth was asked for"
    assert any(k in numu for k in first) and any(k in nue for k in first), (
        "the roots are permuted together"
    )
    labels = sorted(e[2] for e in events.values())
    assert labels == [0] * 3 + [1] * 2 + [2] * 10, "numuCC, nueCC and NC from the metadata rows"
    assert all(e[0][:, 0].max() < 1050 for e in events.values()), "W channels rebased to 0..1049"


def test_shards_match_what_direct_dataset_reads(two_roots):
    """The two readers share the parsing functions; this pins that they agree on the pixels,
    the view filter and the truth of every event."""
    tmp_path, numu, nue = two_roots
    out = tmp_path / "shards"
    cs.create_shards(
        datadirs=[str(tmp_path / "numu")],
        apa=0,
        view="W",
        outdir=str(out),
        shard_size=5,
        seed=1,
        block_files=2,
        threads=2,
        writers=1,
    )
    events, _ = _read_back(out)
    ds = DirectDataset(tmp_path / "numu", apa=0, view="W", cache_dir=tmp_path / "cache")
    assert len(ds) == len(numu) == len(events)
    for i in range(len(ds)):
        vox, meta = ds[i]
        coords, feats, label, vx = events[meta["event_key"]]
        np.testing.assert_array_equal(vox.coordinate_tensor.numpy(), coords)
        np.testing.assert_array_equal(vox.feature_tensor.numpy(), feats)
        assert meta["label"] == label
        assert float(meta["vertex_xyz"][0]) == vx


def test_a_run_killed_between_blocks_resumes_to_the_same_shards(two_roots):
    tmp_path, numu, nue = two_roots
    roots = [str(tmp_path / "numu"), str(tmp_path / "nue")]
    kw = dict(apa=0, view="W", shard_size=4, seed=42, block_files=2, threads=2, writers=1)

    cs.create_shards(datadirs=roots, outdir=str(tmp_path / "whole"), **kw)

    # Die on the first shard of the second block: block one's shards, state and carry are on
    # disk, block two's are not.
    real = cs.write_shard
    calls = {"n": 0}

    def flaky(path, arrays):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("killed")
        return real(path, arrays)

    # Writers are processes and a local function does not pickle into one, so the writer
    # pool runs in threads for this test; nothing about the resume depends on the pool kind.
    with (
        mock.patch.object(cs, "ProcessPoolExecutor", ThreadPoolExecutor),
        mock.patch.object(cs, "write_shard", flaky),
        pytest.raises(RuntimeError, match="killed"),
    ):
        cs.create_shards(datadirs=roots, outdir=str(tmp_path / "resumed"), **kw)
    assert (tmp_path / "resumed" / "state.json").exists()
    partial = json.loads((tmp_path / "resumed" / "metadata.json").read_text())
    assert partial["n_samples"] < 15, "the interim metadata counts written events only"

    cs.create_shards(datadirs=roots, outdir=str(tmp_path / "resumed"), **kw)
    whole, sizes_w = _read_back(tmp_path / "whole")
    resumed, sizes_r = _read_back(tmp_path / "resumed")
    assert sizes_w == sizes_r == [4, 4, 4, 3]
    for shard in ("shard_00000.h5", "shard_00003.h5"):
        with (
            h5py.File(tmp_path / "whole" / shard) as a,
            h5py.File(tmp_path / "resumed" / shard) as b,
        ):
            assert list(a["event_key"][:]) == list(b["event_key"][:]), (
                "the resumed run reproduces the interrupted run shard for shard"
            )
    assert not (tmp_path / "resumed" / "state.json").exists()


def test_a_resume_with_different_arguments_is_refused(two_roots):
    tmp_path, numu, nue = two_roots
    roots = [str(tmp_path / "numu"), str(tmp_path / "nue")]
    out = tmp_path / "shards"
    (out).mkdir()
    (out / "state.json").write_text(json.dumps({"signature": {"seed": 7}, "state": {}}))
    with pytest.raises(RuntimeError, match="different arguments"):
        cs.create_shards(datadirs=roots, apa=0, view="W", outdir=str(out), shard_size=4, seed=42)


class _LinearGeometry:
    """Anode 0 spans y, z in [-100, 100] cm; a point projects to channel 1700 + z, tick 600 + x."""

    def apa_bbox(self, anode):
        return np.array([-100.0, 100.0, -100.0, 100.0])

    def project(self, xyz, apa=0):
        return 0, 0, 800, 1700 + int(round(xyz[2])), 600.0 + float(xyz[0])


def test_rich_truth_is_the_same_through_shards_pack_and_direct(tmp_path):
    """The rich tier, carry path included (blocks of two files, shards of four events), reaches
    the shards and the pack exactly as `DirectDataset` parses it: per-pixel arrays and every
    table, event by event."""
    from wcfm.data import direct, truth
    from wcfm.data.packed import PackedDataset
    from wcfm.data.prep.pack_dataset import pack_dataset
    from wcfm.data.sharded import ShardedDataset

    rng = np.random.default_rng(3)
    keys = write_production(
        tmp_path / "prod", 18020, n_files=3, groups=3, pdg=14, rng=rng, with_truth=True
    )
    geom = _LinearGeometry()
    with (
        mock.patch.object(direct, "load_geometry", return_value=geom),
        mock.patch.object(cs, "load_geometry", return_value=geom),
    ):
        cs.create_shards(
            datadirs=[str(tmp_path / "prod")],
            apa=0,
            view="W",
            outdir=str(tmp_path / "shards"),
            shard_size=4,
            seed=0,
            block_files=2,
            threads=2,
            writers=1,
            with_rich_truth=True,
        )
        pack_dataset(
            str(tmp_path / "prod"),
            str(tmp_path / "pack.npz"),
            cache_dir=str(tmp_path / "cache"),
            num_workers=0,
        )
        ds = DirectDataset(
            tmp_path / "prod", apa=0, view="W", cache_dir=tmp_path / "cache", return_rich_truth=True
        )

    meta = json.loads((tmp_path / "shards" / "metadata.json").read_text())
    assert meta["rich_truth"] and meta["n_samples"] == len(keys) == 9
    assert meta["vertex"]["t0_ticks"] < 0

    sharded = ShardedDataset(
        str(tmp_path / "shards"), batch_size=1, shuffle=False, return_rich_truth=True
    )
    from_shards = {
        b.meta["event_key"][0]: {k: v[0] for k, v in b.meta.items() if isinstance(v, list)}
        for b in sharded
    }
    packed = PackedDataset(tmp_path / "pack.npz", return_rich_truth=True)
    from_pack = {m["event_key"]: m for _, m in (packed[i] for i in range(len(packed)))}

    compared = (*truth.pixel_keys(True, True, True), *truth.TABLE_COLUMNS)
    n_tables = 0
    for i in range(len(ds)):
        _, ref = ds[i]
        for other in (from_shards[ref["event_key"]], from_pack[ref["event_key"]]):
            for k in compared:
                np.testing.assert_array_equal(np.asarray(other[k]), ref[k], err_msg=k)
        n_tables += len(ref["mcpart_trackid"])
        assert set(ref["pixel_pdg"][ref["pixel_trackid"] == -7]) <= {22}, (
            "a dropped particle's PDG comes from simchnl"
        )
        assert (ref["pixel_trackid2"] != 0).sum() == (ref["pixel_labels2"] != 0).sum()
        assert ref["vertex_type"][0] == 0, "the interaction vertex comes first"
    assert n_tables == 3 * len(keys), "every event keeps its whole particle list"
