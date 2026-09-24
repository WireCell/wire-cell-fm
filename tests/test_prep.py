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


def write_production(root, run: int, n_files: int, groups: int, pdg: int, rng) -> list[str]:
    """One raw tree in the production's layout; returns the event keys it should yield."""
    keys = []
    for seq in range(1, n_files + 1):
        stem = f"monte-carlo-{run:06d}-{seq:06d}_1_1_1_20260101T000000Z"
        d = root / f"{run:06d}" / "001" / f"out_{stem}"
        d.mkdir(parents=True)
        with (
            h5py.File(d / f"{stem}_pixeldata-anode0.h5", "w") as pix,
            h5py.File(d / f"{stem}_metadata.h5", "w") as meta,
        ):
            for g in range(1, groups + 1):
                n = int(rng.integers(20, 60))
                # Channels straddle the V/W boundary so the view filter has something to drop.
                coords = np.stack(
                    [rng.integers(1500, 2650, n), rng.integers(0, 1500, n)], axis=1
                ).astype(np.int32)
                frame = pix.create_group(f"{g}/frame_rebinned_reco")
                frame.create_dataset("coords", data=coords)
                frame.create_dataset("features", data=rng.random(n).astype(np.float32))
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
