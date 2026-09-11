"""The data layer: the primitive, collation, the per-rank split, and the two loader shapes.

Anything touching torch or warpconvnet is marked ``stack``; anything reading the production is
marked ``needs_data``. Neither needs a GPU -- ``/gpfs01`` is mounted on the login node and
sparse readers are pure IO -- but both need more than a bare CPU venv, and a suite that errors
instead of skipping on a laptop teaches people to ignore it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Declared AND applied: `-m stack` has to select these, or the marker is a claim the README
# makes and the suite does not honour. importorskip stays, so a bare run skips rather than errors.
pytestmark = pytest.mark.stack

torch = pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

from wcfm.data.build import per_rank_batch_size  # noqa: E402
from wcfm.data.collate import collate, collate_meta  # noqa: E402
from wcfm.data.voxels import Batch, offsets_from_counts, voxels_from  # noqa: E402

SHARD_DIR = Path("/gpfs01/lbne/users/fm/cffm-data/shards_prod-jay-2026-06-11_mixed_apa0W")


def _voxels(n: int, c: int = 1):
    return voxels_from(
        torch.arange(2 * n, dtype=torch.int32).reshape(n, 2),
        torch.randn(n, c),
        torch.tensor([0, n], dtype=torch.int64),
    )


# ---------------------------------------------------------------- the primitive


def test_offsets_from_counts_is_a_csr_row_pointer():
    assert offsets_from_counts([3, 2, 0, 4]).tolist() == [0, 3, 5, 5, 9]
    assert offsets_from_counts([]).tolist() == [0]


def test_voxels_from_builds_a_batched_voxels():
    v = voxels_from(
        torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int32),
        torch.zeros(3, 1),
        torch.tensor([0, 2, 3], dtype=torch.int64),
    )
    assert v.coordinate_tensor.shape == (3, 2)
    assert v.offsets.tolist() == [0, 2, 3]


def test_voxels_from_rejects_a_row_count_mismatch():
    """The failure the twelve hand-written sites could each make independently."""
    with pytest.raises(ValueError, match="disagree on row count"):
        voxels_from(torch.zeros(3, 2, dtype=torch.int32), torch.zeros(2, 1),
                    torch.tensor([0, 3], dtype=torch.int64))


def test_voxels_from_rejects_offsets_that_do_not_span_the_rows():
    with pytest.raises(ValueError, match="offsets must run"):
        voxels_from(torch.zeros(3, 2, dtype=torch.int32), torch.zeros(3, 1),
                    torch.tensor([0, 2], dtype=torch.int64))


# ---------------------------------------------------------------- Batch


def test_batch_unpacks_as_a_pair_so_ported_trainer_code_keeps_working():
    """`train_dino.py:739` is `xs, _ = batch`; the port must not have to rewrite that."""
    b = Batch(_voxels(4), {"label": torch.tensor([1])})
    voxels, meta = b
    assert voxels is b.voxels and meta["label"].tolist() == [1]


def test_batch_size_counts_samples_not_rows():
    v = voxels_from(torch.zeros(7, 2, dtype=torch.int32), torch.zeros(7, 1),
                    torch.tensor([0, 3, 7], dtype=torch.int64))
    assert Batch(v).batch_size == 2


def test_batch_to_moves_tensors_and_leaves_strings_alone():
    b = Batch(_voxels(3), {"label": torch.tensor([2]), "event_key": ["a.h5:1"]})
    moved = b.to("cpu")
    assert moved.meta["event_key"] == ["a.h5:1"]
    assert isinstance(moved.meta["label"], torch.Tensor)


# ---------------------------------------------------------------- collation


def test_collate_concatenates_pixels_and_stacks_truth():
    items = [(_voxels(3), {"label": 0, "nu_energy": 1.5, "event_key": "a:1"}),
             (_voxels(5), {"label": 2, "nu_energy": 2.5, "event_key": "b:2"})]
    batch = collate(items)
    assert batch.batch_size == 2
    assert batch.voxels.coordinate_tensor.shape[0] == 8
    assert batch.voxels.offsets.tolist() == [0, 3, 8]
    assert batch.meta["label"].tolist() == [0, 2]
    assert batch.meta["event_key"] == ["a:1", "b:2"]


def test_collate_meta_omits_tiers_the_source_never_carried():
    """Legacy reco shards carry no truth at all; a consumer checks for a key, not a flag."""
    assert collate_meta([{}, {}]) == {}
    out = collate_meta([{"label": 1}, {"label": 0}])
    assert set(out) == {"label"}


def test_collate_meta_keeps_per_pixel_tiers_per_sample():
    """Per-pixel truth is CSR-aligned, so it stays split by sample rather than concatenated."""
    import numpy as np
    metas = [{"pixel_labels": np.zeros(3)}, {"pixel_labels": np.zeros(5)}]
    out = collate_meta(metas)
    assert [a.shape[0] for a in out["pixel_labels"]] == [3, 5]


# ---------------------------------------------------------------- the per-rank split


def test_per_rank_batch_size_divides_the_global_batch():
    assert per_rank_batch_size(100, 1) == 100
    assert per_rank_batch_size(100, 2) == 50


def test_per_rank_batch_size_refuses_an_inexact_split():
    """A floor would silently change the effective batch -- the surprise this exists to remove."""
    with pytest.raises(ValueError, match="not divisible"):
        per_rank_batch_size(100, 3)


class _Dummy:
    """A map-style dataset that is just a list, for exercising the subsetting arithmetic."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def test_subset_caps_a_map_style_dataset():
    """Only the sharded reader takes n_subset itself; the other two must be wrapped here.

    The old trainer did this at train_dino.py:445-448 and again at :509-512. Missing it would
    mean n_subset=400 capping the shard path and silently reading the whole production on the
    other two -- exactly the divergence build.py exists to prevent.
    """
    from wcfm.data.build import subset

    assert len(subset(_Dummy(1000), 400, seed=42)) == 400


def test_subset_is_a_no_op_when_it_would_not_shrink_anything():
    from wcfm.data.build import subset

    d = _Dummy(100)
    assert subset(d, -1, seed=42) is d
    assert subset(d, 0, seed=42) is d
    assert subset(d, 500, seed=42) is d


def test_subset_is_seeded_so_two_runs_pick_the_same_samples():
    from wcfm.data.build import subset

    a = subset(_Dummy(1000), 50, seed=7)
    b = subset(_Dummy(1000), 50, seed=7)
    c = subset(_Dummy(1000), 50, seed=8)
    assert a.indices == b.indices
    assert a.indices != c.indices


def test_build_loader_rejects_an_unknown_backend():
    from omegaconf import OmegaConf

    from wcfm.data.build import build_loader

    cfg = OmegaConf.create({"backend": "synthetic", "global_batch_size": 8})
    with pytest.raises(ValueError, match="unknown data.backend"):
        build_loader(cfg)


# ---------------------------------------------------------------- against the production


@pytest.mark.needs_data
def test_sharded_loader_yields_batches_with_truth():
    from omegaconf import OmegaConf

    from wcfm.data.build import build_loader

    cfg = OmegaConf.create({
        "backend": "sharded", "sharded_dir": str(SHARD_DIR), "buffer_size": 200,
        "n_subset": 400, "global_batch_size": 8, "datadir": "", "packed_path": "",
        "cache_dir": "./data", "apa": 0, "view": "W",
    })
    loader = build_loader(cfg, world_size=1, num_workers=0)
    batch = next(iter(loader))

    assert isinstance(batch, Batch)
    assert batch.batch_size == 8
    assert batch.voxels.coordinate_tensor.shape[1] == 2
    assert batch.voxels.feature_tensor.shape[0] == batch.voxels.coordinate_tensor.shape[0]
    # These shards carry event truth; the three classes are numuCC / nueCC / NC.
    assert batch.meta["label"].shape == (8,)
    assert set(batch.meta["label"].tolist()) <= {-1, 0, 1, 2}
    assert len(batch.meta["event_key"]) == 8


@pytest.mark.needs_data
def test_shuffle_false_reaches_the_sharded_reader():
    """Diagnostics need an identical event sequence on every pass.

    The sharded path owns its own shard ordering and never sees a sampler, so `shuffle` has to
    reach the reader's constructor. Passing it only to DistributedSampler would leave
    build_loader(shuffle=False) still shuffling here, and silently.
    """
    from omegaconf import OmegaConf

    from wcfm.data.build import build_loader

    cfg = OmegaConf.create({
        "backend": "sharded", "sharded_dir": str(SHARD_DIR), "buffer_size": 200,
        "n_subset": 2000, "global_batch_size": 8, "datadir": "", "packed_path": "",
        "cache_dir": "./data", "apa": 0, "view": "W",
    })
    keys = [
        next(iter(build_loader(cfg, shuffle=False, num_workers=0))).meta["event_key"]
        for _ in range(2)
    ]
    assert keys[0] == keys[1]

    shuffled = next(iter(build_loader(cfg, shuffle=True, num_workers=0))).meta["event_key"]
    assert shuffled != keys[0]


@pytest.mark.needs_data
def test_n_subset_reaches_the_sharded_reader():
    from omegaconf import OmegaConf

    from wcfm.data.build import build_loader

    def n_batches(n_subset):
        cfg = OmegaConf.create({
            "backend": "sharded", "sharded_dir": str(SHARD_DIR), "buffer_size": 200,
            "n_subset": n_subset, "global_batch_size": 8, "datadir": "", "packed_path": "",
            "cache_dir": "./data", "apa": 0, "view": "W",
        })
        return len(build_loader(cfg, shuffle=False, num_workers=0).dataset)

    assert n_batches(1000) < n_batches(3000)


@pytest.mark.needs_data
def test_sharded_batch_size_is_per_rank():
    """world_size=2 halves what each rank yields; the global batch stays what config said."""
    from omegaconf import OmegaConf

    from wcfm.data.build import build_loader

    # Shards hold 1000 samples each (metadata.json), and the reader refuses to start when the
    # full-shard count cannot feed world_size x num_workers readers -- so two ranks need at
    # least two full shards before the per-rank question can even be asked.
    base = {
        "backend": "sharded", "sharded_dir": str(SHARD_DIR), "buffer_size": 200,
        "n_subset": 4000, "global_batch_size": 8, "datadir": "", "packed_path": "",
        "cache_dir": "./data", "apa": 0, "view": "W",
    }
    one = next(iter(build_loader(OmegaConf.create(base), world_size=1, num_workers=0)))
    two = next(iter(build_loader(OmegaConf.create(base), rank=0, world_size=2, num_workers=0)))
    assert one.batch_size == 8
    assert two.batch_size == 4
