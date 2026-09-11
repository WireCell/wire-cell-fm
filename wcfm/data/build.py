"""Turn a `DataConfig` into a `DataLoader` that yields `Batch`.

The engine just sees a loader that yields `Batch` and nothing else.
However, the three backends have different shapes:
- `sharded` is an `IterableDataset` that assembles its own batches and is driven with
 `DataLoader(batch_size=None)`.
- `direct` and `packed` are map-style and are batched by the DataLoader through `collate`. 

On top of this, in case of DPP, batch size is global, and need to be divided per rank 
The two shapes consume the result differently: map-style passes it to the DataLoader 
as `batch_size`, while the sharded reader takes it in its constructor and yields 
pre-batched items.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

from wcfm.config.io import per_rank_batch_size
from wcfm.data.collate import collate

BACKENDS = ("direct", "sharded", "packed")


# Re-exported: `wcfm.config.io` owns `per_rank_batch_size` because it is
# torch-free and the `config.yaml` a run records has to agree with the loader that read it.
__all__ = ["BACKENDS", "build_dataset", "build_loader", "per_rank_batch_size", "subset"]


def subset(dataset, n_subset: int, seed: int):
    """Cap a map-style dataset at `n_subset` samples, drawn by a seeded permutation.

    Only the sharded reader takes `n_subset` itself, so the map-style readers are capped by
    wrapping them here. Skipping this would mean `n_subset=400` caps the shard path and
    silently reads the whole production on the other two.
    """
    if n_subset is None or n_subset <= 0 or n_subset >= len(dataset):
        return dataset
    rng = torch.Generator().manual_seed(seed)
    return Subset(dataset, torch.randperm(len(dataset), generator=rng)[:n_subset].tolist())


def build_dataset(
    cfg: Any,
    *,
    per_rank: int,
    rank: int,
    world_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
):
    """Construct the reader named by `cfg.backend`.

    `per_rank` and `shuffle` reach the sharded reader here, because it batches and shuffles
    internally; the map-style readers get neither and are batched and shuffled by the DataLoader.

    `cfg.splits` is deliberately not consulted: held-out data arrives with the next production,
    so every split today is `train_frac: 1.0` and wiring it now would be wiring a no-op.
    """
    backend = cfg.backend
    pixel = bool(getattr(cfg, "return_pixel_truth", False))
    extra = bool(getattr(cfg, "return_extra_truth", False))

    if backend == "direct":
        from wcfm.data.direct import DirectDataset

        return subset(
            DirectDataset(
                datadir=cfg.datadir,
                apa=cfg.apa,
                view=cfg.view,
                cache_dir=cfg.cache_dir,
                return_pixel_truth=pixel,
                return_extra_truth=extra,
            ),
            cfg.n_subset,
            seed,
        )
    if backend == "packed":
        from wcfm.data.packed import PackedDataset

        return subset(
            PackedDataset(
                cfg.packed_path, return_pixel_truth=pixel, return_extra_truth=extra
            ),
            cfg.n_subset,
            seed,
        )
    if backend == "sharded":
        from wcfm.data.sharded import ShardedDataset

        return ShardedDataset(
            root_dir=cfg.sharded_dir,
            batch_size=per_rank,
            buffer_size=cfg.buffer_size,
            shuffle=shuffle,
            n_subset=cfg.n_subset,
            rank=rank,
            world_size=world_size,
            num_workers=num_workers,
            seed=seed,
            return_pixel_truth=pixel,
            return_extra_truth=extra,
        )
    raise ValueError(f"unknown data.backend {backend!r}; expected one of {BACKENDS}")


def build_loader(
    cfg: Any,
    *,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: int = 42,
    drop_last: bool = True,
) -> DataLoader:
    """The one entry point. Yields `Batch` regardless of which backend produced it.

    `shuffle=False` means a deterministic event sequence on every pass -- what extraction and
    the run-to-run comparisons need. It has to reach the sharded reader's constructor, since
    that path owns its own shard ordering and never sees a sampler.
    """
    per_rank = per_rank_batch_size(cfg.global_batch_size, world_size)
    dataset = build_dataset(
        cfg,
        per_rank=per_rank,
        rank=rank,
        world_size=world_size,
        num_workers=num_workers,
        shuffle=shuffle,
        seed=seed,
    )

    if cfg.backend == "sharded":
        # Already batched by the reader, which also owns the shard-level shuffle and the
        # world_size x num_workers partition. Auto-batching off; no sampler.
        return DataLoader(dataset, batch_size=None, num_workers=num_workers)

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        if world_size > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=per_rank,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        # Training drops a short tail to match the sharded reader; extraction passes False,
        # because there the tail is events that would otherwise be scored by one batch size
        # and not another.
        drop_last=drop_last,
    )
