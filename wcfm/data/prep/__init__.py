"""Dataset builders: turn the raw production HDF5 tree into the packed and sharded forms.

These are one-shot tools, not part of a training run. Both parse events with the functions of
`wcfm.data.direct`, so the anode selection, the view filter and the channel rebasing are the
same operations the training loop performs -- a pack or a shard set is the production as
`DirectDataset` sees it, materialised. `pack_dataset` reads through `DirectDataset` itself;
`create_shards` reads every source file once, whole, because that is what a 200k-file tree on
GPFS allows in an afternoon.

They sit here rather than under `wcfm/cli/` because each writer is one half of a contract
with a reader in the package above: `create_shards` writes exactly the layout
`ShardedDataset` reads, and `pack_dataset` writes the arrays `PackedDataset` expects.

    python -m wcfm.data.prep.create_shards --help
    python -m wcfm.data.prep.pack_dataset  --help
"""
