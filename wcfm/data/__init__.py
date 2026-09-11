"""This directory contains everything related to the data loading.
Every reader yields `Batch(voxels, meta)` from interchangeable backends.

 Three dataset classes are support:
 - direct: reads the original H5DF files individually
 - packed: reads the entire dataset from a single packed file
 - sharded: reads the entire dataset from shard files
 """
