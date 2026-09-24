# `conf/` — the configuration tree

A run is not configured by one file. It is assembled out of interchangeable blocks (data production, 
optimizer, objective, backbone, masker, ...). Each subdirectory here is a slot and each file in it 
is an option you can drop into that slot by name.

For example: `wcfm train model=dino data=prod_jay_100k` picks two of them; every other slot
falls back to the default listed in [`config.yaml`](config.yaml).

## How Hydra works

**Compose.** Hydra merges the blocks you selected into one nested config. A directory is a
group, each file in it an option:

```
conf/model/dino.yaml       # `model=dino`
conf/model/hybrid.yaml     # `model=hybrid`
conf/model/kd.yaml         # `model=kd`
```

`wcfm train model=dino optim.lr=3e-4` means: assemble the tree, put `dino.yaml` in the `model`
slot, then set that one leaf to `3e-4`. The result is a `DictConfig` — plain data. 

**Instantiate.** If a node carries a `_target_` key, `hydra.utils.instantiate` imports that
class and calls it with the sibling keys as keyword arguments — recursively, so a nested config
becomes a nested object graph:

```yaml
teacher:
  _target_: wcfm.model.modules.EmaTeacher
  momentum_start: 0.996
  momentum_end: 1.0
```

becomes `EmaTeacher(momentum_start=0.996, momentum_end=1.0)`. So the whole mechanism is a YAML
merger plus a `_target_` to `__init__` caller.

## The blocks

| Group | Default | Other options | What it is |
|---|---|---|---|
| `model/` | `mae` | `dino`, `hybrid`, `kd`, `polarmae` | the training objective |
| `data/` | `prod_jay_200k_mixed_sharded` | `prod_jay_200k_mixed_packed`, `prod_jay_100k`, `fdhd_2M_mixed_sharded` | which production, and how to read it |
| `optim/` | `adamw_cosine` | — | the optimizer and its schedules |
| `run/` | `default` | — | name, seed, precision, resume, checkpoint cadence |
| `metrics/` | `default` | `minimal`, `full` | which collectors run |
| `launch/` | `single_gpu` | `multi_2gpu`, `multi_6gpu` | devices, strategy, DDP flags |
| `experiment/` | — | `+experiment=<name>` | a whole run pinned as one file |

The `data/` blocks are three productions across three readers:

| Block | Reader | Events | Where |
|---|---|---|---|
| `prod_jay_200k_mixed_sharded` | `sharded` — streams 200 HDF5 shards | 199,870 | `fm/cffm-data/shards_fhdh_sparse_200k_mixed_apa0W` |
| `prod_jay_200k_mixed_packed` | `packed` — one 27.8 GB `.npz` held in RAM | 199,870 | `fm/cffm-data/packed/packed_fhdh_sparse_200k_mixed_apa0W.npz` |
| `prod_jay_100k` | `direct` — reads the production tree | ~100k | `bnayak/cffm-data/prod-jay-100k-truth-2026-06-11` |
| `fdhd_2M_mixed_sharded` | `sharded` — ~500 shards of 4000, event truth only | ~2.0M | `fm/cffm-data/shards_fdhd_sparse_2M_mixed_apa0W` |

The two `200k_mixed` blocks are the *same events*, so a run can change reader without changing
what it trains on. `packed` needs `request_memory` well above `wcfm submit`'s 32 GB default.
`fdhd_2M_mixed_sharded` is the training set: numu and nue productions shuffled together, with
no per-pixel truth, so evaluation stays on the 200k set, whose runs are disjoint from it. A shard
set is built by `wcfm datagen <job> create_shards ...`, which queues
`wcfm.data.prep.create_shards` on a CPU worker; `python -m wcfm.data.prep.create_shards --help`
lists the arguments, and an archived production is first unpacked with
`gridutils/datagen/unpack_apa.sh`.

A `model/` preset is itself a recipe: it selects one option from each sub-group below. You can
swap any of them without touching the preset. `polarmae` is the one preset on a different
module, `pointmae`: it masks tokens of a point cloud rather than pixels, so it selects no
augment and no teacher, and its two terms run only under it.

| Sub-group | Options |
|---|---|
| `model/backbone/` | `attn_mae`, `polarmae` |
| `model/augment/` | `crop_mask`, `mask_only`, `mask_region`, `none` |
| `model/masker/` | `block`, `pixel`, `region` |
| `model/teacher/` | `ema`, `none` |
| `model/term/` | `dino`, `charge`, `occupancy`, `distill`; `chamfer`, `energy` under `polarmae` |
| `model/cropper/` | `default` |
| `model/normalize/` | `log` |

Note: most numbers are not in these files. A block carries only what it changes, which is why
[`model/backbone/attn_mae.yaml`](model/backbone/attn_mae.yaml) is two lines and lists no values:

```yaml
defaults:
  - base_minkunet
```

`base_minkunet` is a typed dataclass registered into Hydra's ConfigStore from Python. 
Every default lives there and nowhere else: model blocks in
[`wcfm/model/config.py`](../wcfm/model/config.py), framework blocks in
[`wcfm/config/schema.py`](../wcfm/config/schema.py). Being typed is also what rejects a typo at
compose time, before a GPU is touched:

```
$ wcfm train model=dino run.name=demo optim.lr_rate=3e-4
ConfigCompositionException : Could not override 'optim.lr_rate'.
```

## Running

You can configure entirely on the command line, entirely in a file, or mix the two.

1. From the command line:
```bash
wcfm train model=dino run.name=demo
wcfm train model=hybrid run.name=demo data=prod_jay_200k_mixed_packed launch=multi_2gpu
wcfm train model=dino run.name=demo optim.lr=3e-4 model.backbone.heads=8
```
`group=option` swaps a block, `some.nested.key=value` sets one leaf. Nothing is mandatory —
every group has a default, so `wcfm train run.name=demo` trains `mae` on the sharded 200k
mixed production on one GPU.

2. You can pin a whole run in a file. Put it in [`experiment/`](experiment/) and select it with
`+experiment=<name>`. Note the `+`: `experiment` is not one of `config.yaml`'s slots:

```bash
wcfm train +experiment=hybrid_baseline_mixed_b100_pefix
```

The file sets its own `run.name` and whatever else it wants. If it re-points a group
`config.yaml` has already filled, it must say `override`:

```yaml
# @package _global_
defaults:
  - override /model: hybrid          # NOT `- /model: hybrid`
  - override /data: prod_jay_100k
  - _self_
run:
  name: my_run
optim:
  epochs: 100
```

3. Command-line overrides win, so a file is a starting point you can vary from:

```bash
wcfm train +experiment=hybrid_baseline_mixed_b100_pefix run.name=my_variant optim.epochs=5
```

### Validation
Whichever form you use, check it before spending a queue slot:

```bash
wcfm train --dry-run model=dino run.name=demo   # compose, validate, construct on CPU, exit
```