# wire-cell-fm

A training framework for Wire-Cell foundation model (`wcfm`), and a model that runs on
it. Self-supervised pretraining on LArTPC detector images, with an offline probe suite to
evaluate what the learned representation is worth.

```bash
wcfm train run.name=my_run            # train the default objective on one GPU
wcfm submit model=hybrid run.name=x   # the same run, queued on the cluster
wcfm eval submit runs/my_run          # score its checkpoints
```

## Installation

One script builds the GPU stack and the framework's own dependencies.
Every dependency and version is pinned at the top of that script.
Set `WCFM_PYENV` to specify the location of the python environemnt.

```bash
git clone <this repo> && cd wire-cell-fm
gridutils/build_env.sh                      # ~10 min; writes to $WCFM_PYENV
source $WCFM_PYENV/bin/activate
wcfm env-check                              # check the staack
```

The script relies on pre-compiled wheels for both `torch+warpconvnet` and `flash-attn`. These are available from one shared directory, `$WCFM_WHEELHOUSE`, which defaults to `/gpfs01/lbne/users/fm/shared/wheels`. 

Since `flash-attn` wheels are not always easily available, a build script is also provided:

```bash
condor_submit gridutils/build_flash_attn.sub   # once per torch pin, by one person
```

## The one structural idea

The framework does not know what a model is doing. Framework packages — `config`, `data`,
`engine`, `metrics`, `eval`, `cli` — may not import `wcfm.model`. 
The two sides meet at a single contract, `wcfm.engine.protocol.TrainingModule`: six
methods, plus optional hooks reached by `getattr`.

Everything else follows from that rule. The engine cannot mention a view, a crop, a mask, a
teacher or a term, so what the model needs from the engine is handed to it as a capability
(`StepContext`), and what the engine needs from the model is a contract the model implements.
The model is never imported by name either, but it is constructed from a `_target_` string
in the config.

## The packages

| Package | What it is |
|---|---|
| `wcfm/config/` | Typed dataclasses that give Hydra its defaults, the ConfigStore registration, and where a run's files go. The dataclasses are never instantiated — Hydra merges them into a `DictConfig` and they are gone |
| `wcfm/data/` | `Batch(voxels, meta)` from three interchangeable readers: `direct` (the production tree), `packed` (one `.npz` in RAM), `sharded` (streamed HDF5 shards) |
| `wcfm/engine/` | The `TrainingModule` contract, the training loop on Lightning Fabric, optimizer and schedules, checkpoints, preemption |
| `wcfm/metrics/` | Collectors with a declared cadence, append-only JSONL, and no collectives inside `compute()`, so a data-dependent branch cannot hang a job |
| `wcfm/model/` | Backbone, augment stage, loss terms, and the `SslModule` that implements `TrainingModule` |
| `wcfm/eval/` | Offline: extract features per checkpoint, pool, then the probe suite as a Condor DAG |
| `wcfm/cli/` | Eight commands. Composition and submission live here, and nothing else is an entry point |

Supporting trees: `conf/` is the Hydra config tree, `gridutils/` holds the Condor job scripts.

## How a run executes

`Trainer` is constructed in exactly one place. The whole chain, top to bottom:

```
pyproject.toml:29      wcfm = "wcfm.cli.__main__:main"          console script
  cli/__main__.py:32   import_module(COMMANDS[name]).main(rest) dispatch on argv[0]
    cli/train.py       register_all(); hydra.compose(...)       -> cfg
      cli/train.py:148   run(cfg)
        train.py:124       instantiate(cfg.model)               the module is built here
        train.py:162       Trainer(cfg, module, argv).fit()     the only Trainer(...) call
          trainer.py:347     for epoch in ...
          trainer.py:391       for batch_idx, batch in ...
          trainer.py:467         module.training_step(batch, ctx)
          trainer.py:492         fabric.backward(out.loss)
```

`instantiate(cfg.model)` is where the science enters: it reads `_target_:
wcfm.model.modules.SslModule` out of the composed config and builds the backbone, the augment
stage and the loss terms under it. No framework file names any of those classes.

Inside the step, `SslModule.training_step` normalizes in place, builds a view plan, runs the
optional EMA teacher under `no_grad`, takes one forward over every scored view — the
backbone and every active term's head in a single call, so DDP arms its reducer once — sums the
weighted term losses, and returns the total. The engine differentiates it. One forward, one
backward, one optimizer step.

On the cluster the top of the chain is `wcfm submit`, which writes a `.sub` and queues
`gridutils/train/trainjob.sh`; that script unpacks the run's `repo.tgz` on the worker and execs
`python -m wcfm.cli` — or `torchrun --nproc_per_node=N -m wcfm.cli` for multi-GPU.
Same entry point, one process per rank.

## Commands

`wcfm <command>`; every one takes `--help`.

| Command | What it does |
|---|---|
| `env-check` | What stack is this, versions of the four pinned GPU packages and the framework's own, plus the detected CUDA arch |
| `train` | Compose a config, build the module, hand both to the engine. `--dry-run` composes, validates and constructs on CPU, then exits |
| `submit` | Compose a run's config on the login node, then queue it. `--smoke` for a reduced-scale run, `--dry-run` to write the `.sub` and stop |
| `sweep` | Hydra enumerates, Condor launches. The full sweep syntax works: `a,b`, `range(1,4)`, `choice(a,b)`, `glob(*)` |
| `metrics` | Read the stream back out: `wcfm metrics summary <run_dir>` |
| `eval` | The offline pipeline: `extract`, `probe`, `merge`, `compare`, and `submit` for the whole DAG |
| `plot` | Figures over the metrics streams and the probe JSONs: `wcfm plot <run_dir>`, several run directories to overlay them |
| `diff` | What two runs actually differ by — configs, and with `--code` the source trees they executed |
| `test` | The suites a CPU cannot run: `--dist-cpu` runs the distributed suite locally on 2 CPU ranks, `--gpu` submits the GPU suites to Condor |
| `datagen` | Queue a dataset builder on a CPU worker: `wcfm datagen <job> create_shards ...` or `pack_dataset ...`, arguments passed through |

```bash
wcfm train model=hybrid run.name=demo optim.lr=3e-4   # train here
wcfm submit model=dino run.name=demo --smoke          # a short queued run
wcfm sweep model=dino,hybrid run.seed=range(1,4) --id seeds
wcfm eval submit runs/demo                            # extract -> probes -> merge
wcfm plot runs/demo                                   # diagnostics + probes -> runs/demo/plots
wcfm plot runs/a runs/b --out-dir=cmp                 # the same panels, one colour per run
wcfm diff runs/a runs/b --code
wcfm datagen shards_2M create_shards --datadir A B --apa 0 --outdir S --shard_size 4000
```

## Configuring a run

A run is not configured by one file. It is assembled out of interchangeable blocks — objective,
data production, optimizer, launch, metrics — one per slot:

```bash
wcfm train run.name=demo                                    # every slot takes its default
wcfm train model=dino data=prod_jay_100k run.name=demo      # swap two blocks
wcfm train model=hybrid run.name=demo optim.lr=3e-4         # set one leaf
wcfm train +experiment=my_run                               # a whole run pinned in a file
```

Three objectives ship as presets: `mae` (the default), `dino` and `hybrid`. They are the same
module with different terms and teacher settings, not different code paths.

Defaults do not live in the YAML. They live in typed dataclasses registered into Hydra's
ConfigStore — framework blocks in [`wcfm/config/schema.py`](wcfm/config/schema.py), model
blocks in [`wcfm/model/config.py`](wcfm/model/config.py) — which is also what rejects a typo at
compose time, before a GPU is touched.

**[`conf/README.md`](conf/README.md) is the full guide**: every group and its options, how
compose and `_target_` instantiation work, the data productions and their readers, and how to
pin a run in `conf/experiment/`.

Validate before spending a queue slot:

```bash
wcfm train --dry-run model=dino run.name=demo
```
