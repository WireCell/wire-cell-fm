# wire-cell-fm

A training framework for Wire-Cell foundation models (`wcfm`), and a model that runs on it:
self-supervised pretraining on LArTPC detector images, with an offline probe suite.

`README.md` is the overview and `conf/README.md` the config guide. The development tree also
carries `docs/` (a module-by-module map, and the decision records) and `tests/`; neither is
part of a public checkout, so nothing here depends on them being present.

## The one rule everything follows from

Framework packages — `config`, `data`, `engine`, `metrics`, `eval`, `cli` — may not import
`wcfm.model`, including by a relative import that resolves there. The two sides meet at
`wcfm.engine.protocol.TrainingModule` and nowhere else: six methods, plus optional hooks
reached by `getattr`. Where the test tree is present, `tests/test_import_graph.py` walks the
AST of every framework module and fails the build on a violation.

So the engine cannot mention a view, a crop, a mask, a teacher or a term. What the model needs
from the engine is handed to it as a capability on `StepContext`; what the engine needs from
the model is a contract the model implements. Never widen the protocol for a collector's
convenience — add a `getattr` hook.

## Configs

Read `conf/README.md` before changing anything here. The conventions:

- Defaults live only in the typed dataclasses: `wcfm/config/schema.py` for the five framework
  groups (`FRAMEWORK_GROUPS`), `wcfm/model/config.py` for the model groups (`GROUPS`). YAML in
  `conf/` restates a value only to change it, which is why most option files are two lines.
- Registration is per group entry, one option at a time. Never a Union of dataclasses —
  OmegaConf cannot represent one.
- A new configurable class needs three things: a dataclass carrying `_target_`, an entry in
  that module's `GROUPS`, and a `conf/<group>/<name>.yaml` whose `defaults:` selects the base
  node. Add `_convert_: "all"` where nested containers must arrive as plain dicts rather than
  `DictConfig`.
- The model axis arrives through the `wcfm.config_schemas` entry point, read from package
  metadata, so `wire_cell_fm.egg-info/` must sit beside `wcfm/`. Without it the whole `model=`
  axis silently does not exist, and Hydra reports it as `Could not find 'model/term/base_dino'`.
- A preset file carries `# @package _global_`; a sub-group option file does not.
- A term is added with `+model/term@model.terms.X=X` and removed with `~model.terms.X`. A
  preset that only adds one term earns no file.
- `hydra.job.chdir` stays false: the Condor job manages its own working directory.
- Everything a run writes goes under
  `<output_root>/<name>/{checkpoints,debug,probes,features,metrics}`. `wcfm plot` adds `plots/`,
  which holds no data: every figure is a view over the metrics streams and the probe JSONs, it
  rebuilds in seconds, and no job writes or syncs it.

## Training

- The engine owns the backward. `training_step` computes a loss, returns it on `StepOutput`,
  and the `Trainer` differentiates it. One forward, one backward, one optimizer step.
- One DDP path: the whole module is wrapped, and a model reaches its parameters through
  `ctx.module`. Everything trainable runs inside that one `forward` — a head applied outside
  it is marked unused and then receives a gradient, which DDP raises on.
- `Term.compute` is parameter-free; a term's heads are built eagerly in `build()`, never
  lazily on first use.
- Four pytest markers: unmarked needs only the config dependencies, `stack` needs torch
  importable, `needs_data` needs `/gpfs01`, `gpu` and `distributed` need devices. No backbone
  forward is CPU-testable, so `wcfm test --gpu` is required before a stage lands.
- Never import the old model into `wcfm` to compare the two frameworks. Run both and compare
  their evaluations.

## Comments and docstrings

Describe the code as it is. A reader wants to know what runs, and what rules they have to
follow to use it. These rules cover code — `wcfm/`, `gridutils/`, and `tests/` where it is
present. They do not cover Markdown, which is where history, decisions and measurements
belong.

Leave out:

- What git and the Markdown already hold: history ("the old repo", "until 2026-09-10", "no
  longer", "used to"), pointers to where something was settled (ADRs, the plan, numbered
  contracts, document paths), and evidence for past decisions (cluster numbers, spike letters,
  loss curves from other runs).
- Definition by absence ("there is deliberately no `backward` here") and rhetorical contrast
  ("constructed, not named"). State what is there, plainly.
- Restating the header in the body, or the body in the header.

Keep:

- The rule a caller has to follow and what breaks if they do not, especially when it breaks
  silently.
- Mechanism the code does not show: why DDP reduces nothing when a forward skips the wrapper,
  why a lazily created buffer lands on the CPU during `load_state_dict`.
- A warning where an obvious cleanup would be wrong, with enough reason to stop someone acting
  on it.
- The test that pins a behaviour, when there is one.

Style:

- Real names only. Quote signatures, keywords and call sites that exist, and check them before
  writing: an invented example such as `ctx.module(x, tap="backbone")` outlives the comment
  that carried it.
- Single backticks for identifiers. No bold, no RST double backticks. `-` for bullets.
- Plain sentences. Prefer the shorter one.

Placement: one explanation, next to the code it constrains. A rule about a field belongs on
that field; the module header names it in a line and stops. Nothing in `engine/protocol.py`
executes, so the mechanics of wrapping and backward are documented in `engine/trainer.py`,
where they happen.

## Docs

Markdown carries history, decisions and measurements. In the development tree that is
`docs/`, where ADRs are numbered, one decision each, and are never edited in place —
supersede with a new file.

## Cluster jobs

`gridutils/` holds one Condor script per command. None is ever run from the checkout: a submit
packs the tree into `repo.tgz` and stages a copy of the script beside it, so an edit made during
the queue window cannot reach a job that has not started. Each script unpacks the archive into
scratch, puts that on `PYTHONPATH`, and asserts that torch came from the venv it was handed as
`$2` and `wcfm` from the archive rather than the venv's editable install.

| script | runs |
|---|---|
| `train/trainjob.sh` | `wcfm train`, one process per rank under torchrun |
| `eval/evaljob.sh` | `wcfm eval extract`, GPU, one checkpoint |
| `eval/probesjob.sh` | the probe suite, CPU only |
| `eval/mergejob.sh` | the table, CPU, seconds |
| `test/test_gpu_job.sh` | the `gpu` and `distributed` suites |

A run writes to node-local scratch and rsyncs to GPFS every 300 s. On a restart the script
restores the newest checkpoint and the two metrics streams back from GPFS first, because scratch
is empty on every new execution and `run.resume=auto` would otherwise find nothing and start
from epoch 0 in silence.

`wcfm eval submit` builds a DAG, not a loop: `extract -> probes` per checkpoint, all fanning
into one `merge`. Extraction takes a GPU once per checkpoint while every probe stage runs on a
CPU slot, so a queue with one free GPU still makes progress on a campaign. A PRE script skips an
extraction whose `provenance.json` already records that checkpoint's sha256, which is what makes
re-submitting the normal way to extend a campaign rather than a workaround — though DAGMan's own
leftover `eval.dag.*` files have to be cleared first. `merge` cannot fail the DAG: its table is a
view over the probe JSONs and rebuilds in seconds.

Extraction is one process, so it takes the per-rank batch and not the global one: warpconvnet
packs the batch index into 9 bits and refuses more than 512 images in a single forward.

## Tooling

Line length 100, ruff `E,F,I,B,UP`, Python 3.11.

One venv holds everything, and `gridutils/build_env.sh` is the only thing that creates it. Every
pinned version is in the block at the top of that script and nowhere else. `pyproject.toml`
declares the framework's own dependencies and deliberately not the GPU stack, which it must
never resolve. numpy is not pinned: every wheel here declares it unpinned, so the build verifies
that torch did not move rather than dictating a version.

Compiled wheels come from `$WCFM_WHEELHOUSE`, so a rebuild needs no network and cannot pick up a
re-tagged release. warpconvnet is downloaded there on first use; flash-attn has to be there
already, because upstream publishes none for the pinned torch. The staged one was repackaged
from an existing install — `build_flash_attn.sub` compiles a fresh one and has not completed
since the GPU node lost `gcc-c++`.

warpconvnet stays below 1.8. NVIDIA's 1.8 wheels link `libcuda`, so nothing importing
`wcfm.data.voxels` can run where there is no GPU driver, and the CPU suite stops being a
pre-queue gate. 1.7.11 is the ceiling and needs one rename in the backbone.

A job installs nothing. `getenv=False` keeps uv off the worker's PATH and the venv has no pip,
so anything a job imports has to be in the venv before it is queued. `PYTHONPATH` on a worker
carries the unpacked archive and nothing else — a second entry is how a package ends up ahead of
the venv at a version nothing records.
