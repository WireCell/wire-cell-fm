# Working in wire-cell-fm

`wcfm` is a training framework for Wire-Cell foundation models and the models that run on it.
`README.md` is the overview and `conf/README.md` the config guide; read them first. This file is
how to work here: the rules the code follows, the order to do things in, and what to check
before anything lands. `tests/` and `docs/` exist in the development tree only, so nothing may
depend on them being present.

## The structure to preserve

Framework packages — `config`, `data`, `engine`, `metrics`, `eval`, `cli` — never import
`wcfm.model`, not even through a relative import that resolves there. The two sides meet at
`wcfm.engine.protocol.TrainingModule` and nowhere else: six methods, plus optional hooks the
engine reaches by `getattr`. `tests/test_import_graph.py` walks the AST of every framework
module and fails on a violation.

Consequences to hold in mind when changing either side:

- The engine cannot know a view, a crop, a mask, a teacher or a term. What a model needs from
  the engine is a capability on `StepContext`; what the engine needs from a model is a method
  on the protocol. Never widen the protocol for a collector's convenience — add a `getattr` hook.
- The engine owns the backward. `training_step` returns a loss on `StepOutput`; the `Trainer`
  differentiates it. One forward, one backward, one optimizer step.
- One DDP path: the whole module is wrapped and a model reaches it as `ctx.module`. Everything
  trainable runs inside that one `forward`; a head applied outside it is marked unused, then
  receives a gradient, and DDP raises.
- `Term.compute` is parameter-free; a term's heads are built eagerly in `build()`.
- `run.precision` is an autocast and nothing more: `build_fabric` installs
  `AutocastOnlyPrecision` so a forward receives its arguments in the dtype the caller passed.
  Geometry that has to stay exact under autocast still disables it locally, as
  `wcfm.model.backbones.polarmae.ops.sq_dists` does.
- Never import the old model into `wcfm` to compare frameworks. Run both, compare evaluations.

## Configs

- Defaults live only in the typed dataclasses: `wcfm/config/schema.py` (`FRAMEWORK_GROUPS`) and
  `wcfm/model/config.py` (`GROUPS`). A YAML in `conf/` restates a value only to change it.
- Registration is per group entry, one option at a time. Never a Union of dataclasses.
- A new configurable class needs three things: a dataclass carrying `_target_`, an entry in
  that module's `GROUPS`, and `conf/<group>/<name>.yaml` whose `defaults:` selects the base
  node. Add `_convert_: "all"` where nested containers must arrive as plain dicts.
- A preset carries `# @package _global_` and says `override` for any group `config.yaml` has
  already filled; a sub-group option file carries neither.
- A term is added with `+model/term@model.terms.X=X` and removed with `~model.terms.X`. A
  preset that only adds one term earns no file.
- The model axis reaches Hydra through the `wcfm.config_schemas` entry point, read from
  `wire_cell_fm.egg-info/` beside `wcfm/`. Without it `model=` silently does not exist and Hydra
  reports `Could not find 'model/term/base_dino'`.
- `hydra.job.chdir` stays false. A run writes under
  `<output_root>/<name>/{checkpoints,debug,probes,features,metrics}`; `wcfm plot` adds `plots/`,
  a view over the streams that no job writes or syncs.
- Check a composition before spending a queue slot: `wcfm train --dry-run <overrides>`.

## Changing code

- One implementation per behaviour. When two functions, two tests or two config paths do the
  same thing, fold them; a second copy is where the two drift apart.
- When a change replaces something, remove what it replaced in the same change: the old code
  path, its flag, its config option, its test. Nothing stays as a fallback "in case"; git
  keeps it. Code kept only because a test still pins it needs the test to pin a behaviour
  someone uses, not the code.
- A change should leave the file smaller or clearer than it found it. If it grows the file,
  the growth is the new behaviour and nothing else.
- Prefer the shorter way when it is as clear: a `getattr` hook over a new protocol method, a
  plain function over a class with one method, a dataclass field over a parallel dict.

## Before anything lands

In this order, every time:

1. `ruff check` and `ruff format --check` — line length 100, rules `E,F,I,B,UP`, Python 3.11.
2. The CPU suite: `pytest tests -m "not gpu and not distributed"`. Unmarked tests need only
   the config dependencies, `stack` needs torch, `needs_data` needs `/gpfs01` (mounted on the
   login node, so it runs there too).
3. `wcfm test --gpu` — submits the `gpu` and `distributed` suites to Condor and leaves a report
   at `<output>/test_gpu_report.txt`. No backbone forward is CPU-testable, and the distributed
   suite is the only place DDP's reducer runs over the real wrapping, so this is required.

Leave changes in the working tree; the user reviews diffs and commits.

## Cluster jobs

- One venv, created only by `gridutils/build_env.sh`; every pin is in the block at its top.
  Point the commands at it with `WCFM_PYENV=<venv>`; the default is
  `/gpfs01/lbne/users/fm/<user>/uvenv`. `pyproject.toml` declares the framework's dependencies and
  never the GPU stack. warpconvnet stays below 1.8 (1.8 links `libcuda`, and the CPU suite stops
  being a pre-queue gate).
- A job installs nothing and imports only from the venv plus the unpacked archive. Nothing runs
  from the checkout: `wcfm submit` packs the tree into `repo.tgz` and stages a copy of the
  script, so an edit during the queue window cannot reach a job.
- Resources are environment variables on the submit: `WCFM_REQUEST_MEMORY` (MB),
  `WCFM_REQUEST_CPUS`, `WCFM_OUTPUT_BASE`. The login shell is tcsh; set them inline,
  `env WCFM_PYENV=... wcfm submit ...`.
- A run rsyncs scratch to GPFS every 300 s and, on restart, restores the newest checkpoint and
  the two metrics streams first. Resubmitting under an existing `run.name` therefore resumes;
  delete the run directory or pick a new name to start over.
- Extraction is one process and takes the per-rank batch: warpconvnet refuses more than 512
  images in one forward. `wcfm eval submit` builds a DAG (`extract -> probes` per checkpoint,
  one `merge`); clear DAGMan's `eval.dag.*` files before resubmitting to extend a campaign.
- After a submit, read the first synced `metrics/step.jsonl` before trusting a run: the scalars
  a model reports (`n_tokens`, `n_points`, step time as `global_batch / samples_per_s`) are
  where a wrong input shows up while the loss still falls.

## Comments and docstrings

Describe the code as it is: what runs, and the rule a caller has to follow. These rules cover
code in `wcfm/`, `gridutils/`, `tests/`; Markdown is where history, decisions and measurements
belong.

Leave out history ("no longer", "used to", "the old repo"), pointers to where something was
settled, evidence for past decisions, definition by absence ("there is deliberately no
`backward` here"), rhetorical contrast, and restating the header in the body.

Keep the rule a caller must follow and what breaks silently if they do not; mechanism the code
does not show; a warning where an obvious cleanup would be wrong; the test that pins a
behaviour.

Real names only — quote signatures and call sites that exist, and check them. Single backticks
for identifiers, `-` for bullets, plain short sentences. One explanation, next to the code it
constrains: a rule about a field goes on that field; nothing in `engine/protocol.py` executes,
so wrapping and backward are documented in `engine/trainer.py`.

## Docs

Markdown carries history, decisions and measurements. In the development tree that is `docs/`:
ADRs are numbered, one decision each, never edited in place — supersede with a new file.
