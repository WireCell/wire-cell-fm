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
- `launch.find_unused_parameters` is on because a head that runs on only some views of a step
  leaves its bucket unreduced without it. A model whose forward uses every parameter every
  step turns it off in its own preset, as `conf/model/polarmae.yaml` does, and turns it back
  on the day it grows a head that runs only on some steps.
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

## Objectives and their presets

Four objectives are nominal. Each has a final recipe in `conf/experiment/`, and a new run starts
from one of these, not from a `model=` preset alone:

- `polarmae`, PoLAr-MAE trained on its own: a point-cloud MAE on `pointmae` (chamfer and energy
  terms over hidden token groups; no crop, mask or teacher). Short run `polarmae_short`
  (2 ranks x 8, lr 7e-5, 6 epochs, bf16). Long run `polarmae_long` (4 ranks x 16, lr 3.5e-5,
  42 epochs): it degrades from step ~60,000, so score `checkpoint_epoch15.pt` and earlier.
- `kd`, a PoLAr-MAE checkpoint distilled into the MinkUNet `attn_mae` backbone: `kd_polarmae`
  (whole image in, cosine to the frozen teacher's per-voxel features, 32-true). The teacher is a
  wcfm checkpoint (`model.terms.distill.checkpoint`), never a foreign package.
- `dino`, per-pixel DINO against an EMA teacher, removed pixels absent: `dino_ctrl` and its three
  augmentation arms `dino_fullteacher`, `dino_multicrop`, `dino_croponly`.
- `hybrid`, DINO whose masked pixels are reinjected as `masked` tokens and scored by the same
  cross-entropy (`score_injected: true`); no occupancy term: `hybrid_ddp6_eb600`.

`mae` (charge plus occupancy) stays `config.yaml`'s default and is not nominal.

Missing: no experiment yet tests charge plus occupancy where the model has to find the true
pixels inside a large wiped region. `mae` asks it only of a candidate list the masker enumerates
up front (`model.augment.masker.build_candidates`, `neg_per_pos`). The generative grow path,
where coordinates are grown from the bottleneck into the hole and the prediction set is the
candidate set, is not in wcfm. Adding it and giving it a preset is the open fifth objective.

A preset's `run.name` is the run it produces, and the file name need not match it:
`polarmae_short` writes `wcfm_polarmae_200k_b80k`, `polarmae_long` writes
`wcfm_polarmae_200k_10M_lr35`.
Compose a preset against its run's `config.yaml` before editing it; only `data.cache_dir`,
`run.output_root` and, for the two PoLAr-MAE runs, the unread `run.save_every_minutes` differ.
`hybrid_ddp6_eb600` has no run of its own yet.

## Data

- Training reads `conf/data/fdhd_2M_mixed_200k.yaml`, the first 200,000 events of the 2M mixed
  production, through `conf/config.yaml`'s `data:` default. A preset says `override /data:` only
  to leave it, and `fdhd_2M_mixed_sharded` is the whole set for the preset that wants it. No
  preset restates `n_subset`; the subset is the data option's.
- Evaluation reads `prod_jay_200k_mixed_sharded`, the production with per-pixel truth, through
  `wcfm eval`'s `--data` default: 10,000 of its events per eval set (`--max-images`), one
  `--eval-set-root` shared across the runs being compared. Its runs are disjoint from the 2M
  production, so every probe table is out of sample. A run is never scored on what it trained on.
- The shared eval set is `/gpfs01/lbne/users/fm/mvicenzi/CONDOR_OUT/wcfm_hybrid_ddp6_eb600/features/eval_set`
  (`n10000-3dee7078a2ba`). Extraction reads the eval production at the run's per-rank batch, and
  the reader drops short final batches, so the events it yields depend on the batch: pass
  `wcfm eval submit ... --batch-size=8`, which reads exactly the set's events. At 16 the pass
  misses 16 of them and every extract fails with "the eval set was built from different events".
- CAVEAT: the per-pixel truth of `prod_jay_200k_mixed_sharded`, and so of the shared eval set,
  predates the Michel labelling fix: the decay electron of a stopping mu- is labelled `Blip`,
  not `Michel` (about a third of all Michels, and every Michel of a primary CC mu-). Until the
  truth is regenerated, any Michel or Blip number, and per-pixel semantic scores that pool over
  them, are provisional. Regenerating means rebuilding the shards, a new eval set, and
  rescoring every run against it. The 2M training shards carry event truth only and are
  unaffected.
- `wcfm submit` under an existing `run.name` resumes that run, so a preset whose data changed
  keeps its name only if the old run directory has moved.

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
- A run rsyncs scratch to GPFS every 300 s, once more from its exit trap, and on restart
  restores the newest checkpoint and the two metrics streams first. Resubmitting under an
  existing `run.name` therefore resumes. To start over: `condor_rm`, wait until the job has
  left `condor_q` and its `.log` says "Job was aborted", then delete the run directory, then
  submit. A directory deleted while the job is still exiting is recreated by that last sync,
  and the new job resumes the old run in silence.
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
