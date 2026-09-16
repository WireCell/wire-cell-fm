"""`wcfm submit`: compose a run's config on the login node, then queue it.

This is Python rather than a shell script because a run's config does not exist as a file until
the run writes one: Hydra composes `conf/` plus the overrides on this command line, and the run
name, the GPU count and the batch-size check all come from composing that config here, before
anything is queued. The worker-side half stays shell, in `gridutils/train/trainjob.sh`.

The payoff is that a run is validated before it is queued, by the same code path
`wcfm train --dry-run` uses: an empty `run.name`, a `global_batch_size` that does not divide by
the device count, a model that does not satisfy `TrainingModule`. Each of those otherwise costs
a queue slot and is discovered from a log.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from wcfm.cli.jobpack import (
    ARCHIVE_NAME,
    check_repo,
    git_environment,
    pack_repo,
    request_disk,
    stage_executable,
)

__all__ = ["main"]

USAGE = """usage: wcfm submit [options] <hydra overrides ...>

  wcfm submit model=dino_small run.name=my_run
  wcfm submit model=dino_small run.name=my_run launch=multi_2gpu
  wcfm submit model=dino_small run.name=my_run --smoke
  wcfm submit model=dino_small run.name=my_run --dry-run

options:
  --smoke            reduced-scale run: appends _smoke to run.name and overrides
                     optim.epochs=${SMOKE_EPOCHS:-2}, data.n_subset=${SMOKE_NSUBSET:-2000},
                     run.save_every=1
  --dry-run          compose, validate, build the .sub, submit nothing
  --config-dir DIR   a conf/ tree other than the repo's
  --repo DIR         checkout to package for the job (default: this package's repository)

`run.name` is required: every C9 path is <output_root>/<name>/..., so an unnamed run would
write into <output_root>/ and the next one would overwrite it.

The GPU count is NOT a flag. `request_gpus` is read from `launch.devices` in the resolved
config, so `launch=multi_2gpu` (or `launch.devices=2`) is the one place it is said. There used
to be a `--gpus N` that set Condor's allocation while `launch.devices` set Fabric's world, and
nothing reconciled them: `--gpus 2` with the default `launch=single_gpu` was accepted and
produced two ranks each building a one-device Fabric.
"""

DEFAULT_REQUIREMENTS = '(GPUs_DeviceName == "NVIDIA L40S") && (GPUs_Capability == 8.9)'


def _pop_opt(args: list[str], name: str, default: str | None = None) -> str | None:
    for i, arg in enumerate(args):
        if arg == name:
            value = args[i + 1] if i + 1 < len(args) else default
            del args[i : i + 2]
            return value
        if arg.startswith(name + "="):
            del args[i]
            return arg.split("=", 1)[1]
    return default


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    args = list(argv)
    smoke = "--smoke" in args
    if smoke:
        args.remove("--smoke")
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")

    # `--gpus` was retired on 2026-09-10. Caught by name rather than left to Hydra, which
    # rejects it as an override with a lexer error that names nothing useful.
    if any(a == "--gpus" or a.startswith("--gpus=") for a in args):
        print(
            "wcfm submit: --gpus was removed. The GPU count now comes from `launch.devices` "
            "in the config, so it is stated once instead of twice:\n"
            "    launch=single_gpu | launch=multi_2gpu | launch=multi_6gpu\n"
            "    launch.devices=N        (for a count with no preset)\n"
            "One count sets both Condor's request_gpus and Fabric's world, so the two cannot "
            "disagree.",
            file=sys.stderr,
        )
        return 2

    from wcfm.cli.train import _conf_dir

    repo = Path(_pop_opt(args, "--repo", "") or "").expanduser()
    config_dir, overrides = _conf_dir(args)
    repo = repo.resolve() if str(repo) else config_dir.parent

    if smoke:
        # `data.n_subset` and `optim.epochs` are plain overrides now, so this is a few entries
        # rather than the JSON surgery the old --smoke had to do.
        overrides = [o for o in overrides if not o.startswith(("optim.epochs=", "run.save_every="))]
        overrides += [
            f"optim.epochs={os.environ.get('SMOKE_EPOCHS', '2')}",
            f"data.n_subset={os.environ.get('SMOKE_NSUBSET', '2000')}",
            "run.save_every=1",
        ]

    import hydra
    from hydra.errors import HydraException
    from omegaconf import OmegaConf

    from wcfm.cli.train import build_module, explain_missing_model, validate
    from wcfm.config.store import register_all

    register_all()
    if not config_dir.is_dir():
        print(f"wcfm submit: no config directory at {config_dir}", file=sys.stderr)
        return 2

    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        try:
            cfg = hydra.compose(config_name="config", overrides=list(overrides))
        except HydraException as exc:
            print(f"wcfm submit: {exc}", file=sys.stderr)
            explain_missing_model(exc, config_dir)
            return 2
        OmegaConf.resolve(cfg)

        if smoke:
            cfg.run.name = f"{cfg.run.name}_smoke"

        # The same checks `--dry-run` runs, for the same reason: each one otherwise costs a
        # queue slot and is found in a log rather than in a terminal.
        try:
            validate(cfg)
        except (ValueError, AssertionError) as exc:
            print(f"wcfm submit: {exc}", file=sys.stderr)
            return 2
        # No divisibility check here: `validate(cfg)` above runs
        # `per_rank_batch_size(global_batch_size, launch.devices)` and refuses the same case
        # with the same authority. A second check against a count from somewhere other than the
        # config would be asking a different question from the one the job asks.
        gpus = int(cfg.launch.devices)
        try:
            module = build_module(cfg)
        except Exception as exc:  # noqa: BLE001 - any construction failure is the finding
            print(f"wcfm submit: the model does not build: {exc}", file=sys.stderr)
            return 2

        run_name = str(cfg.run.name)
        n_params = sum(p.numel() for p in module.parameters())
        epochs = int(cfg.optim.epochs)

    # Send the name back as an override. The worker recomposes the config from `overrides`,
    # so any name this process computed rather than read -- `--smoke`'s `_smoke` suffix, above
    # all -- does not exist there: the engine wrote `runs/<original>/` while trainjob.sh
    # rsynced `runs/<original>_smoke/`, a directory the script's own `mkdir -p` had just
    # created empty. Cluster 2263 therefore reported "Training complete!", exited 0, and
    # discarded every checkpoint, metric and history it had produced. Hydra takes the last
    # value, so appending pins it whatever the source.
    overrides = [o for o in overrides if not o.startswith("run.name=")]
    overrides.append(f"run.name={run_name}")

    user = os.environ.get("USER", "unknown")
    pyenv = Path(os.environ.get("WCFM_PYENV", f"/gpfs01/lbne/users/fm/{user}/uvenv"))
    cache = Path(os.environ.get("WCFM_CACHE_DIR", f"/gpfs01/lbne/users/fm/{user}/cache"))
    output_base = Path(
        os.environ.get("WCFM_OUTPUT_BASE", f"/gpfs01/lbne/users/fm/{user}/CONDOR_OUT")
    )

    job_script = repo / "gridutils" / "train" / "trainjob.sh"
    if not job_script.is_file():
        print(f"wcfm submit: no job script at {job_script}", file=sys.stderr)
        return 2
    # The job installs nothing, so the venv has to carry everything before it is queued. hydra
    # and omegaconf are runtime dependencies, not test ones: the config is composed on the
    # worker, so a venv without them dies on `import hydra` before it trains anything.
    if not (pyenv / "bin" / "python").is_file():
        print(
            f"wcfm submit: no venv at {pyenv}. Build it once:\n"
            f"  gridutils/build_env.sh",
            file=sys.stderr,
        )
        return 2

    out_dir = output_base / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"run:     {run_name}  ({epochs} epochs, {n_params:,} parameters)")
    print(f"repo:    {repo}")
    print(f"gpus:    {gpus}  (from launch.devices)")
    print(f"output:  {out_dir}")

    # Quoted individually so an override containing a space or a bracket survives Condor's
    # argument parsing, which splits on whitespace outside quotes.
    #
    # `initialdir` does not decide what the job imports: with file transfer on, the worker
    # always starts in its own scratch directory, and `trainjob.sh` unpacks the archive into
    # `scratch/repo/` so that the cwd itself holds nothing importable. What it decides is
    # submit-side: which directory relative paths in this `.sub` resolve against, and
    # where Condor would return output files to. Both are stated absolutely below, so this is
    # the run's address rather than a load-bearing setting; it is kept because `condor_q -af
    # Iwd` is how a running job is traced back to its run directory.
    quoted = " ".join(f"'{o}'" for o in overrides)

    # One archive of the tree per run, transferred to the worker. Condor reads
    # `transfer_input_files` when the job STARTS, not when it is submitted, so an archive
    # written once here is what pins the run -- pointing the transfer at the live checkout
    # would leave the whole queue window open to an edit. `job/` is inside the run's own
    # output directory, so what ran is stored with what it produced.
    job_dir = out_dir / "job"
    archive = job_dir / ARCHIVE_NAME
    staged_script = job_dir / job_script.name
    try:
        check_repo(repo)
    except FileNotFoundError as exc:
        print(f"wcfm submit: {exc}", file=sys.stderr)
        return 2
    # `git_environment` is empty off a checkout -- an unpacked archive re-submitting, say --
    # so the parts are joined rather than interpolated, and the `.sub` has no dangling space.
    job_env = " ".join(
        p
        for p in (
            "CLUSTER_ID=$(ClusterId) JOB_ID=$(ProcId)",
            git_environment(repo),
        )
        if p
    )

    sub_file = out_dir / f"{run_name}.sub"
    sub_file.write_text(
        f"""universe                = vanilla
notification            = never
executable              = {staged_script}
arguments               = "{archive.name} {pyenv} {out_dir} {run_name} {cache} {gpus} {quoted}"
environment             = "{job_env}"
initialdir              = {out_dir}
+JobBatchName           = "{run_name}"
output                  = {out_dir}/$(ClusterId).$(ProcId).out
error                   = {out_dir}/$(ClusterId).$(ProcId).err
log                     = {out_dir}/$(ClusterId).$(ProcId).log
getenv                  = False
request_memory          = {os.environ.get("WCFM_REQUEST_MEMORY", "32000")}
request_cpus            = {os.environ.get("WCFM_REQUEST_CPUS", "4")}
request_gpus            = {gpus}
request_disk            = {request_disk("50000000")}
Requirements            = {os.environ.get("WCFM_GPU_REQUIREMENTS", DEFAULT_REQUIREMENTS)}
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
transfer_input_files    = {archive}
transfer_output_files   = ""
queue 1
"""
    )

    if dry_run:
        # The archive is NOT written: a dry run of a sweep would leave one per point. What it
        # would contain is checked above, which is the half that catches a real failure.
        print(f"\ndry run: wrote {sub_file}, submitting nothing\n")
        print(sub_file.read_text())
        return 0

    pack_repo(repo, job_dir)
    stage_executable(job_script, job_dir)

    if not shutil.which("condor_submit"):
        print("wcfm submit: condor_submit not on PATH; submit from a login node", file=sys.stderr)
        return 2

    print(f"\nSubmitting {sub_file}")
    completed = subprocess.run(["condor_submit", str(sub_file)], check=False)
    print(f"\nWatch with:  condor_q -nobatch ; tail -F {out_dir}/*.out")
    return completed.returncode
