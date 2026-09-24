"""`wcfm datagen`: queue a dataset-building module on a CPU worker.

Shard and pack creation stream a whole production through `DirectDataset`, two HDF5 opens per
event over GPFS, which is hours at two million events and nothing a GPU helps with. The job is
queued under the same rules as `wcfm submit`: the tree is packed into an archive beside the
job's output, the job script is staged next to it, and the worker imports from the venv plus
the unpacked archive. The worker-side half is `gridutils/datagen/datajob.sh`.
"""

from __future__ import annotations

import importlib.util
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
    pop_opt,
    request_disk,
    stage_executable,
)

__all__ = ["main"]

USAGE = """usage: wcfm datagen [options] <job_name> <module> [module args ...]

  wcfm datagen shards_2M create_shards \\
      --datadir /gpfs01/.../fdhd_sparse_numu_training /gpfs01/.../fdhd_sparse_nue_training_apa0 \\
      --apa 0 --view W --outdir /gpfs01/.../shards_fdhd_sparse_2M_mixed_apa0W \\
      --shard_size 4000 --threads 32 --writers 4
  wcfm datagen shards_2M create_shards ... --dry-run

<module> is a module under wcfm.data.prep: `create_shards` or `pack_dataset`. Its arguments
pass through unchanged; `python -m wcfm.data.prep.<module> --help` lists them.

options:
  --dry-run          write the .sub file and print it, submit nothing
  --repo DIR         checkout to package for the job (default: this package's repository)

Resources are environment variables: WCFM_REQUEST_MEMORY (MB, default 24000: create_shards
holds one block of events, about 11 GB at --block_files 20000; a pack holds the whole dataset
and needs 64000 at 200k events), WCFM_REQUEST_CPUS (default 8: the shard writers compress on
them, the reader threads mostly wait on GPFS), WCFM_PYENV, WCFM_OUTPUT_BASE. Condor's files go under
<WCFM_OUTPUT_BASE>/datagen/<job_name>/; the shards or the pack go where the module's own
--outdir or --out_path says. `create_shards` skips shard files that already exist, so
resubmitting the same command resumes an interrupted build.
"""

DEFAULT_MEMORY_MB = "24000"
DEFAULT_CPUS = "8"


def _repo_root() -> Path:
    import wcfm

    return Path(wcfm.__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    args = list(argv)
    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")
    repo = Path(pop_opt(args, "--repo", str(_repo_root()))).expanduser().resolve()

    if len(args) < 2:
        print("wcfm datagen: a job name and a module are required\n" + USAGE, file=sys.stderr)
        return 2
    job_name, module, module_args = args[0], args[1], args[2:]
    if not job_name.replace("_", "").replace("-", "").isalnum():
        print(f"wcfm datagen: job name {job_name!r} must be [A-Za-z0-9_-]", file=sys.stderr)
        return 2
    dotted = f"wcfm.data.prep.{module}"
    if importlib.util.find_spec(dotted) is None:
        print(
            f"wcfm datagen: no module {dotted}. The prep tools are the modules under "
            f"wcfm/data/prep/: create_shards, pack_dataset.",
            file=sys.stderr,
        )
        return 2

    user = os.environ.get("USER", "unknown")
    pyenv = Path(os.environ.get("WCFM_PYENV", f"/gpfs01/lbne/users/fm/{user}/uvenv"))
    output_base = Path(
        os.environ.get("WCFM_OUTPUT_BASE", f"/gpfs01/lbne/users/fm/{user}/CONDOR_OUT")
    )

    job_script = repo / "gridutils" / "datagen" / "datajob.sh"
    if not job_script.is_file():
        print(f"wcfm datagen: no job script at {job_script}", file=sys.stderr)
        return 2
    # The job installs nothing: the venv has to carry h5py, torch and warpconvnet before it is
    # queued, because `wcfm.data` imports all three.
    if not (pyenv / "bin" / "python").is_file():
        print(
            f"wcfm datagen: no venv at {pyenv}. Build it once:\n  gridutils/build_env.sh",
            file=sys.stderr,
        )
        return 2

    out_dir = output_base / "datagen" / job_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"job:     {job_name}  ({dotted})")
    print(f"repo:    {repo}")
    print(f"output:  {out_dir}")

    job_dir = out_dir / "job"
    archive = job_dir / ARCHIVE_NAME
    staged_script = job_dir / job_script.name
    try:
        check_repo(repo)
    except FileNotFoundError as exc:
        print(f"wcfm datagen: {exc}", file=sys.stderr)
        return 2
    job_env = " ".join(
        p for p in ("CLUSTER_ID=$(ClusterId) JOB_ID=$(ProcId)", git_environment(repo)) if p
    )
    # Quoted individually so an argument with a space survives Condor's splitting.
    quoted = " ".join(f"'{a}'" for a in module_args)

    sub_file = out_dir / f"{job_name}.sub"
    sub_file.write_text(
        f"""universe                = vanilla
notification            = never
executable              = {staged_script}
arguments               = "{archive.name} {pyenv} {dotted} {quoted}"
environment             = "{job_env}"
initialdir              = {out_dir}
+JobBatchName           = "datagen-{job_name}"
output                  = {out_dir}/$(ClusterId).$(ProcId).out
error                   = {out_dir}/$(ClusterId).$(ProcId).err
log                     = {out_dir}/$(ClusterId).$(ProcId).log
getenv                  = False
request_memory          = {os.environ.get("WCFM_REQUEST_MEMORY", DEFAULT_MEMORY_MB)}
request_cpus            = {os.environ.get("WCFM_REQUEST_CPUS", DEFAULT_CPUS)}
request_disk            = {request_disk("5000000")}
should_transfer_files   = YES
stream_output           = True
stream_error            = True
when_to_transfer_output = ON_EXIT
transfer_input_files    = {archive}
transfer_output_files   = ""
queue 1
"""
    )

    if dry_run:
        print(f"\ndry run: wrote {sub_file}, submitting nothing\n")
        print(sub_file.read_text())
        return 0

    pack_repo(repo, job_dir)
    stage_executable(job_script, job_dir)

    if not shutil.which("condor_submit"):
        print("wcfm datagen: condor_submit not on PATH; submit from a login node", file=sys.stderr)
        return 2

    print(f"\nSubmitting {sub_file}")
    completed = subprocess.run(["condor_submit", str(sub_file)], check=False)
    print(f"\nWatch with:  condor_q -nobatch ; tail -F {out_dir}/*.out")
    return completed.returncode
