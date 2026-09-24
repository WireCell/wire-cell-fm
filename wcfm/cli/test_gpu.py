"""`wcfm test --gpu`: the named trigger for the suites a CPU cannot run.

No backbone forward is CPU-testable, since sparse convolutions raise without CUDA, so the `gpu`
and `distributed` markers carry the parts of the framework that matter most on the cluster and
least on a laptop. This command is what makes them a gate rather than a marker nobody runs.

It submits and returns. The job runs two stages -- the `gpu` suite as one process on one device,
then the `distributed` suite under `torchrun --nproc_per_node=2` so that every rank is a pytest
process -- and writes a report next to its output. Polling for it is `condor_q`'s job.

Two preconditions are checked here rather than on the worker, because a job that dies in six
seconds for a reason visible from the login node wastes a queue slot:

- the shared library directories exist. Nothing is installed by the job, and `getenv=False`
  leaves `uv` off the worker's PATH.
- the tree packages, in particular that `wire_cell_fm.egg-info` is there, without which no
  `model=` preset resolves on the worker.

The tests the job runs are the ones in the tree at submit time: they are packaged into an
archive beside the report and transferred (`wcfm/cli/jobpack.py`), so the checkout can be edited
the moment this command returns.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import datetime
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

USAGE = """usage: wcfm test (--gpu | --dist-cpu) [options]

  wcfm test --dist-cpu                 run the distributed suite HERE, on 2 CPU ranks (~7 s)
  wcfm test --gpu                      submit the gpu and distributed suites to Condor
  wcfm test --gpu -k reduce            pass a -k expression through to pytest
  wcfm test --gpu --dry-run            write the .sub file and print it, submit nothing

options:
  --dist-cpu         run `torchrun --nproc_per_node=2 -m pytest -m distributed` locally, over
                     gloo. DDP's reduction and its per-forward arming are backend-independent,
                     so this tests the same thing as the 2-GPU job and needs no queue slot.
  --gpu              submit to Condor: the gpu suite on one device, then the distributed suite
                     under torchrun on two
  -k EXPR            pytest -k expression, forwarded to both stages
  --dry-run          build the submit file without submitting
  --repo DIR         checkout to package for the job (default: this package's repository)
  --gpus N           request N GPUs (default 2; fewer makes the distributed suite vacuous)

The CPU suites need no trigger: `pytest -m "not gpu and not distributed"`.
"""

DEFAULT_REQUIREMENTS = '(GPUs_DeviceName == "NVIDIA L40S") && (GPUs_Capability == 8.9)'
# Deliberately smaller than a training job's. `trainjob.sh` and the spikes ask for 32000, and
# for a run that streams shards into a buffer that is right -- but `condor_q -better-analyze`
# on the first submissions showed only 3 of the 13 L40S slots carry >=32 GB, and just one could
# run the job at all: it sat idle for an hour. These suites train a two-layer toy and build
# sparse convolutions of a few thousand voxels, so the request was costing slots it had no use
# for. Override with WCFM_REQUEST_MEMORY when a test genuinely needs more.
DEFAULT_MEMORY_MB = "12000"


def _repo_root() -> Path:
    import wcfm

    return Path(wcfm.__file__).resolve().parent.parent


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except Exception:  # noqa: BLE001 - no git, no repo, or a timeout are all "unknown"
        return None
    return out.stdout.strip()


def _dist_cpu(args: list[str]) -> int:
    """The distributed suite here and now, on two CPU ranks over gloo.

    This exists because the 2-GPU slot is scarce: the first attempt to verify a DDP
    correctness fix sat idle for ninety minutes, and the same suite runs in about seven
    seconds locally. DDP's gradient reduction and the arming of its reducer are
    backend-independent -- measured with a control, which is itself a test in the suite -- so
    this is the same signal, not a weaker one. What it cannot cover is anything that needs a
    device: the `gpu` suite stays on Condor.
    """
    repo = Path(pop_opt(args, "--repo", str(_repo_root()))).expanduser().resolve()
    extra: list[str] = []
    if k := pop_opt(args, "-k", ""):
        extra += ["-k", k]
    if args:
        print(f"wcfm test: unrecognised arguments {args}", file=sys.stderr)
        return 2

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        "-m",
        "pytest",
        "-m",
        "distributed",
        "-q",
        *extra,
    ]
    print(f"running: {' '.join(cmd)}\n  cwd={repo}")
    # `python -m torch.distributed.run` rather than the `torchrun` script, so this works
    # wherever the interpreter is, without needing its bin/ on PATH -- the same reason the
    # job scripts pass absolute paths around.
    return subprocess.run(cmd, cwd=repo, check=False).returncode


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    args = list(argv)
    if "--dist-cpu" in args:
        args.remove("--dist-cpu")
        return _dist_cpu(args)
    if "--gpu" not in args:
        print(
            "wcfm test: --gpu or --dist-cpu is required. The CPU suites are plain pytest:\n"
            '  pytest -m "not gpu and not distributed"',
            file=sys.stderr,
        )
        return 2
    args.remove("--gpu")

    dry_run = "--dry-run" in args
    if dry_run:
        args.remove("--dry-run")

    pytest_k = pop_opt(args, "-k", "")
    repo = Path(pop_opt(args, "--repo", str(_repo_root()))).expanduser().resolve()
    gpus = int(pop_opt(args, "--gpus", "2"))
    if args:
        print(f"wcfm test: unrecognised arguments {args}", file=sys.stderr)
        return 2

    user = os.environ.get("USER", "unknown")
    pyenv = Path(os.environ.get("WCFM_PYENV", f"/gpfs01/lbne/users/fm/{user}/uvenv"))
    output_base = Path(
        os.environ.get("WCFM_OUTPUT_BASE", f"/gpfs01/lbne/users/fm/{user}/CONDOR_OUT")
    )
    # Passed as an argument rather than read from the environment inside the job: the submit
    # sets getenv=False, so a WCFM_CACHE_DIR read on the worker would silently take the default
    # every time. Same path and same reason as submit.py:157.
    cache = Path(os.environ.get("WCFM_CACHE_DIR", f"/gpfs01/lbne/users/fm/{user}/cache"))

    job_script = repo / "gridutils" / "test" / "test_gpu_job.sh"
    if not job_script.is_file():
        print(f"wcfm test: no job script at {job_script}", file=sys.stderr)
        return 2

    # The job installs nothing, so the venv has to carry pytest and the framework before it is
    # queued, alongside the GPU stack.
    if not (pyenv / "bin" / "python").is_file():
        print(f"wcfm test: no venv at {pyenv}. Build it once:\n  gridutils/build_env.sh",
              file=sys.stderr)
        return 2

    if gpus < 2:
        # Said here as well as in the job script and the fixture, because it is the one way to
        # get a green distributed suite that tested nothing.
        print(
            f"wcfm test: --gpus {gpus} will skip the distributed suite. Every assertion in it "
            "is about ranks agreeing, and one rank always agrees with itself.",
            file=sys.stderr,
        )

    run_name = f"wcfm_test_gpu_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = output_base / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    sha = _git("rev-parse", "--short", "HEAD", cwd=repo) or "unknown"
    dirty = bool(_git("status", "--porcelain", cwd=repo))
    print(f"repo:   {repo} @ {sha}{' (DIRTY)' if dirty else ''}")
    print(f"pyenv:  {pyenv}")
    print(f"output: {out_dir}")
    if dirty:
        # Not an error -- iterating on a GPU test with a dirty tree is the normal way to use
        # this -- but the report must not claim to describe the commit. The tree is archived
        # below, so what ran stays readable even though the sha does not identify it.
        print("note: the working tree is dirty; the archive, not the sha, is what ran.")

    job_dir = out_dir / "job"
    archive = job_dir / ARCHIVE_NAME
    staged_script = job_dir / job_script.name
    try:
        check_repo(repo, also=("tests",))
    except FileNotFoundError as exc:
        print(f"wcfm test: {exc}", file=sys.stderr)
        return 2

    # `git_environment` is empty off a checkout, so the parts are joined rather than
    # interpolated: no dangling space in the `.sub`.
    env = " ".join(
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
arguments               = "{archive.name} {pyenv} {out_dir} '{pytest_k}' {cache}"
environment             = "{env}"
+JobBatchName           = "{run_name}"
output                  = {out_dir}/$(ClusterId).$(ProcId).out
error                   = {out_dir}/$(ClusterId).$(ProcId).err
log                     = {out_dir}/$(ClusterId).$(ProcId).log
getenv                  = False
request_memory          = {os.environ.get("WCFM_REQUEST_MEMORY", DEFAULT_MEMORY_MB)}
request_cpus            = {os.environ.get("WCFM_REQUEST_CPUS", "4")}
request_gpus            = {gpus}
request_disk            = {request_disk("20000000")}
Requirements            = {os.environ.get("WCFM_GPU_REQUIREMENTS", DEFAULT_REQUIREMENTS)}
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
        print("wcfm test: condor_submit not on PATH; submit from a login node", file=sys.stderr)
        return 2

    print(f"\nSubmitting {sub_file}")
    completed = subprocess.run(["condor_submit", str(sub_file)], check=False)
    print(f"\nReport will be at {out_dir}/test_gpu_report.txt")
    print(f"Watch with:  condor_q -nobatch ; tail -F {out_dir}/*.out")
    return completed.returncode
