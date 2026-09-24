"""The evaluation DAG: per-epoch `extract -> probes`, then one `merge`.

## What this replaces

`collect_probes.sh` was a login-node process sleeping 120 s in a loop with a 12-hour timeout, no
retries and no hold recovery. Every failure mode it hand-rolled is one Condor already handles,
and it handled them worse: a login node that gets rebooted loses the whole campaign, and the
"features are older than the checkpoint" staleness rule it used is defeated in one direction by
a re-run against an unchanged checkpoint and in the other by a slow GPFS write.

So: a DAG. Condor owns retries (`RETRY`), holds, and the ordering. Nothing sleeps.

## The shape

    extract_ep5  --> probes_ep5  --\\
    extract_ep10 --> probes_ep10 ---+--> merge
    extract_ep20 --> probes_ep20  --/

`extract` needs a GPU; `probes` and `merge` do not, and that is the point of the split. The GPU
pass happens once per checkpoint while every metric stays on a CPU slot, which is what makes the
suite cheap to re-run. Epochs are independent, so a queue that only has one GPU free still makes
progress rather than blocking the campaign.

`merge` is not allowed to fail the DAG. It produces no new data -- every number in its table is
read out of JSONs that are the durable result -- so a failed merge costs a table that rebuilds
in seconds, while failing the DAG over it would discard a campaign's worth of completed probe
jobs. It is declared with `RETRY ... UNLESS-EXIT 0` semantics through a `SCRIPT POST` that
always succeeds.

Staleness is decided by the PRE script, on hashes. `provenance.json` records
`checkpoint_sha256`, so an extraction is current if and only if it was produced from this
checkpoint's bytes. Comparing modification times instead is defeated by a re-run against an
unchanged checkpoint, and watching for a file to stop growing is defeated by a slow GPFS write.
Extraction writes through a temp file and renames, so a reader never sees a half-written array
and there is nothing to settle for.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

__all__ = ["DagPlan", "build_dag", "checkpoints_of", "epoch_of_checkpoint"]

DEFAULT_REQUIREMENTS = os.environ.get(
    "WCFM_GPU_REQUIREMENTS",
    '(GPUs_DeviceName == "NVIDIA L40S") && (GPUs_Capability == 8.9)',
)


def epoch_of_checkpoint(path: Path) -> int:
    digits = "".join(c for c in Path(path).stem if c.isdigit())
    return int(digits) if digits else -1


def checkpoints_of(run_dir: Path, epochs: str = "") -> list[Path]:
    """Which checkpoints the DAG covers.

    `latest.pt` is excluded even when it is the only file: it duplicates some
    `checkpoint_epochN.pt` under a name whose meaning changes between runs, so a features
    directory called `latest` would silently mean a different epoch in each of them.
    """
    found = sorted(
        (run_dir / "checkpoints").glob("checkpoint_epoch*.pt"), key=epoch_of_checkpoint
    )
    if not epochs:
        return found
    want = {int(e) for e in epochs.split(",") if e.strip()}
    by_epoch = {epoch_of_checkpoint(p): p for p in found}
    missing = sorted(want - set(by_epoch))
    if missing:
        raise SystemExit(
            f"{run_dir.name} has no checkpoint for epochs {missing}; it has {sorted(by_epoch)}. "
            "`save_at` in the run config is what makes a sweep's points land on the same epochs."
        )
    return [by_epoch[e] for e in sorted(want)]


@dataclass
class DagPlan:
    """Everything the DAG needs, resolved on the login node before anything is queued."""

    run_dir: Path
    out_root: Path
    repo: Path
    #: The tree the nodes run from, as one archive transferred to each worker. Condor reads
    #: `transfer_input_files` when a job STARTS -- and a DAG's later nodes start hours after
    #: submission -- so this is what stops an edit to the checkout from reaching a queued
    #: node. Written by `wcfm eval submit`; named here so `--dry-run` prints the real `.sub`.
    repo_archive: Path
    pyenv: Path
    cache: Path
    checkpoints: list[Path]
    eval_set_root: Path
    stages: str
    sources: str
    taps: str
    rows: str
    max_images: int
    device: str = "cuda"
    retry: int = 2
    extract_flags: tuple[str, ...] = ()
    #: The repo's identity as Condor `environment` assignments, from
    #: `jobpack.git_environment`. The workers have no `.git`, so without this every eval run's
    #: provenance records a null git block -- on exactly the runs whose provenance matters.
    git_env: str = ""
    request_memory_gpu: str = "32000"
    request_memory_cpu: str = "16000"
    request_cpus: str = "4"
    #: Never set before file transfer was turned on. The default Condor computes is the size
    #: of the inputs -- 1.2 MB -- while extraction writes whole feature stores into the same
    #: scratch directory. The L40S nodes advertise terabytes.
    request_disk_gpu: str = "50000000"
    request_disk_cpu: str = "4000000"

    @property
    def epochs(self) -> list[int]:
        return [epoch_of_checkpoint(p) for p in self.checkpoints]


def _sub(
    *,
    executable: Path,
    arguments: str,
    log_dir: Path,
    name: str,
    gpus: int,
    memory: str,
    cpus: str,
    disk: str,
    archive: Path,
    git_env: str,
    initialdir: Path,
) -> str:
    """One node's submit description.

    `executable` is a *copy* of the job script, staged beside the archive: Condor transfers the
    executable at job start like any other input, so a node naming the live checkout would run
    whatever that file said hours later -- and bash reads a script incrementally, so an edit to
    a running node's script truncates it silently.

    `transfer_output_files = ""` is not decoration. With transfer on and no such line, Condor
    copies everything created at the top of the scratch directory back to `initialdir` -- which
    is where these jobs already rsync their real output, so the duplicate would land on top of
    the correct result and look right.
    """
    requirements = f"Requirements            = {DEFAULT_REQUIREMENTS}\n" if gpus else ""
    return f"""universe                = vanilla
notification            = never
executable              = {executable}
arguments               = "{arguments}"
environment             = "CLUSTER_ID=$(ClusterId) JOB_ID=$(ProcId) {git_env}"
initialdir              = {initialdir}
+JobBatchName           = "{name}"
output                  = {log_dir}/{name}.$(ClusterId).out
error                   = {log_dir}/{name}.$(ClusterId).err
log                     = {log_dir}/{name}.$(ClusterId).log
getenv                  = False
request_memory          = {memory}
request_cpus            = {cpus}
request_gpus            = {gpus}
request_disk            = {disk}
{requirements}should_transfer_files   = YES
stream_output           = True
stream_error            = True
when_to_transfer_output = ON_EXIT
transfer_input_files    = {archive}
transfer_output_files   = ""
queue 1
"""


def build_dag(plan: DagPlan) -> dict[Path, str]:
    """Every file the DAG needs, as `{path: contents}`. Writes nothing.

    Returned rather than written so `--dry-run` prints exactly what a real submission would
    queue -- the failure this avoids is a dry run that exercises a different code path from the
    real one and therefore proves nothing about it.
    """
    out: dict[Path, str] = {}
    dag_dir = plan.out_root / "dag"
    log_dir = dag_dir / "logs"
    job_dir = dag_dir / "job"
    gridutils = plan.repo / "gridutils"

    # The three job scripts are *copied* into the DAG's own directory and the copies are what
    # the nodes execute. Returned as content like every other file this builds, so `--dry-run`
    # still writes nothing and still describes exactly what a real submission would queue.
    staged = {}
    for script in ("evaljob.sh", "probesjob.sh", "mergejob.sh"):
        staged[script] = job_dir / script
        out[job_dir / script] = (gridutils / "eval" / script).read_text()

    lines = [
        f"# wcfm eval submit -- {plan.run_dir.name}",
        f"# {len(plan.checkpoints)} epoch(s): {plan.epochs}",
        "",
    ]

    probe_nodes = []
    for ckpt in plan.checkpoints:
        epoch = epoch_of_checkpoint(ckpt)
        store = plan.out_root / f"epoch{epoch}"
        ex_name, pr_name = f"extract_ep{epoch}", f"probes_ep{epoch}"

        extract_flags = [
            f"--epochs={epoch}",
            f"--eval-set-root={plan.eval_set_root}",
            f"--sources={plan.sources}",
            f"--rows={plan.rows}",
            f"--max-images={plan.max_images}",
            f"--device={plan.device}",
            *(["--taps=" + plan.taps] if plan.taps else []),
            *plan.extract_flags,
        ]
        out[dag_dir / f"{ex_name}.sub"] = _sub(
            executable=staged["evaljob.sh"],
            arguments=" ".join(
                [
                    plan.repo_archive.name,
                    str(plan.pyenv),
                    str(plan.run_dir),
                    str(plan.out_root),
                    str(plan.cache),
                    *(f"'{f}'" for f in extract_flags),
                ]
            ),
            log_dir=log_dir,
            name=ex_name,
            gpus=1,
            memory=plan.request_memory_gpu,
            cpus=plan.request_cpus,
            disk=plan.request_disk_gpu,
            archive=plan.repo_archive,
            git_env=plan.git_env,
            initialdir=plan.out_root,
        )
        out[dag_dir / f"{pr_name}.sub"] = _sub(
            executable=staged["probesjob.sh"],
            arguments=" ".join(
                [
                    plan.repo_archive.name,
                    str(plan.pyenv),
                    str(store),
                    str(plan.run_dir / "probes"),
                    plan.stages,
                ]
            ),
            log_dir=log_dir,
            name=pr_name,
            gpus=0,  # the whole point of the split: metrics stay on CPU slots
            memory=plan.request_memory_cpu,
            cpus=plan.request_cpus,
            disk=plan.request_disk_cpu,
            archive=plan.repo_archive,
            git_env=plan.git_env,
            initialdir=plan.out_root,
        )

        lines += [
            f"JOB {ex_name} {dag_dir / f'{ex_name}.sub'}",
            f"JOB {pr_name} {dag_dir / f'{pr_name}.sub'}",
            f"PARENT {ex_name} CHILD {pr_name}",
            # The PRE script skips an extraction whose provenance already records this
            # checkpoint's hash. Exit 1 from a PRE script means "do not run the node".
            f"SCRIPT PRE {ex_name} {dag_dir / 'stale.sh'} {ckpt} {store}",
            f"RETRY {ex_name} {plan.retry}",
            f"RETRY {pr_name} {plan.retry}",
            "",
        ]
        probe_nodes.append(pr_name)

    merge_name = "merge"
    out[dag_dir / f"{merge_name}.sub"] = _sub(
        executable=staged["mergejob.sh"],
        arguments=" ".join(
            [plan.repo_archive.name, str(plan.pyenv), str(plan.run_dir / "probes")]
        ),
        log_dir=log_dir,
        name=merge_name,
        gpus=0,
        memory="4000",
        cpus="1",
        disk=plan.request_disk_cpu,
        archive=plan.repo_archive,
        git_env=plan.git_env,
        initialdir=plan.out_root,
    )
    lines += [
        f"JOB {merge_name} {dag_dir / f'{merge_name}.sub'}",
        f"PARENT {' '.join(probe_nodes)} CHILD {merge_name}",
        # `merge` produces no new data -- its table rebuilds from the JSONs in seconds -- so
        # failing the DAG over it would discard a campaign's worth of completed probe jobs for
        # a view. The POST script always exits 0.
        f"SCRIPT POST {merge_name} {dag_dir / 'always_ok.sh'}",
        "",
    ]

    out[dag_dir / "eval.dag"] = "\n".join(lines)
    out[dag_dir / "stale.sh"] = _STALE_SH
    out[dag_dir / "always_ok.sh"] = _ALWAYS_OK_SH
    return out


#: PRE script. Exit 1 tells DAGMan to skip the node.
_STALE_SH = """#!/bin/bash
#
# Skip an extraction that is already current. Written by `wcfm eval submit`.
#
#   $1 checkpoint   $2 store directory
#
# An extraction is current when provenance.json's checkpoint_sha256 matches this checkpoint's
# bytes, and by nothing else. Comparing modification times is defeated by a re-run against an
# unchanged checkpoint, and waiting for a file to stop growing is defeated by a slow GPFS write.
#
# Exit 1 = do not run the node (DAGMan treats a non-zero PRE as a skip we have asked for).
set -u
ckpt=$1
store=$2
prov="${store}/provenance.json"

[ -f "$prov" ] || { echo "no ${prov}: extracting"; exit 0; }

recorded=$(python3 -c '
import json,sys
try:
    print(json.load(open(sys.argv[1])).get("checkpoint_sha256",""))
except Exception:
    print("")
' "$prov")
actual=$(sha256sum "$ckpt" | cut -d" " -f1)

if [ -n "$recorded" ] && [ "$recorded" = "$actual" ]; then
  echo "up to date: ${store} was extracted from this checkpoint (${actual:0:12})"
  exit 1
fi
echo "stale: recorded=${recorded:0:12} actual=${actual:0:12} -- extracting"
exit 0
"""

_ALWAYS_OK_SH = """#!/bin/bash
# POST script for the merge node. The table is a view over the probe JSONs and rebuilds in
# seconds, so a failed merge must not fail a DAG carrying a campaign's completed probe jobs.
echo "merge finished with status ${1:-?}; not failing the DAG (the table is a view)"
exit 0
"""


def quote(s: str) -> str:
    return shlex.quote(s)
