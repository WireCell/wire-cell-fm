#!/bin/bash
#
# The gpu and distributed suites on a worker. Submitted by `wcfm test --gpu`.
#
# Args (positional):
#   $1 archive  -- basename of the transferred repo tarball (unpacked into scratch)
#   $2 pyenv    -- the cluster uv venv (torch/warpconvnet stack)
#   $3 outdir   -- where to leave the report
#   $4 pytest_k -- optional -k expression, "" for none
#   $5 cache_dir -- base for the warpconvnet benchmark cache (GPFS)
#
# Two stages, because "can I run this" has two different answers: the `gpu` suite is one
# process on one device, and the `distributed` suite is `torchrun --nproc_per_node=2` so that
# every rank is a pytest process. Both need the same environment, so they share one job.
#
# NB: this file is a COPY, staged beside the archive at submit time, and Condor transfers it
# here like any other input. Editing the checkout's copy cannot reach a queued or running job.

set -uo pipefail

# --- the repo arrives as an archive ------------------------------------------
# The submitting command packaged this tree (wcfm/cli/jobpack.py) and Condor transferred it here
# when the job started. Transferred files land in the job's scratch directory, which is also
# where we were started, so `$1` is a bare filename.
#
# Unpacked into `repo/` rather than over the scratch root, so that the cwd still holds nothing
# importable: `python -m wcfm.cli` puts the cwd first on sys.path, ahead of PYTHONPATH, and
# cluster 2261 imported a login-node checkout exactly that way.
repo_archive=$(readlink -f "$1")
repodir="${_CONDOR_SCRATCH_DIR:-$PWD}/repo"
mkdir -p "$repodir"
tar xzf "$repo_archive" -C "$repodir" || { echo "FATAL: cannot unpack ${repo_archive}"; exit 3; }
[ -d "${repodir}/wcfm" ] && [ -d "${repodir}/wirecell_fm.egg-info" ] || {
  echo "FATAL: ${repo_archive} has no wcfm/ + wirecell_fm.egg-info. The model's config schemas"
  echo "come from an entry point read from that metadata; without it no model= preset resolves."
  exit 3
}
pyenv=$2
outdir=$3
pytest_k=${4:-}
cache_dir=$5

# A rank that fails an assertion does not fail the test: pytest does not coordinate ranks, so
# the surviving rank blocks in the next collective until the job's wall clock runs out. A hard
# timeout turns that into a reported hang instead of a silently eaten slot.
DIST_TIMEOUT=${DIST_TIMEOUT:-900}
GPU_TIMEOUT=${GPU_TIMEOUT:-900}

echo "wcfm test --gpu on $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
ngpus=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
echo "  ngpus=${ngpus}"

# Same rule as trainjob.sh. sgpu0003/4 are PCIe-only L40S whose GPU-to-GPU P2P transport
# hangs at this driver/NCCL level: the group initialises, the startup broadcasts succeed, and
# the first AllReduce never returns. Without these a hang reads as "the engine is broken".
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# The benchmark cache is copied to node-local scratch and merged back at the end, so a job does
# not pay warpconvnet's autotuning on every start and new op shapes reach later jobs. Left unset
# it defaults to ~/.cache/warpconvnet on NFS *home*, which is wrong twice over: home is the wrong
# filesystem for job scratch, and `_do_save` takes an fcntl.flock(LOCK_EX) with no timeout, which
# over NFS can block indefinitely -- two ranks of the distributed suite contend for exactly that
# lock. trainjob.sh:48-57 and evaljob.sh:41-46 have always done this; this script and
# spike_c_job.sh were the two that missed it.
wp_cache_gpfs="${cache_dir}/warpconvnet"
wp_cache="${_CONDOR_SCRATCH_DIR}/warpconvnet"
mkdir -p "$wp_cache_gpfs"
cp -r "$wp_cache_gpfs" "$wp_cache" 2>/dev/null || mkdir -p "$wp_cache"
export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"
export PYTHONUNBUFFERED=1
echo "  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "  WARPCONVNET_BENCHMARK_CACHE_DIR=${wp_cache}"

source "${pyenv}/bin/activate"

# Nothing is installed by this job. lightning-fabric and pytest live in shared directories on
# GPFS, built once, reached by PYTHONPATH -- so the production venv stays byte-identical and
# the worker never needs uv on its PATH (getenv=False leaves ~/.local/bin off it, which killed
# an earlier submission in six seconds).
#
# Rebuild them with:
#   uv pip install --python ${pyenv}/bin/python --target ${WCFM_LIBS} --no-deps \
#       'lightning-fabric>=2.6,<3' lightning-utilities
#   uv pip install --python ${pyenv}/bin/python --target ${WCFM_TESTLIBS} \
#       pytest 'hydra-core>=1.3.6,<1.4' 'omegaconf>=2.3,<2.4'
#
# --no-deps and lightning-fabric (not the umbrella `lightning`) both matter. Without --no-deps
# the resolver drops a SECOND torch into the target, which shadows the pinned 2.10.0+cu128 on
# PYTHONPATH. The umbrella package pulls lightning.pytorch and therefore torchmetrics, absent.
WCFM_LIBS="${WCFM_LIBS:-/gpfs01/lbne/users/fm/${USER}/wcfm-libs}"
WCFM_TESTLIBS="${WCFM_TESTLIBS:-/gpfs01/lbne/users/fm/${USER}/wcfm-testlibs}"
for d in "${WCFM_LIBS}/lightning_fabric" "${WCFM_TESTLIBS}/pytest"; do
  [ -d "$d" ] || { echo "FATAL: missing ${d}. Build it once (commands in this script)."; exit 3; }
done
export PYTHONPATH="${WCFM_LIBS}:${WCFM_TESTLIBS}:${repodir}${PYTHONPATH:+:$PYTHONPATH}"
echo "  WCFM_LIBS=${WCFM_LIBS}"
echo "  WCFM_TESTLIBS=${WCFM_TESTLIBS}"

# The stack has to be the pinned one. A second torch on PYTHONPATH is the failure this repo's
# pyproject.toml warns about, and it would make every result below meaningless.
python - <<'PY' || { echo "FATAL: wrong torch on the path"; exit 3; }
import sys
import torch, lightning_fabric, pytest, hydra
assert "uvenv" in torch.__file__, f"torch came from {torch.__file__}, not the pinned venv"
print(f"torch {torch.__version__} | lightning-fabric {lightning_fabric.__version__} | "
      f"pytest {pytest.__version__} | hydra {hydra.__version__}")
print(f"cuda {torch.cuda.is_available()} devices {torch.cuda.device_count()}")
sys.exit(0 if torch.cuda.device_count() >= 1 else 4)
PY

cd "${repodir}" || exit 3
mkdir -p "$outdir"
kflag=()
[ -n "$pytest_k" ] && kflag=(-k "$pytest_k")

echo ""
echo "########## STAGE 1: the gpu suite, one process, one device ##########"
# -s: the AMP drift and timing numbers are printed, not asserted -- and pytest captures and
# DISCARDS stdout from a passing test, so without this the report that is the whole point of
# "reported, not asserted" never reaches the log. Cluster 2250 lost them exactly that way.
timeout "${GPU_TIMEOUT}" python -m pytest -m "gpu and not distributed" -v -ra -s \
    "${kflag[@]}" --junit-xml="${outdir}/gpu.xml"
rc_gpu=$?
[ $rc_gpu -eq 124 ] && echo "TIMEOUT after ${GPU_TIMEOUT}s"

echo ""
echo "########## STAGE 2: the distributed suite, torchrun, two ranks ##########"
if [ "$ngpus" -lt 2 ]; then
  # Not a skip. Every assertion in that suite is about ranks agreeing, and one rank always
  # agrees with itself -- reporting it green on one device is the false confidence the suite
  # exists to remove. The fixture asserts world_size==2 too; this is the earlier, clearer say.
  echo "FATAL: the distributed suite needs 2 GPUs, got ${ngpus}."
  echo "Running it on one rank would report green while testing nothing."
  rc_dist=5
else
  timeout "${DIST_TIMEOUT}" torchrun --standalone --nnodes=1 --nproc_per_node=2 \
      -m pytest -m distributed -v -ra "${kflag[@]}" --junit-xml="${outdir}/distributed.xml"
  rc_dist=$?
  if [ $rc_dist -eq 124 ]; then
    echo "TIMEOUT after ${DIST_TIMEOUT}s."
    echo "A rank that fails an assertion leaves the others in a collective forever, so a"
    echo "timeout here usually means one rank asserted. Look for the first rank to report."
  fi
fi

report="${outdir}/test_gpu_report.txt"
{
  echo "wcfm test --gpu -- $(date -Is) -- $(hostname)"
  echo "ngpus=${ngpus}  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
  echo ""
  echo "STAGE 1  gpu suite (1 device)       : $([ $rc_gpu -eq 0 ] && echo PASS || echo FAIL)  (rc=${rc_gpu})"
  echo "STAGE 2  distributed suite (2 ranks): $([ $rc_dist -eq 0 ] && echo PASS || echo FAIL)  (rc=${rc_dist})"
  echo ""
  [ $rc_gpu -eq 124 ] && echo "stage 1 TIMED OUT (${GPU_TIMEOUT}s)"
  [ $rc_dist -eq 124 ] && echo "stage 2 TIMED OUT (${DIST_TIMEOUT}s) -- probably one rank asserted"
  [ $rc_dist -eq 5 ] && echo "stage 2 NOT RUN -- fewer than 2 GPUs; the suite would be vacuous"
  echo "junit: ${outdir}/gpu.xml ${outdir}/distributed.xml"
} | tee "$report"

# Merge the autotune results back. rsync writes a temp name and renames, so concurrent merges
# from several jobs are safe.
rsync -a "${wp_cache}/" "${wp_cache_gpfs}/" || true

echo ""
echo "Report written to ${report}"
exit $(( rc_gpu != 0 || rc_dist != 0 ? 1 : 0 ))
