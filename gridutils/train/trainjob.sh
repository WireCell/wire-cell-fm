#!/bin/bash
#
# `wcfm train` on a Condor worker. Submitted by `wcfm submit`.
#
# Args (positional):
#   $1 archive   -- basename of the transferred repo tarball (unpacked into scratch)
#   $2 pyenv     -- the venv built by gridutils/build_env.sh (the whole stack)
#   $3 outdir    -- run directory on GPFS; scratch is rsynced here
#   $4 run_name  -- must match `run.name` in the overrides
#   $5 cache_dir -- base for the warpconvnet benchmark cache and the data index
#   $6 devices   -- `launch.devices` from the resolved config; the rank count
#   $7+ ...      -- Hydra overrides, passed through to `wcfm train`
#
# I/O strategy: write everything to $_CONDOR_SCRATCH_DIR (fast local disk) and
# rsync to GPFS, so partial outputs survive failure and preemption.
# `run.output_root` points at scratch.
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
[ -d "${repodir}/wcfm" ] && [ -d "${repodir}/wire_cell_fm.egg-info" ] || {
  echo "FATAL: ${repo_archive} has no wcfm/ + wire_cell_fm.egg-info. The model's config schemas"
  echo "come from an entry point read from that metadata; without it no model= preset resolves."
  exit 3
}
pyenv=$2
outdir=$3
run_name=$4
cache_dir=$5
devices=$6
shift 6
overrides=("$@")

# `devices` is `launch.devices` from the resolved config -- the ONE place the rank count is
# stated. `wcfm submit` composed the config and wrote it into the .sub alongside
# `request_gpus`, so the two cannot disagree by construction.
#
# What CAN still differ is what Condor actually granted, which is what CUDA_VISIBLE_DEVICES
# says. That is a real failure and it is fatal here rather than absorbed: launching
# `--nproc_per_node` from the granted count instead would silently run a different job than
# the config describes, and launching from the config on a smaller allocation puts two ranks
# on one device. The engine repeats the check against WORLD_SIZE (trainer.py::build_fabric)
# because torchrun is not the only way in.
granted=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-}")
if [ "${granted}" -ne "${devices}" ]; then
  echo "FATAL: launch.devices=${devices} but Condor granted ${granted} GPU(s)"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "  request_gpus is derived from launch.devices, so this means the allocation changed"
  echo "  under the job, not that the config is wrong. Resubmit."
  exit 4
fi
ngpus=${devices}

echo "Running ${CLUSTER_ID:-?}.${JOB_ID:-?} on $(hostname)"
echo "  run_name=${run_name}  ngpus=${ngpus} (launch.devices, granted ${granted})"
echo "  repodir=${repodir}"
echo "  outdir=${outdir}"
echo "  overrides=${overrides[*]}"
echo ""

# --- NCCL on PCIe-only L40S ---------------------------------------------------
# sgpu0003/4 have no NVLink and their GPU-to-GPU P2P transport hangs at this driver/NCCL
# level: the group initialises, the startup broadcasts succeed, and the first AllReduce never
# returns. Forcing the host shared-memory transport fixes it at no real cost. Harmless on one
# GPU, where no collective is issued.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

# The benchmark cache is copied to local scratch and merged back at the end, so a job does not
# pay warpconvnet's autotuning on every start and new op shapes reach later jobs. rsync writes
# each file to a temp name and renames, so concurrent merges from several jobs are safe.
wp_cache_gpfs="${cache_dir}/warpconvnet"
wp_cache="${_CONDOR_SCRATCH_DIR}/warpconvnet"
data_cache="${cache_dir}/data"
mkdir -p "$wp_cache_gpfs" "$data_cache"
cp -r "$wp_cache_gpfs" "$wp_cache" 2>/dev/null || mkdir -p "$wp_cache"
export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"
export PYTHONUNBUFFERED=1
echo "  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}  WARPCONVNET_BENCHMARK_CACHE_DIR=${wp_cache}"

source "${pyenv}/bin/activate"

# Nothing is installed by this job. Everything it needs is in the venv, built by
# `gridutils/build_env.sh` -- including hydra and omegaconf, which are runtime dependencies here
# because the config is composed on the worker. The repo is the only thing PYTHONPATH adds, and
# it is the archive this job unpacked.
export PYTHONPATH="${repodir}${PYTHONPATH:+:$PYTHONPATH}"

# Scratch is where we already are, but say it: nothing between here and the trainer may leave
# the job somewhere importable. See the unpack block at the top for why that matters.
cd "${_CONDOR_SCRATCH_DIR:-/tmp}"


# torch from the venv, `wcfm` from the archive -- not from the editable install `build_env.sh`
# leaves in the venv, pointing at whatever checkout built it. PYTHONPATH wins today; if that
# ever inverts the job runs different code with no other sign. realpath both sides: GPFS serves
# this tree as /gpfs01/... and as /gpfs/mnt/gpfs01/....
python - "$pyenv" "$repodir" <<'PY' || { echo "FATAL: wrong environment"; exit 3; }
import os, sys, torch, lightning_fabric, wcfm
pyenv, repodir = (os.path.realpath(p) for p in sys.argv[1:3])
assert os.path.realpath(torch.__file__).startswith(pyenv), \
    f"torch came from {torch.__file__}, not the venv at {pyenv}"
assert os.path.realpath(wcfm.__file__).startswith(repodir), \
    f"wcfm came from {wcfm.__file__}, not the unpacked archive at {repodir}"
print(f"torch {torch.__version__} | lightning-fabric {lightning_fabric.__version__} | "
      f"cuda {torch.cuda.is_available()} devices {torch.cuda.device_count()}")
PY

# `run.output_root` is scratch; the engine creates <root>/<run_name>/{checkpoints,debug,
# probes,features,metrics} itself, so the rsync is one directory.
scratch_root="${_CONDOR_SCRATCH_DIR}/runs"
scratch_run="${scratch_root}/${run_name}"
mkdir -p "$scratch_root"

# --- resume: bring back what the engine reads at startup ----------------------
# Scratch starts empty on every execution, so without this `run.resume=auto` finds no checkpoint
# and restarts from epoch 0 in silence, overwriting the GPFS metrics on the first sync.
#
# To avoid this, the job rsyncs back the last checkpoint (if available) and the two metrics streams.
# arrays/ is never read back and schema.json is rewritten from the writer's own
# names. rsync runs without --delete, so the older checkpoints on GPFS stay there.
# run_metadata.json is also not restored.
if [ -d "${outdir}/checkpoints" ]; then
  echo "Resuming: restoring from ${outdir}"
  mkdir -p "${scratch_run}/checkpoints" "${scratch_run}/metrics"
  restore() {
    [ -f "$1" ] || return 0
    rsync -a "$1" "$2" || {
      echo "FATAL: cannot restore $1. Refusing to continue, because the alternative is a"
      echo "silent restart from epoch 0 that then overwrites the checkpoints on GPFS."
      exit 3
    }
    echo "  restored $(basename "$1")"
  }
  restore "${outdir}/checkpoints/latest.pt" "${scratch_run}/checkpoints/"
  newest=$(ls -1 "${outdir}/checkpoints"/checkpoint_epoch*.pt 2>/dev/null | sort -V | tail -1)
  [ -n "$newest" ] && restore "$newest" "${scratch_run}/checkpoints/"
  restore "${outdir}/metrics/step.jsonl" "${scratch_run}/metrics/"
  restore "${outdir}/metrics/epoch.jsonl" "${scratch_run}/metrics/"
fi

sync_back() {
  # `run_metadata.json` and not the directory: the restore above creates $scratch_run, so its
  # existence no longer says the engine wrote anything.
  if [ ! -f "${scratch_run}/run_metadata.json" ]; then
    echo "WARNING: no run_metadata.json at ${scratch_run}, so the engine did not write here." \
         "It writes <output_root>/<run.name>, so run_name=${run_name} disagrees with the" \
         "composed run.name. Present under ${scratch_root}:" \
         "$(ls "$scratch_root" 2>/dev/null | tr '\n' ' ')"
    return 1
  fi
  echo "Syncing ${scratch_run} -> ${outdir}"
  mkdir -p "$outdir"
  rsync -a "${scratch_run}/" "${outdir}/" || echo "WARNING: rsync of the run directory failed"
  rsync -a "${wp_cache}/" "${wp_cache_gpfs}/" || true
}

# Periodic mid-run sync. Checkpoints land in scratch as soon as they are written, but via the
# exit path alone they would only reach GPFS when the job ends -- invisible to monitoring and
# probing for the whole run. A checkpoint caught mid-write transfers truncated and is repaired
# on the next tick, so treat one on GPFS as settled once a later one exists.
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"
( while sleep "$SYNC_INTERVAL"; do sync_back; done ) &
sync_loop_pid=$!

if [ "${ngpus:-1}" -gt 1 ]; then
  echo "Executing wcfm train under torchrun (${ngpus} ranks) ..."
  launcher=(torchrun --standalone --nnodes=1 --nproc_per_node="$ngpus" -m wcfm.cli)
else
  echo "Executing wcfm train ..."
  launcher=(python -u -m wcfm.cli)
fi

# --- preemption ---------------------------------------------------------------
# The trainer is backgrounded and the handler WAITS for it. `PreemptionGuard` writes
# `latest.pt` on SIGTERM at the next step boundary, so a handler that rsynced on receipt would
# copy a checkpoint mid-write and `--resume auto` would then fail on it. Bash runs the trap
# while blocked in `wait`, and a second `wait` on the same pid returns the child's real status.
#
# The signal goes to torchrun rather than to the ranks: torchrun owns rank lifetimes and
# forwards it to its workers itself.
# `run.output_root` -> local scratch (rsynced back by sync_back); `data.cache_dir` -> the
# GPFS path prepared above, because the DirectDataset index is worth keeping between jobs and
# scratch is wiped. Both are appended LAST so they win over anything in conf/ or on the
# submit line. cache_dir has to be one of them: left to the composed config it is whatever
# path that config happens to name, which is not necessarily this user's.
"${launcher[@]}" train "${overrides[@]}" \
  "run.output_root=${scratch_root}" "data.cache_dir=${data_cache}" &
trainer_pid=$!

on_term() {
  echo "SIGTERM received; forwarding to the trainer and waiting for it to checkpoint"
  kill -TERM "$trainer_pid" 2>/dev/null || true
  wait "$trainer_pid"
  rc=$?
  echo "trainer exited rc=${rc}; latest.pt is settled, syncing"
  kill "$sync_loop_pid" 2>/dev/null || true
  sync_back
  exit 143
}
trap on_term SIGTERM

wait "$trainer_pid"
rc=$?
kill "$sync_loop_pid" 2>/dev/null || true
sync_back
synced=$?

if [ "$rc" -ne 0 ]; then
  echo "Training FAILED rc=${rc}"
elif [ "$synced" -ne 0 ]; then
  # The trainer succeeded and the outputs are gone, which is worse than a crash: it is a
  # green job with nothing to show. Do not report it as a success.
  echo "Training FAILED: the trainer exited 0 but produced nothing at ${scratch_run}"
  rc=4
else
  echo "Training complete!"
fi
exit "$rc"
