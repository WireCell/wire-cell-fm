#!/bin/bash
#
# `wcfm eval extract` on a Condor worker.
#
# Args (positional):
#   $1 archive   -- basename of the transferred repo tarball (unpacked into scratch)
#   $2 pyenv     -- the venv built by gridutils/build_env.sh (the whole stack)
#   $3 rundir    -- the RUN directory on GPFS: checkpoints/, config.yaml, run_metadata.json
#   $4 outdir    -- where the feature stores go on GPFS
#   $5 cache_dir -- base for the warpconvnet benchmark cache and the data index
#   $6+ ...      -- flags passed through to `wcfm eval extract`
#
# The extraction needs a GPU: warpconvnet's sparse convolution asserts `coords.is_cuda` in its
# hash table, so the login node can read the shards and build the module but cannot run a
# forward. That is why this exists rather than a login-node loop.
#
# Same I/O strategy as trainjob.sh -- write to $_CONDOR_SCRATCH_DIR and rsync to GPFS -- for a
# reason specific to this step: features are written per checkpoint and a job that dies on the
# fourth of five epochs should leave the first three behind, complete and readable.
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
rundir=$3
outdir=$4
cache_dir=$5
shift 5
flags=("$@")

echo "Running ${CLUSTER_ID:-?}.${JOB_ID:-?} on $(hostname)"
echo "  repodir=${repodir}"
echo "  rundir=${rundir}"
echo "  outdir=${outdir}"
echo "  flags=${flags[*]}"
echo ""

wp_cache_gpfs="${cache_dir}/warpconvnet"
wp_cache="${_CONDOR_SCRATCH_DIR}/warpconvnet"
mkdir -p "$wp_cache_gpfs" "${cache_dir}/data"
cp -r "$wp_cache_gpfs" "$wp_cache" 2>/dev/null || mkdir -p "$wp_cache"
export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"
export PYTHONUNBUFFERED=1

source "${pyenv}/bin/activate"

# Nothing is installed by this job -- see trainjob.sh. Everything is in the venv; the repo is
# the only thing PYTHONPATH adds, and it is the archive this job unpacked.
export PYTHONPATH="${repodir}${PYTHONPATH:+:$PYTHONPATH}"

# Stay in scratch, which holds `repo/` and the archive but nothing importable -- see the
# unpack block at the top.
cd "${_CONDOR_SCRATCH_DIR:-/tmp}"

# torch from the venv handed in as $2, `wcfm` from the archive this job unpacked -- see
# trainjob.sh for why both, and why realpath.
python - "$pyenv" "$repodir" <<'PY' || { echo "FATAL: wrong environment"; exit 3; }
import os, sys, torch, wcfm
pyenv, repodir = (os.path.realpath(p) for p in sys.argv[1:3])
assert os.path.realpath(torch.__file__).startswith(pyenv), \
    f"torch came from {torch.__file__}, not the venv at {pyenv}"
assert os.path.realpath(wcfm.__file__).startswith(repodir), \
    f"wcfm came from {wcfm.__file__}, not the unpacked archive at {repodir}"
print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} "
      f"devices {torch.cuda.device_count()}")
assert torch.cuda.is_available(), "extraction needs a GPU: warpconvnet's sparse conv is CUDA-only"
PY

scratch_out="${_CONDOR_SCRATCH_DIR}/features"
mkdir -p "$scratch_out"

python -u -m wcfm.cli eval extract "$rundir" --out-root="$scratch_out" "${flags[@]}"
rc=$?

# A job that exits 0 having written nothing is the failure mode this cluster produces most
# often (cluster 2263 did exactly that for training), so the outputs are counted, not assumed.
n_prov=$(find "$scratch_out" -name provenance.json 2>/dev/null | wc -l)
echo "wrote ${n_prov} provenance.json under ${scratch_out}"
if [ "$rc" -eq 0 ] && [ "$n_prov" -eq 0 ]; then
  echo "FAILED: extract exited 0 but wrote no provenance.json -- nothing was produced"
  rc=4
fi

echo "Syncing ${scratch_out} -> ${outdir}"
mkdir -p "$outdir"
rsync -a "${scratch_out}/" "${outdir}/" || { echo "FAILED: rsync"; rc=5; }
rsync -a "${wp_cache}/" "${wp_cache_gpfs}/" || true

if [ "$rc" -ne 0 ]; then
  echo "Extraction FAILED rc=${rc}"
else
  echo "Extraction complete!"
  find "$outdir" -name provenance.json -exec echo "  {}" \;
fi
exit "$rc"
