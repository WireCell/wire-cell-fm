#!/bin/bash
#
# A dataset-building module on a CPU worker. Submitted by `wcfm datagen`.
#
# Args (positional):
#   $1 archive  -- basename of the transferred repo tarball (unpacked into scratch)
#   $2 pyenv    -- the venv built by gridutils/build_env.sh
#   $3 module   -- dotted module to run, e.g. wcfm.data.prep.create_shards
#   $4+ ...     -- its arguments, passed through unchanged
#
# The module writes its shards or pack straight to the GPFS paths in its arguments; nothing is
# rsynced from scratch. `create_shards` skips shard files that exist, so a killed job is
# resubmitted as is.
#
# NB: this file is a COPY, staged beside the archive at submit time, and Condor transfers it
# here like any other input. Editing the checkout's copy cannot reach a queued or running job.

set -uo pipefail

# The archive is unpacked into `repo/` rather than over the scratch root, so the cwd holds
# nothing importable: `python -m` puts the cwd first on sys.path, ahead of PYTHONPATH.
repo_archive=$(readlink -f "$1")
repodir="${_CONDOR_SCRATCH_DIR:-$PWD}/repo"
mkdir -p "$repodir"
tar xzf "$repo_archive" -C "$repodir" || { echo "FATAL: cannot unpack ${repo_archive}"; exit 3; }
[ -d "${repodir}/wcfm" ] || { echo "FATAL: ${repo_archive} has no wcfm/"; exit 3; }
pyenv=$2
module=$3
shift 3

echo "wcfm datagen on $(hostname): python -m ${module} $*"
echo "  _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR:-unset}"

# `wcfm.data.voxels` imports warpconvnet, whose benchmark cache defaults to ~/.cache on NFS home
# and takes a flock there. Nothing in a CPU job is worth keeping, so it goes to scratch.
export WARPCONVNET_BENCHMARK_CACHE_DIR="${_CONDOR_SCRATCH_DIR:-$PWD}/warpconvnet"
# `create_shards.write_shard` builds each shard under `tempfile.gettempdir()` and copies it to
# GPFS in one pass; that directory has to be the node-local scratch, not a shared filesystem.
export TMPDIR="${_CONDOR_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
export PYTHONUNBUFFERED=1

source "${pyenv}/bin/activate"
export PYTHONPATH="${repodir}${PYTHONPATH:+:$PYTHONPATH}"

# torch and h5py have to come from the venv and wcfm from the archive, or the job builds
# shards with code other than the tree that was submitted.
python - "$pyenv" "$repodir" <<'PY' || { echo "FATAL: wrong environment"; exit 3; }
import os, sys
import h5py, torch, wcfm
pyenv, repodir = (os.path.realpath(p) for p in sys.argv[1:3])
assert os.path.realpath(torch.__file__).startswith(pyenv), \
    f"torch came from {torch.__file__}, not the venv at {pyenv}"
assert os.path.realpath(wcfm.__file__).startswith(repodir), \
    f"wcfm came from {wcfm.__file__}, not the unpacked archive at {repodir}"
print(f"torch {torch.__version__} | h5py {h5py.__version__} | cpus {os.cpu_count()}")
PY

cd "${repodir}" || exit 3
python -m "${module}" "$@"
rc=$?
echo "python -m ${module} exited ${rc}"
exit $rc
