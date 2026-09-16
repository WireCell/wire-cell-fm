#!/bin/bash
#
# Tabulate a run's probe JSONs. CPU, one core, seconds.
#
# Args (positional):
#   $1 archive  -- basename of the transferred repo tarball (unpacked into scratch)
#   $2 pyenv    -- the venv built by gridutils/build_env.sh (the whole stack)
#   $3 dir      -- the run's probes/ directory
#
# This node is NOT allowed to fail the DAG (see wcfm/eval/dag.py): the table is a view over the
# JSONs and rebuilds in seconds, so failing a campaign's worth of completed probe jobs over it
# would be the wrong trade. The DAG wraps it in a POST script that always exits 0; this script
# still returns an honest status so the log says what happened.
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
dir=$3

echo "Running ${CLUSTER_ID:-?}.${JOB_ID:-?} on $(hostname)"
echo "  probes dir=${dir}"

export PYTHONUNBUFFERED=1
source "${pyenv}/bin/activate"
# Everything is in the venv; the repo is the only thing PYTHONPATH adds, and it is the archive
# this job unpacked -- see trainjob.sh.
export PYTHONPATH="${repodir}${PYTHONPATH:+:$PYTHONPATH}"
cd "${_CONDOR_SCRATCH_DIR:-/tmp}"

python - "$repodir" <<'PY' || { echo "FATAL: wrong environment"; exit 3; }
import os, sys, wcfm
repodir = os.path.realpath(sys.argv[1])
assert os.path.realpath(wcfm.__file__).startswith(repodir), \
    f"wcfm came from {wcfm.__file__}, not the unpacked archive at {repodir}"
PY

shopt -s nullglob
files=("${dir}"/*.json)
if [ ${#files[@]} -eq 0 ]; then
  echo "no probe JSONs in ${dir} -- nothing to merge"
  exit 2
fi

python -u -m wcfm.cli eval compare "${files[@]}" \
    --csv="${dir}/table.csv" --markdown | tee "${dir}/table.md"
rc=${PIPESTATUS[0]}
echo "merge rc=${rc}"
exit "$rc"
