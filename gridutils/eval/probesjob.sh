#!/bin/bash
#
# The probe suite on a Condor worker. CPU only -- that is the point of splitting it from
# extraction: the GPU pass happens once per checkpoint while every metric stays on a CPU slot,
# so a queue with one free GPU still makes progress on a whole campaign.
#
# Args (positional):
#   $1 archive  -- basename of the transferred repo tarball (unpacked into scratch)
#   $2 pyenv    -- the cluster uv venv
#   $3 store    -- ONE checkpoint's feature store: <run>/features/epoch<N>
#   $4 outdir   -- where the probe JSONs go (one directory per RUN, not per epoch)
#   $5 stages   -- comma-separated stage list
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
store=$3
outdir=$4
stages=$5

echo "Running ${CLUSTER_ID:-?}.${JOB_ID:-?} on $(hostname)"
echo "  store=${store}"
echo "  outdir=${outdir}"
echo "  stages=${stages}"
echo ""

export PYTHONUNBUFFERED=1
source "${pyenv}/bin/activate"

# Nothing is installed by this job -- see trainjob.sh for the three ways that goes wrong.
WCFM_LIBS="${WCFM_LIBS:-/gpfs01/lbne/users/fm/${USER}/wcfm-libs}"
for pkg in hydra omegaconf; do
  if [ ! -d "${WCFM_LIBS}/${pkg}" ]; then
    echo "FATAL: no ${pkg} at ${WCFM_LIBS}"
    exit 3
  fi
done
export PYTHONPATH="${WCFM_LIBS}:${repodir}${PYTHONPATH:+:$PYTHONPATH}"

# Stay in scratch, which holds `repo/` and the archive but nothing importable -- see the
# unpack block at the top.
cd "${_CONDOR_SCRATCH_DIR:-/tmp}"

if [ ! -f "${store}/provenance.json" ]; then
  echo "FATAL: no provenance.json in ${store} -- extraction did not produce this store"
  exit 4
fi

# Written straight to GPFS rather than to scratch and rsynced, unlike extraction: probe JSONs
# are kilobytes and `write_json` is already atomic (temp file in the same directory, renamed
# over), so there is nothing for a sync step to protect against and a job killed mid-suite
# leaves every completed stage behind rather than none of them.
mkdir -p "${outdir}"
python -u -m wcfm.cli eval probe "${store}" --stages="${stages}" --out-dir="${outdir}" --device=cpu
rc=$?

n_json=$(find "${outdir}" -name '*.json' 2>/dev/null | wc -l)
echo "probe JSONs in ${outdir}: ${n_json}"
if [ "$rc" -eq 0 ] && [ "$n_json" -eq 0 ]; then
  echo "FAILED: probes exited 0 but wrote nothing"
  rc=4
fi

if [ "$rc" -ne 0 ]; then
  echo "Probes FAILED rc=${rc}"
else
  echo "Probes complete!"
fi
exit "$rc"
