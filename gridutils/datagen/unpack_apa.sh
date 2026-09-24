#!/bin/bash
#
# Extract one APA's pixel file and the metadata file from a directory of production tgz
# archives, into the bucketed tree `DirectDataset` scans:
#
#   <dest>/<run>/<bucket>/<archive stem>/<event>_pixeldata-anode<apa>.h5
#   <dest>/<run>/<bucket>/<archive stem>/<event>_metadata.h5
#
# `bucket_of` reads run and sequence from the archive name `out_monte-carlo-<run>-<seq>_...tgz`;
# the bucket is seq // 1000, one-based and zero-padded, so no directory holds more than ~1000
# entries. An archive named otherwise is recorded in failed.txt and skipped.
#
# Usage:
#   bash gridutils/datagen/unpack_apa.sh <src_dir> <dest_dir> [apa=0]
#
# <dest_dir> may be <src_dir> itself: the archives stay at the top level and the extracted
# tree sits beside them under <run>/, which is the numu layout. `DirectDataset` scans for
# `*anode<apa>.h5` only, so the archives are invisible to it either way.
#
# PARALLEL (default 8) archives are extracted at once. Archives whose two files already exist
# under <dest> are skipped, so a killed run is rerun as is. Every failed archive is appended to
# <dest>/failed.txt and the run continues; a non-empty file at the end means some events are
# missing, and those archives are retried by rerunning.

set -uo pipefail

src=${1:?src_dir}
dest=${2:?dest_dir}
apa=${3:-0}
PARALLEL=${PARALLEL:-8}

mkdir -p "$dest"
: > "$dest/failed.txt"

bucket_of() {
  local stem=$1 run seq
  read -r run seq < <(sed -En 's/^out_monte-carlo-([0-9]+)-([0-9]+)_.*/\1 \2/p' <<< "$stem")
  [ -n "$seq" ] || return 1
  printf '%s/%03d' "$run" $((10#$seq / 1000 + 1))
}

extract_one() {
  local f=$1 stem bucket dir
  stem=$(basename "$f" .tgz)
  bucket=$(bucket_of "$stem") || { echo "unparseable name: $f" >&2; echo "$f" >> "$dest/failed.txt"; return; }
  dir="$dest/$bucket/$stem"
  if [ -f "$dir/${stem#out_}_pixeldata-anode${apa}.h5" ] && [ -f "$dir/${stem#out_}_metadata.h5" ]; then
    return
  fi
  mkdir -p "$dir"
  if ! tar xzf "$f" -C "$dir" --strip-components=1 \
        --wildcards "*_pixeldata-anode${apa}.h5" "*_metadata.h5" 2>>"$dest/tar_errors.log"; then
    echo "$f" >> "$dest/failed.txt"
    rm -rf "$dir"
  fi
}
export -f bucket_of extract_one
export dest apa

# find, not a glob: 100k archives overflow the argument list.
n=$(find "$src" -maxdepth 1 -name '*.tgz' | wc -l)
echo "unpacking $n archives from $src -> $dest (apa $apa, $PARALLEL parallel)"
find "$src" -maxdepth 1 -name '*.tgz' -print0 \
  | xargs -0 -P "$PARALLEL" -I{} bash -c 'extract_one "$1"' _ {}

n_out=$(find "$dest" -mindepth 3 -maxdepth 3 -type d | wc -l)
n_fail=$(wc -l < "$dest/failed.txt")
echo "done: $n_out event directories, $n_fail failed (see $dest/failed.txt)"
