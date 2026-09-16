#!/bin/bash
#
# Builds the one venv `wcfm` runs from: the pinned GPU stack and the framework dependencies in
# the same place. `wcfm env-check` prints what it produced. Run it from the checkout root.
#
#   gridutils/build_env.sh                 build (or top up) $WCFM_PYENV
#   gridutils/build_env.sh --flash-attn    build the flash-attn wheel; needs a GPU node
#
# uv skips what is already installed at the pinned version, so re-running after a
# failure is harmless. 
#
# Every version this environment pins is in the block below and nowhere else.

set -uo pipefail

PYTHON_VERSION=3.11
TORCH=2.10.0
TORCH_MM=2.10          # major.minor, as warpconvnet spells it in its wheel names
CUDA=cu128
TORCHVISION=0.25.0
WARPCONVNET=1.7.8
TORCH_SCATTER=2.1.2
FLASH_ATTN=2.8.3

USER_NAME="${USER:-$(id -un)}"
PYENV="${WCFM_PYENV:-/gpfs01/lbne/users/fm/${USER_NAME}/uvenv}"

# flash-attn publishes no wheel for torch 2.10 (its newest builds stop at 2.8)
# It is built once and shared, but you can also build it yourself on a GPU node.
# See the --flash-attn section below.
WHEELHOUSE="${WCFM_WHEELHOUSE:-/gpfs01/lbne/users/fm/shared/wheels}"
UV="${UV:-$(command -v uv || echo "${HOME}/.local/bin/uv")}"

WARP_WHEEL_URL="https://github.com/NVlabs/WarpConvNet/releases/download/v${WARPCONVNET}/warpconvnet-${WARPCONVNET}+torch${TORCH_MM}${CUDA}-cp311-cp311-linux_x86_64.whl"

# GPFS. The default hardlink mode fails across the cache/venv boundary here.
export UV_LINK_MODE=copy

[ -x "$UV" ] || { echo "FATAL: no uv at ${UV}. Install it, or set UV=/path/to/uv."; exit 3; }

# Writes progress to stderr: stdout is the resolved path the caller captures.
#   $1 glob naming the wheel   $2 upstream URL, or "" when there is none
stage_wheel() {
  local pattern="$1" url="$2" found
  found=$(ls "${WHEELHOUSE}"/${pattern} 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    echo "   staged: $(basename "$found")" >&2
    echo "$found"
    return 0
  fi
  [ -n "$url" ] || return 1
  mkdir -p "$WHEELHOUSE" || return 1
  echo "   fetching $(basename "$url") -> ${WHEELHOUSE}" >&2
  # To a temp name and renamed, so a second build never sees a half-written wheel.
  local dest="${WHEELHOUSE}/$(basename "$url")"
  curl -fsSL -o "${dest}.part" "$url" || { rm -f "${dest}.part"; return 1; }
  mv "${dest}.part" "$dest" || return 1
  echo "$dest"
}

# --- flash-attn wheel build (GPU node) ---------------------------------------
# Submitted by gridutils/build_flash_attn.sub. Builds nothing else: the venv it compiles
# against must already carry torch, which the normal path installs in step 2.
if [ "${1:-}" = "--flash-attn" ]; then
  [ -d "$PYENV" ] || { echo "FATAL: no venv at ${PYENV}. Run build_env.sh first."; exit 3; }
  mkdir -p "$WHEELHOUSE" || { echo "FATAL: cannot write ${WHEELHOUSE}"; exit 3; }
  echo "building flash-attn ${FLASH_ATTN} against ${PYENV} -> ${WHEELHOUSE}"

  # Only this arch is compiled. 
  export TORCH_CUDA_ARCH_LIST="8.9"
  export MAX_JOBS=4          # nvcc parallelism; the full build is memory-hungry

  "$UV" pip install --python "${PYENV}/bin/python" pip wheel setuptools ninja psutil || {
    echo "FATAL: cannot stage the build tools"; exit 3; }

  # --no-build-isolation so it compiles against the torch already in the venv 
  # --no-deps so pip builds flash-attn and not wheels for torch as well.
  "${PYENV}/bin/python" -m pip wheel "flash-attn==${FLASH_ATTN}" \
      --no-build-isolation --no-deps --wheel-dir "$WHEELHOUSE"
  rc=$?
  "$UV" pip uninstall --python "${PYENV}/bin/python" pip >/dev/null 2>&1
  [ "$rc" -eq 0 ] || { echo "FATAL: flash-attn build failed (rc=${rc})"; exit 3; }
  echo "done: $(ls "${WHEELHOUSE}"/flash_attn-*.whl)"
  exit 0
fi

[ -f pyproject.toml ] && [ -d wcfm ] || {
  echo "FATAL: run this from the checkout root (no pyproject.toml + wcfm/ here)."; exit 3; }

echo "venv:       ${PYENV}"
echo "wheelhouse: ${WHEELHOUSE}"
echo "uv:         ${UV}"
echo

# --- 1. the venv -------------------------------------------------------------
if [ ! -x "${PYENV}/bin/python" ]; then
  echo "== creating venv (python ${PYTHON_VERSION})"
  "$UV" venv --python "$PYTHON_VERSION" "$PYENV" || exit 3
fi
PY="${PYENV}/bin/python"

# --- 2. torch, first, so everything after resolves against a satisfied pin ----
echo "== torch ${TORCH}+${CUDA}"
"$UV" pip install --python "$PY" \
    "torch==${TORCH}" "torchvision==${TORCHVISION}" \
    --index-url "https://download.pytorch.org/whl/${CUDA}" || exit 3

# --- 3. torch-scatter --------------------------------------------------------
echo "== torch-scatter ${TORCH_SCATTER}"
"$UV" pip install --python "$PY" "torch-scatter==${TORCH_SCATTER}" \
    --find-links "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html" || exit 3

# --- 4. warpconvnet ----------------------------------------------------------
# Published as a wheel per (version, torch, CUDA, python) since v1.7.7. 
echo "== warpconvnet ${WARPCONVNET}"
warp_wheel=$(stage_wheel "warpconvnet-${WARPCONVNET}+torch${TORCH_MM}${CUDA}-*.whl" "$WARP_WHEEL_URL") || {
  echo "FATAL: cannot stage warpconvnet ${WARPCONVNET} into ${WHEELHOUSE} from"
  echo "  ${WARP_WHEEL_URL}"
  exit 3
}
"$UV" pip install --python "$PY" "$warp_wheel" || exit 3

# --- 5. flash-attn, from the shared wheelhouse -------------------------------
echo "== flash-attn ${FLASH_ATTN}"
flash_wheel=$(stage_wheel "flash_attn-${FLASH_ATTN}*.whl" "")
if [ -z "$flash_wheel" ]; then
  echo
  echo "FATAL: no flash_attn-${FLASH_ATTN}*.whl in ${WHEELHOUSE}."
  echo
  echo "flash-attn publishes no wheel for torch ${TORCH}, so it has to be compiled once on a"
  echo "GPU node. torch is installed now, which is all that build needs:"
  echo
  echo "  condor_submit gridutils/build_flash_attn.sub"
  echo
  echo "Then re-run this script; it will skip everything already done."
  echo
  exit 3
fi
"$UV" pip install --python "$PY" "$flash_wheel" || exit 3

# --- 6. the framework, and this checkout --------------------------------------
# hydra, omegaconf, lightning-fabric, pytest and ruff come from pyproject.toml.
# It also writes wire_cell_fm.egg-info/, which every `wcfm submit` requires (jobpack.REQUIRED)
echo "== framework dependencies + this checkout (editable)"

install_editable() { "$UV" pip install --python "$PY" -e '.[dev,analysis]'; }
if ! install_editable; then
  echo "   editable install failed; clearing build/ and retrying once (NFS cleanup race)"
  rm -rf build
  sleep 5
  install_editable || exit 3
fi

# --- 7. check and fail if wrong -------------------------------
# `wcfm env-check` does not hard fail, so we do it here.
"$PY" -c "
import torch, torch_scatter  # noqa: F401  -- the import IS the ABI check
want = '${TORCH}+${CUDA}'
assert torch.__version__ == want, f'torch is {torch.__version__}, expected {want}'
" || { echo "FATAL: torch is not the pinned version, or torch_scatter cannot load against it"; exit 3; }

echo
"${PYENV}/bin/wcfm" env-check
