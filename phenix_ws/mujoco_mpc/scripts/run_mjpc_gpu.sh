#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"${PROJECT_ROOT}/build"}
MJPC_BIN="${BUILD_DIR}/bin/mjpc"

if [[ ! -x "${MJPC_BIN}" ]]; then
  echo "[ERROR] MuJoCo MPC binary not found at ${MJPC_BIN}." >&2
  echo "        Build the project first: cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . --target mjpc" >&2
  exit 1
fi

export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"

# Optional hint for telemetry
if command -v glxinfo >/dev/null 2>&1; then
  renderer=$(GALLIUM_DRIVER="${GALLIUM_DRIVER}" glxinfo | grep -m1 "OpenGL renderer") || true
  [[ -n "${renderer:-}" ]] && echo "[INFO] Using renderer: ${renderer}" >&2
fi

echo "[INFO] Launching mjpc with GPU acceleration (GALLIUM_DRIVER=${GALLIUM_DRIVER})." >&2
exec "${MJPC_BIN}" "$@"
