#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../../.." && pwd)"
DEMO_SCRIPT="${SCRIPT_DIR}/task000525_visual_demo.py"
ISAAC_PYTHON_BIN="${ISAAC_PYTHON_BIN:-/isaac-sim/python.sh}"
KIT_ARGS="--ext-folder ${REPO_ROOT}/source --enable cyclo_lab"

if [[ ! -x "${ISAAC_PYTHON_BIN}" ]]; then
  echo "[ERROR] Isaac Sim Python was not found at ${ISAAC_PYTHON_BIN}." >&2
  echo "[ERROR] Enter the cyclo_lab container first: ./docker/container.sh enter" >&2
  exit 2
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "[ERROR] DISPLAY is unset, so the Isaac Sim GUI cannot open." >&2
  echo "[ERROR] Start/recreate cyclo_lab with X11 enabled, then enter the container." >&2
  exit 2
fi

# No --headless flag is supplied here. Extra CLI arguments can override the
# input file, source-frame speed, playback rate, device, or 16-env default.
exec "${ISAAC_PYTHON_BIN}" "${DEMO_SCRIPT}" \
  "$@" \
  --kit_args "${KIT_ARGS}"
