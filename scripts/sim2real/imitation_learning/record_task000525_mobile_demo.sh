#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RECORDER="${SCRIPT_DIR}/recorder/record_demos.py"
TASK_ID="Cyclo-Real-Showroom-Task000525-FFW-SG2-v0"
ISAAC_PYTHON_BIN="${ISAAC_PYTHON_BIN:-/isaac-sim/python.sh}"
KIT_ARGS="--ext-folder ${REPO_ROOT}/source --enable cyclo_lab"

exec "${ISAAC_PYTHON_BIN}" "${RECORDER}" \
  "$@" \
  --task "${TASK_ID}" \
  --robot_type FFW_SG2 \
  --task525_phase_markers \
  --task525_base_mode dijkstra \
  --kit_args "${KIT_ARGS}"
