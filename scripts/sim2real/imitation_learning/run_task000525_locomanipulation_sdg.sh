#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
TASK_GENERATOR="${SCRIPT_DIR}/tasks/task_000525/generate_trajectories.py"
TASK_ID="Cyclo-Real-Showroom-Task000525-Locomanipulation-SDG-FFW-SG2-v0"
ISAAC_PYTHON_BIN="${ISAAC_PYTHON_BIN:-/isaac-sim/python.sh}"
KIT_ARGS="--ext-folder ${REPO_ROOT}/source --enable cyclo_lab"

exec "${ISAAC_PYTHON_BIN}" "${TASK_GENERATOR}" \
  "$@" \
  --task "${TASK_ID}" \
  --active_side right \
  --navigation_mode fixed_yaw_holonomic \
  --navigation_yaw 0.0 \
  --approach_distance 0.0 \
  --linear_max 0.10 \
  --angular_max 0.25 \
  --following_offset 0.20 \
  --distance_threshold 0.05 \
  --randomize_placement "" \
  --kit_args "${KIT_ARGS}"
