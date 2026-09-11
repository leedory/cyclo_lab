#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <000001|000002> [record_demos.py options]" >&2
  exit 2
fi

TASK_NUMBER="$1"
shift
case "${TASK_NUMBER}" in
  000001|000002) ;;
  *)
    echo "Task number must be 000001 or 000002, got: ${TASK_NUMBER}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RECORDER="${SCRIPT_DIR}/recorder/record_demos.py"
TASK_ID="Cyclo-Real-Showroom-Task${TASK_NUMBER}-FFW-SG2-v0"
ISAAC_PYTHON_BIN="${ISAAC_PYTHON_BIN:-/isaac-sim/python.sh}"
KIT_ARGS="--ext-folder ${REPO_ROOT}/source --enable cyclo_lab"
SEED_VERSION="v4"
if [[ "${TASK_NUMBER}" == "000002" ]]; then
  SEED_VERSION="v3"
fi
DATASET_FILE="${REPO_ROOT}/datasets/task_${TASK_NUMBER}_orange_transport_two_can_${SEED_VERSION}_seed.hdf5"

exec "${ISAAC_PYTHON_BIN}" "${RECORDER}" \
  --dataset_file "${DATASET_FILE}" \
  --num_demos 1 \
  "$@" \
  --headless \
  --enable_cameras \
  --camera_view operator \
  --task "${TASK_ID}" \
  --robot_type FFW_SG2 \
  --coffee_transport_seed_automation \
  --kit_args "${KIT_ARGS}"
