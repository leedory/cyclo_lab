#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <000001|000002> [generator options]" >&2
  exit 2
fi

TASK_ID="$1"
shift
case "${TASK_ID}" in
  000001|000002) ;;
  *) echo "Task id must be 000001 or 000002" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

exec "${ISAAC_PYTHON_BIN:-/isaac-sim/python.sh}" \
  "${SCRIPT_DIR}/tasks/coffee_transport/generate_trajectories.py" \
  --task-id "${TASK_ID}" \
  --kit_args "--ext-folder ${REPO_ROOT}/source --enable cyclo_lab" \
  "$@"
