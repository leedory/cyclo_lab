#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Play the recorded 15 FPS source timeline at wall-clock 5x (75 source FPS).
exec "${SCRIPT_DIR}/run_task000525_demo.sh" --speed 5 "$@"
