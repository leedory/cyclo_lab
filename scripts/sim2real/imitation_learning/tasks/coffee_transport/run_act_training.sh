#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CYCLO_LAB_ROOT=${CYCLO_LAB_ROOT:-$(realpath "${SCRIPT_DIR}/../../../../..")}
CYCLO_S2R_ROOT=${CYCLO_S2R_ROOT:-$(realpath -m "${CYCLO_LAB_ROOT}/../cyclo_intelligence_s2r")}
LEROBOT_CONTAINER=${LEROBOT_CONTAINER:-lerobot_server_s2r}

if (( $# != 2 )); then
  echo "usage: $0 TASK_ID CAMPAIGN_ROOT" >&2
  exit 2
fi
TASK_ID=$1
CAMPAIGN=$(realpath "$2")
[[ "${TASK_ID}" == 000001 || "${TASK_ID}" == 000002 ]] || {
  echo "TASK_ID must be 000001 or 000002" >&2; exit 2;
}

[[ -d "${CYCLO_S2R_ROOT}" ]] || {
  echo "missing Cyclo Intelligence S2R root: ${CYCLO_S2R_ROOT}" >&2; exit 2;
}
ARTIFACT_TAG=task_${TASK_ID}_coffee_transport_headcam
if [[ "${TASK_ID}" == 000002 ]]; then
  # Keep new 19D artifacts separate from the legacy lateral-only dataset/model.
  ARTIFACT_TAG=${ARTIFACT_TAG}_full_base
fi
DATASET=${CYCLO_S2R_ROOT}/docker/workspace/dataset/robotis/${ARTIFACT_TAG}_v30
MODEL=${CYCLO_S2R_ROOT}/docker/workspace/model/lerobot/${ARTIFACT_TAG}_act
RECIPE=${CYCLO_S2R_ROOT}/training/task_${TASK_ID}/act.yaml
OUT=${CAMPAIGN}/${ARTIFACT_TAG}
APPROVAL_REQUIREMENT=${CAMPAIGN}/manifests/task_${TASK_ID}_act_training_approval_required.json
APPROVAL=${CAMPAIGN}/manifests/task_${TASK_ID}_act_training_approval.json
mkdir -p "${OUT}/logs"

# Campaigns opt in by creating a requirement marker. Approval binds the exact
# completed dataset metadata so it cannot authorize a replacement in place.
if [[ -e "${APPROVAL_REQUIREMENT}" ]]; then
  [[ -f "${APPROVAL_REQUIREMENT}" && ! -L "${APPROVAL_REQUIREMENT}" ]] || {
    echo "ACT approval requirement must be a regular file: ${APPROVAL_REQUIREMENT}" >&2
    exit 1
  }
  if [[ ! -f "${APPROVAL}" || -L "${APPROVAL}" ]]; then
    echo "ACT_APPROVAL_PENDING task=${TASK_ID} required=${APPROVAL_REQUIREMENT} approval=${APPROVAL}" >&2
    exit 42
  fi
  [[ -f "${DATASET}/meta/info.json" ]] || {
    echo "approved ACT dataset is incomplete: ${DATASET}" >&2
    exit 1
  }
  python3 - "${APPROVAL_REQUIREMENT}" "${APPROVAL}" "${CAMPAIGN}" \
    "${TASK_ID}" "${DATASET}" "${MODEL}" <<'PY'
import hashlib
import json
import pathlib
import sys

requirement_path, approval_path = map(pathlib.Path, sys.argv[1:3])
campaign, task_id = sys.argv[3:5]
dataset, model = map(pathlib.Path, sys.argv[5:7])
requirement = json.loads(requirement_path.read_text())
expected_requirement = {
    "schema": "cyclo.coffee_transport.act_training_approval_required.v1",
    "required": True,
    "task_id": task_id,
    "campaign_root": campaign,
    "dataset_path": str(dataset),
    "model_path": str(model),
    "approval_file": str(approval_path),
}
assert all(requirement.get(key) == value for key, value in expected_requirement.items())

approval = json.loads(approval_path.read_text())
expected_approval = {
    "schema": "cyclo.coffee_transport.act_training_approval.v1",
    "approved": True,
    "approval_scope": "start_or_resume_act_training",
    "task_id": task_id,
    "campaign_root": campaign,
    "dataset_path": str(dataset),
    "model_path": str(model),
}
assert all(approval.get(key) == value for key, value in expected_approval.items())

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert approval.get("dataset_info_sha256") == sha256(dataset / "meta/info.json")
assert approval.get("runtime_rate_provenance_sha256") == sha256(
    dataset / "meta/physical_runtime_rate_provenance.json"
)
assert isinstance(approval.get("approved_utc"), str) and approval["approved_utc"].endswith("Z")
assert approval.get("approved_by") == "user"
PY
fi

[[ -f "${DATASET}/meta/info.json" ]] || { echo "missing completed dataset: ${DATASET}" >&2; exit 1; }

if ! docker inspect --format '{{.State.Running}}' "${LEROBOT_CONTAINER}" 2>/dev/null | grep -qx true; then
  bash "${CYCLO_S2R_ROOT}/docker/container.sh" start-lerobot
fi
(
  cd "${CYCLO_S2R_ROOT}"
  python3 tools/training/act_recipe.py validate "${RECIPE}" \
    --container "${LEROBOT_CONTAINER}"
  python3 tools/training/act_recipe.py plan "${RECIPE}" \
    --container "${LEROBOT_CONTAINER}"
  python3 tools/training/act_recipe.py run "${RECIPE}" \
    --container "${LEROBOT_CONTAINER}"
) 2>&1 | tee -a "${OUT}/logs/act_training.log"

[[ -f "${MODEL}/cyclo_policy.yaml" ]] || {
  echo "training completed without cyclo_policy.yaml: ${MODEL}" >&2; exit 1;
}
if [[ -L "${OUT}/act_policy" ]]; then
  [[ "$(readlink -f "${OUT}/act_policy")" == "$(readlink -f "${MODEL}")" ]] || {
    echo "refusing mismatched policy symlink: ${OUT}/act_policy" >&2; exit 1;
  }
elif [[ -e "${OUT}/act_policy" ]]; then
  echo "refusing existing non-symlink: ${OUT}/act_policy" >&2; exit 1
else
  ln -s "${MODEL}" "${OUT}/act_policy"
fi
echo "COFFEE_ACT_COMPLETE task=${TASK_ID} policy=${MODEL}"
