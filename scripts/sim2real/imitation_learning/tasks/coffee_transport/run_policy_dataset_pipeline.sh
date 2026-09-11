#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
CYCLO_LAB_ROOT=${CYCLO_LAB_ROOT:-$(realpath "${SCRIPT_DIR}/../../../../..")}
CYCLO_LAB_CONTAINER_ROOT=${CYCLO_LAB_CONTAINER_ROOT:-/workspace/cyclo_lab}
CYCLO_LAB_CONTAINER=${CYCLO_LAB_CONTAINER:-cyclo_lab}
CYCLO_S2R_ROOT=${CYCLO_S2R_ROOT:-$(realpath -m "${CYCLO_LAB_ROOT}/../cyclo_intelligence_s2r")}
CYCLO_S2R_CONTAINER=${CYCLO_S2R_CONTAINER:-cyclo_intelligence_s2r}
CYCLO_SOURCE_ROOT_IN_S2R_CONTAINER=${CYCLO_SOURCE_ROOT_IN_S2R_CONTAINER:-/root/ros2_ws/src/cyclo_intelligence}

if (( $# != 3 )); then
  echo "usage: $0 TASK_ID PHYSICAL_50_HDF CAMPAIGN_ROOT" >&2
  exit 2
fi

TASK_ID=$1
PHYSICAL_HOST=$(realpath "$2")
CAMPAIGN=$(realpath "$3")
[[ "${TASK_ID}" == 000001 || "${TASK_ID}" == 000002 ]] || {
  echo "TASK_ID must be 000001 or 000002" >&2; exit 2;
}
[[ -f "${PHYSICAL_HOST}" ]] || { echo "missing physical HDF: ${PHYSICAL_HOST}" >&2; exit 2; }
[[ -d "${CYCLO_LAB_ROOT}" ]] || {
  echo "missing Cyclo Lab root: ${CYCLO_LAB_ROOT}" >&2; exit 2;
}
[[ -d "${CYCLO_S2R_ROOT}" ]] || {
  echo "missing Cyclo Intelligence S2R root: ${CYCLO_S2R_ROOT}" >&2; exit 2;
}
[[ "${PHYSICAL_HOST}" == "${CYCLO_LAB_ROOT}"/* ]] || {
  echo "physical HDF must be below ${CYCLO_LAB_ROOT}" >&2; exit 2;
}

PHYSICAL_CONTAINER=${CYCLO_LAB_CONTAINER_ROOT}/${PHYSICAL_HOST#"${CYCLO_LAB_ROOT}"/}
TASK_TAG=task_${TASK_ID}_coffee_transport_headcam
if [[ "${TASK_ID}" == 000002 ]]; then
  # Never reuse the legacy 17D lateral-only staging or dataset in place.
  TASK_TAG=${TASK_TAG}_full_base
fi
NATIVE_HOST=${CYCLO_LAB_ROOT}/datasets/tmp/${TASK_TAG}_native_50_staging
VISUAL_HOST=${CYCLO_LAB_ROOT}/datasets/tmp/${TASK_TAG}_visual8_400_staging
MERGED_HOST=${CYCLO_S2R_ROOT}/docker/workspace/staging/${TASK_TAG}_450_staging
MERGED_CONTAINER=/workspace/staging/${TASK_TAG}_450_staging
DATASET_HOST=${CYCLO_S2R_ROOT}/docker/workspace/dataset/robotis/${TASK_TAG}_v30
DATASET_CONTAINER=/workspace/dataset/robotis/${TASK_TAG}_v30
REPO_ID=robotis/${TASK_TAG}_v30
PROFILE=cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.task_${TASK_ID}.profiles:TASK${TASK_ID}_VISUAL_REPLAY_AUGMENTATION
TASK_NAME=Cyclo-Real-Showroom-Task${TASK_ID}-FFW-SG2-v0
OUT=${CAMPAIGN}/${TASK_TAG}
mkdir -p "${OUT}/logs"

PHYSICAL_SHA=$(sha256sum "${PHYSICAL_HOST}" | cut -d " " -f 1)
verify_source_manifest() {
  python3 -c 'import json, pathlib, sys; value=json.loads((pathlib.Path(sys.argv[1]) / "manifest.json").read_text()); expected=sys.argv[2]; actual=value.get("source_hdf_sha256"); assert actual == expected, f"stale staging source hash: {actual} != {expected}"; rate=value.get("physical_runtime_rate_provenance", {}); assert rate.get("attestation_id") == "coffee_transport_final450_v12_runtime_rates_20260908"; assert rate.get("rate_config_commit") == "f106306bb450f6fb38cb3abf1f35f528b9c1b994"; assert rate.get("rate_contract") == {"physics_hz": 120.0, "control_hz": 15.0, "camera_hz": 15.0, "decimation": 8, "render_interval": 8, "swerve_controller_hz": 30.0, "swerve_update_interval_physics_steps": 4}' "$1" "${PHYSICAL_SHA}"
}
verify_merge_inputs() {
  python3 -c 'import hashlib, json, pathlib, sys; root=pathlib.Path(sys.argv[1]); inputs=[pathlib.Path(value) for value in sys.argv[2:]]; value=json.loads((root / "manifest.json").read_text()); actual=[item.get("manifest_sha256") for item in value.get("source_manifests", [])]; expected=[hashlib.sha256((item / "manifest.json").read_bytes()).hexdigest() for item in inputs]; assert actual == expected, f"stale merged staging inputs: {actual} != {expected}"' "$1" "$2" "$3"
}
verify_dataset_source() {
  python3 -c 'import hashlib, json, pathlib, sys; dataset=pathlib.Path(sys.argv[1]); staging=pathlib.Path(sys.argv[2]); value=json.loads((dataset / "meta/isaac_action_replay_provenance.json").read_text()); actual=value.get("source_staging_manifest_sha256"); expected=hashlib.sha256((staging / "manifest.json").read_bytes()).hexdigest(); assert actual == expected, f"stale LeRobot source hash: {actual} != {expected}"; rate=json.loads((dataset / "meta/physical_runtime_rate_provenance.json").read_text()); assert value.get("physical_runtime_rate_provenance") == rate; assert rate.get("attestation_id") == "coffee_transport_final450_v12_runtime_rates_20260908"; assert rate.get("rate_contract", {}).get("physics_hz") == 120.0' "$1" "$2"
}

run() {
  local name=$1
  shift
  echo "[$(date '+%F %T %Z')] START ${name}"
  "$@" 2>&1 | tee -a "${OUT}/logs/${name}.log"
  echo "[$(date '+%F %T %Z')] DONE ${name}"
}

if [[ ! -f "${NATIVE_HOST}/manifest.json" ]]; then
  run native docker exec "${CYCLO_LAB_CONTAINER}" bash -lc "
    cd '${CYCLO_LAB_CONTAINER_ROOT}' && /isaac-sim/python.sh \
      scripts/sim2real/imitation_learning/tasks/coffee_transport/export_native_policy_staging.py \
      --task-id ${TASK_ID} --input-file '${PHYSICAL_CONTAINER}' \
      --output-dir 'datasets/tmp/${TASK_TAG}_native_50_staging' --expected-episodes 50"
fi
verify_source_manifest "${NATIVE_HOST}"

VISUAL_ARGS=""
[[ -f "${VISUAL_HOST}/manifest.json" ]] && VISUAL_ARGS="--resume"
run visual docker exec "${CYCLO_LAB_CONTAINER}" bash -lc "
  cd '${CYCLO_LAB_CONTAINER_ROOT}' && /isaac-sim/python.sh \
    scripts/sim2real/imitation_learning/tasks/coffee_transport/replay_visual_policy_staging.py \
    --task-id ${TASK_ID} --task '${TASK_NAME}' --input-file '${PHYSICAL_CONTAINER}' \
    --output-dir 'datasets/tmp/${TASK_TAG}_visual8_400_staging' \
    --randomization-profile '${PROFILE}' --repeats 8 --num-envs 5 \
    --expected-source-episodes 50 --seed $((20260907 + 10#${TASK_ID})) \
    --device cuda:0 --headless ${VISUAL_ARGS}"
verify_source_manifest "${VISUAL_HOST}"

run staging_audit docker exec "${CYCLO_LAB_CONTAINER}" bash -lc "
  cd '${CYCLO_LAB_CONTAINER_ROOT}' && /isaac-sim/python.sh \
    scripts/sim2real/imitation_learning/tasks/coffee_transport/audit_policy_staging.py \
    --task-id ${TASK_ID} --native 'datasets/tmp/${TASK_TAG}_native_50_staging' \
    --augmented 'datasets/tmp/${TASK_TAG}_visual8_400_staging' \
    --expected-sources 50 --augmented-repeats 8 \
    --output 'datasets/tmp/${TASK_TAG}_staging_audit.json'"

if [[ ! -f "${MERGED_HOST}/manifest.json" ]]; then
  run merge python3 \
    "${CYCLO_LAB_ROOT}/scripts/sim2real/imitation_learning/data_converter/merge_replay_staging.py" \
    --input "${NATIVE_HOST}" --input "${VISUAL_HOST}" --output "${MERGED_HOST}"
fi
verify_merge_inputs "${MERGED_HOST}" "${NATIVE_HOST}" "${VISUAL_HOST}"

if ! docker inspect --format '{{.State.Running}}' "${CYCLO_S2R_CONTAINER}" 2>/dev/null | grep -qx true; then
  run start_converter_container bash "${CYCLO_S2R_ROOT}/docker/container.sh" start
fi
install -m 0644 \
  "${CYCLO_LAB_ROOT}/scripts/sim2real/imitation_learning/data_converter/replay_staging_to_lerobot_v30.py" \
  "${CYCLO_S2R_ROOT}/docker/workspace/tools/coffee_replay_staging_to_lerobot_v30.py"
install -m 0644 \
  "${CYCLO_LAB_ROOT}/scripts/sim2real/imitation_learning/data_converter/replay_staging_contract.py" \
  "${CYCLO_S2R_ROOT}/docker/workspace/tools/replay_staging_contract.py"
if [[ ! -f "${DATASET_HOST}/meta/info.json" ]]; then
  run lerobot_v30 docker exec \
    -e COFFEE_STAGING="${MERGED_CONTAINER}" -e COFFEE_DATASET="${DATASET_CONTAINER}" \
    -e COFFEE_REPO_ID="${REPO_ID}" \
    -e COFFEE_CYCLO_SOURCE_ROOT="${CYCLO_SOURCE_ROOT_IN_S2R_CONTAINER}" \
    "${CYCLO_S2R_CONTAINER}" bash -lc '
      python3 /workspace/tools/coffee_replay_staging_to_lerobot_v30.py \
        --staging "${COFFEE_STAGING}" --output "${COFFEE_DATASET}" \
        --repo-id "${COFFEE_REPO_ID}" --robot-type ffw_sg2_rev1 \
        --expected-episodes 450 \
        --cyclo-source-root "${COFFEE_CYCLO_SOURCE_ROOT}"'
fi
verify_dataset_source "${DATASET_HOST}" "${MERGED_HOST}"

install -m 0644 \
  "${CYCLO_LAB_ROOT}/scripts/sim2real/imitation_learning/tasks/common/audit_lerobot_v30.py" \
  "${CYCLO_S2R_ROOT}/docker/workspace/tools/coffee_audit_lerobot_v30.py"
EXPECTED_FRAMES=$(python3 -c "import json; print(json.load(open('${CYCLO_LAB_ROOT}/datasets/tmp/${TASK_TAG}_staging_audit.json'))['total_staged_frames'])")
NAMES=(arm_l_joint1 arm_l_joint2 arm_l_joint3 arm_l_joint4 arm_l_joint5 arm_l_joint6 arm_l_joint7 gripper_l_joint1 arm_r_joint1 arm_r_joint2 arm_r_joint3 arm_r_joint4 arm_r_joint5 arm_r_joint6 arm_r_joint7 gripper_r_joint1)
if [[ "${TASK_ID}" == 000001 ]]; then
  NAMES+=(lift_joint)
  INSTRUCTION="Move the right orange coffee can with the right hand and the left orange coffee can with the left hand to the lower cabinet shelf."
else
  NAMES+=(linear_x linear_y angular_z)
  INSTRUCTION="Move the right orange coffee can with the right hand and the left orange coffee can with the left hand to the matching cabinet on the right."
fi
run lerobot_audit docker exec "${CYCLO_S2R_CONTAINER}" python3 \
  /workspace/tools/coffee_audit_lerobot_v30.py \
  --dataset "${DATASET_CONTAINER}" --expected-episodes 450 --expected-frames "${EXPECTED_FRAMES}" \
  --expected-task "${INSTRUCTION}" \
  --expected-action-semantics pre_step_named_joint_position_plus_optional_body_velocity \
  --expected-state-names "${NAMES[@]}" --expected-action-names "${NAMES[@]}" \
  --expected-source-state-names "${NAMES[@]}" --expected-source-action-names "${NAMES[@]}" \
  --expected-cameras cam_left_head

link_once() {
  local target=$1 link=$2
  if [[ -L "${link}" ]]; then
    [[ "$(readlink -f "${link}")" == "$(readlink -f "${target}")" ]] || {
      echo "refusing mismatched symlink: ${link}" >&2; exit 1;
    }
  elif [[ -e "${link}" ]]; then
    echo "refusing existing non-symlink: ${link}" >&2; exit 1
  else
    ln -s "${target}" "${link}"
  fi
}
link_once "${PHYSICAL_HOST}" "${OUT}/physical_50.hdf5"
link_once "${NATIVE_HOST}" "${OUT}/native_staging_50"
link_once "${VISUAL_HOST}" "${OUT}/visual_staging_400"
link_once "${MERGED_HOST}" "${OUT}/merged_staging_450"
link_once "${DATASET_HOST}" "${OUT}/lerobot_v30_450"
link_once "${CYCLO_LAB_ROOT}/datasets/tmp/${TASK_TAG}_staging_audit.json" "${OUT}/staging_audit.json"
echo "COFFEE_POLICY_DATASET_COMPLETE task=${TASK_ID} output=${OUT}"
