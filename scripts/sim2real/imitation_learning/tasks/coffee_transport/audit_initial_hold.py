"""Audit Coffee Transport arm/root transients over the first recorded frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


EXPECTED_CONTRACT_IDS = {
    "coffee_realized_pose_endpoint_lift_targets_v13",
    "coffee_production_realized_pose_commanded_joint_hold_v12",
}
LEGACY_SG2_RAW31_ARM_INDICES = (
    5,
    10,
    13,
    15,
    17,
    19,
    21,
    6,
    11,
    14,
    16,
    18,
    20,
    22,
)


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _seven_joint_attr(demo: h5py.Group, key: str, *, required: bool):
    encoded = demo.attrs.get(key)
    if encoded is None:
        if required:
            raise ValueError(f"{demo.name}: missing {key}")
        return None
    values = np.asarray(json.loads(_text(encoded)), dtype=float)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError(f"{demo.name}: invalid {key}: {values}")
    return values.tolist()


def _arm_indices(data: h5py.Group, raw_width: int) -> tuple[list[int], str]:
    encoded = data.attrs.get("articulation_joint_names")
    if encoded is not None:
        names = json.loads(_text(encoded))
        indices = [
            index
            for index, name in enumerate(names)
            if name.startswith(("arm_l_joint", "arm_r_joint"))
        ]
        if len(names) != raw_width or len(indices) != 14:
            raise ValueError(
                f"invalid articulation_joint_names: names={len(names)} "
                f"raw_width={raw_width} arms={len(indices)}"
            )
        return indices, "dataset_articulation_joint_names"
    if raw_width == 31:
        return list(LEGACY_SG2_RAW31_ARM_INDICES), "legacy_ffw_sg2_raw31_contract"
    raise ValueError(
        "dataset has no articulation_joint_names and is not the known 31-joint SG2 layout"
    )


def audit(path: Path, frames: int) -> dict:
    with h5py.File(path, "r") as source:
        data = source["data"]
        method = _text(data.attrs.get("generation_method", "missing"))
        contract_id = _text(
            data.attrs.get("initial_normalization_contract_id", "legacy_missing")
        )
        results = []
        index_source = None
        for name in sorted(data):
            demo = data[name]
            hold_targets = {}
            hold_start_actual = {}
            for side in ("left", "right"):
                hold_targets[side] = _seven_joint_attr(
                    demo,
                    f"normalization_external_joint_hold_{side}_target_rad_json",
                    required=contract_id in EXPECTED_CONTRACT_IDS,
                )
                hold_start_actual[side] = _seven_joint_attr(
                    demo,
                    f"normalization_external_joint_hold_{side}_start_actual_rad_json",
                    required=contract_id in EXPECTED_CONTRACT_IDS,
                )
            initial_joint = np.asarray(
                demo["initial_state/articulation/robot/joint_position"]
            )[0]
            state_joint = np.asarray(
                demo["states/articulation/robot/joint_position"]
            )[:frames]
            initial_root = np.asarray(
                demo["initial_state/articulation/robot/root_pose"]
            )[0]
            state_root = np.asarray(
                demo["states/articulation/robot/root_pose"]
            )[:frames]
            indices, index_source = _arm_indices(data, initial_joint.shape[0])
            arm_max = float(
                np.max(np.abs(state_joint[:, indices] - initial_joint[indices]))
            )
            root_xy_max = float(
                np.max(np.linalg.norm(state_root[:, :2] - initial_root[:2], axis=1))
            )
            results.append(
                {
                    "demo": name,
                    "frames": int(min(frames, state_joint.shape[0])),
                    "arm_joint_max_delta_rad": arm_max,
                    "root_xy_max_delta_m": root_xy_max,
                    "recorded_arm_joint_max_delta_rad": demo.attrs.get(
                        "normalization_first10_arm_joint_max_delta_rad"
                    ),
                    "recorded_root_xy_max_delta_m": demo.attrs.get(
                        "normalization_first10_root_xy_max_delta_m"
                    ),
                    "external_joint_hold_steps": demo.attrs.get(
                        "normalization_external_joint_hold_steps"
                    ),
                    "external_joint_hold_targets_rad": hold_targets,
                    "external_joint_hold_start_actual_rad": hold_start_actual,
                }
            )
    arm = np.asarray([item["arm_joint_max_delta_rad"] for item in results])
    root = np.asarray([item["root_xy_max_delta_m"] for item in results])
    return {
        "path": str(path.resolve()),
        "generation_method": method,
        "initial_normalization_contract_id": contract_id,
        "contract_is_current": contract_id == EXPECTED_CONTRACT_ID,
        "arm_index_source": index_source,
        "episodes": len(results),
        "frames_requested": frames,
        "summary": {
            "arm_joint_max_delta_rad_max": float(np.max(arm)),
            "arm_joint_max_delta_rad_median": float(np.median(arm)),
            "root_xy_max_delta_m_max": float(np.max(root)),
            "root_xy_max_delta_m_median": float(np.median(root)),
        },
        "per_episode": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5", type=Path, nargs="+")
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be positive")
    print(
        json.dumps(
            [audit(path, args.frames) for path in args.hdf5],
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item()
            if isinstance(value, np.generic)
            else value,
        )
    )


if __name__ == "__main__":
    main()
