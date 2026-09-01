# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to convert recorded demonstration actions between IK and joint space."""

import argparse
import json
import multiprocessing
import os
from copy import deepcopy

import torch
from tqdm import tqdm

from isaaclab.utils.datasets import HDF5DatasetFileHandler, EpisodeData
from cyclo_lab.robot_specs.ffw.sg2 import converted_action_contract_metadata

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

def convert_joint_to_ik_omy(ep_data: EpisodeData) -> EpisodeData:
    """Convert joint actions to IK (EEF state + gripper) for OMY robot."""
    try:
        eef_pose = ep_data.data["obs"]["eef_pose"]
        joint_actions = ep_data.data["actions"]

        gripper_action = joint_actions[:, -1:]
        new_actions = torch.cat([eef_pose, gripper_action], dim=1)

        ep_data.data["actions"] = new_actions
        return ep_data
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to convert joint to IK for OMY: {str(e)}")

def convert_joint_to_ik_ffw_sg2(ep_data: EpisodeData) -> EpisodeData:
    """Convert joint19[/base3] actions to EEF19[/base3] Mimic actions."""
    try:
        # FFW SG2 has dual arms, need to handle both left and right EEF states
        left_eef_pose = ep_data.data["obs"]["left_eef_pose"]
        right_eef_pose = ep_data.data["obs"]["right_eef_pose"]
        joint_actions = ep_data.data["actions"]

        if joint_actions.ndim != 2 or joint_actions.shape[1] not in (19, 22):
            raise ValueError(
                "FFW_SG2 actions must have shape [N, 19] or [N, 22], "
                f"got {tuple(joint_actions.shape)}."
            )

        # FFW SG2 joint action prefix (19D):
        # [arm_l(7), gripper_l(1), arm_r(7), gripper_r(1), head(2), lift(1)]
        gripper_l_action = joint_actions[:, 7:8]  # Index 7: gripper_l_joint1
        gripper_r_action = joint_actions[:, 15:16]  # Index 15: gripper_r_joint1
        head_action = joint_actions[:, 16:18]       # Indices 16-17: head joints
        lift_action = joint_actions[:, 18:19]       # Index 18: lift joint

        # IK action prefix (19D):
        # [left_eef(7), gripper_l(1), right_eef(7), gripper_r(1), head(2), lift(1)]
        action_parts = [
            left_eef_pose,    # 0-6: left EEF (pos + quat)
            gripper_l_action,  # 7: left gripper
            right_eef_pose,   # 8-14: right EEF (pos + quat)
            gripper_r_action,  # 15: right gripper
            head_action,       # 16-17: head joints
            lift_action        # 18: lift joint
        ]
        if joint_actions.shape[1] == 22:
            # Preserve the measured/generated mobile command without treating
            # it as part of an EEF pose.
            action_parts.append(joint_actions[:, 19:22])
        new_actions = torch.cat(action_parts, dim=1)

        ep_data.data["actions"] = new_actions
        return ep_data
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to convert joint to IK for FFW_SG2: {str(e)}")

def convert_joint_to_ik(ep_data: EpisodeData, robot_type: str) -> EpisodeData:
    """Convert joint actions to IK based on robot type."""
    if robot_type == "OMY":
        return convert_joint_to_ik_omy(ep_data)
    elif robot_type == "FFW_SG2":
        return convert_joint_to_ik_ffw_sg2(ep_data)
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")

def convert_ik_to_joint(ep_data: EpisodeData, robot_type: str | None = None) -> EpisodeData:
    """Convert IK actions to joint targets."""
    try:
        joint_targets = ep_data.data["obs"]["joint_pos_target"]
        if robot_type == "FFW_SG2":
            if joint_targets.ndim != 2 or joint_targets.shape[1] != 19:
                raise ValueError(
                    "FFW_SG2 joint_pos_target must have shape [N, 19] before converting to joint actions. "
                    f"Got {tuple(joint_targets.shape)}."
                )
            ik_actions = ep_data.data["actions"]
            if ik_actions.ndim != 2 or ik_actions.shape[1] not in (19, 22):
                raise ValueError(
                    "FFW_SG2 IK actions must have shape [N, 19] or [N, 22], "
                    f"got {tuple(ik_actions.shape)}."
                )
            if ik_actions.shape[0] != joint_targets.shape[0]:
                raise ValueError(
                    "FFW_SG2 actions and joint_pos_target must have the same number of steps."
                )
            # Observation and action tensors share the canonical 19D public
            # prefix. Preserve the IK dataset's base3 tail for mobile tasks.
            ep_data.data["actions"] = (
                torch.cat((joint_targets, ik_actions[:, 19:22]), dim=1)
                if ik_actions.shape[1] == 22
                else joint_targets
            )
        else:
            ep_data.data["actions"] = joint_targets
        return ep_data
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError(f"Failed to convert IK to joint: {str(e)}")

def _copy_hdf5_attributes(input_handler, output_handler) -> None:
    """Preserve dataset provenance not represented by EpisodeData."""
    source = input_handler._hdf5_data_group.attrs
    target = output_handler._hdf5_data_group.attrs
    for key, value in source.items():
        if key not in ("env_args", "total"):
            target[key] = value


def _write_hdf5_attribute(group, key: str, value: object) -> None:
    """Write structured contract metadata in the recorder's JSON convention."""

    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value)
    group.attrs[key] = value


def _rewrite_ffw_sg2_action_contract(
    input_handler, output_handler, action_type: str, action_dim: int
) -> None:
    """Replace stale input action metadata while preserving immediate provenance."""

    source = input_handler._hdf5_data_group.attrs
    target = output_handler._hdf5_data_group
    output_contract = converted_action_contract_metadata(action_type, action_dim)
    for key, value in output_contract.items():
        if key in source:
            _write_hdf5_attribute(target, f"source_{key}", source[key])
        _write_hdf5_attribute(target, key, value)
    _write_hdf5_attribute(target, "action_representation", action_type)


def process_dataset(
    input_file: str,
    output_file: str,
    action_type: str,
    robot_type: str,
    output_env_name: str | None = None,
) -> None:
    """Process dataset episodes and convert actions to the desired type."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input dataset file does not exist: {input_file}")

    input_handler = HDF5DatasetFileHandler()
    output_handler = HDF5DatasetFileHandler()

    input_handler.open(input_file)
    source_env_name = input_handler.get_env_name()
    output_handler.create(output_file, env_name=output_env_name or source_env_name)
    _copy_hdf5_attributes(input_handler, output_handler)

    try:
        episode_names = list(input_handler.get_episode_names())
        if not episode_names:
            raise ValueError("Input dataset contains no episodes.")
        if robot_type == "FFW_SG2":
            first_actions = input_handler._hdf5_data_group[episode_names[0]].get(
                "actions"
            )
            if first_actions is None:
                raise ValueError(f"{episode_names[0]} has no actions dataset.")
            _rewrite_ffw_sg2_action_contract(
                input_handler, output_handler, action_type, int(first_actions.shape[-1])
            )
        else:
            output_handler._hdf5_data_group.attrs["action_representation"] = action_type
        skipped_episodes = []
        
        for name in tqdm(episode_names, desc="Processing episodes"):
            try:
                ep_data = input_handler.load_episode(name, device="cpu")

                if ep_data.success is not None and not ep_data.success:
                    continue

                processed = deepcopy(ep_data)
                
                # Apply conversion based on action type
                if action_type == "ik":
                    processed = convert_joint_to_ik(processed, robot_type)
                elif action_type == "joint":
                    processed = convert_ik_to_joint(processed, robot_type)
                
                output_handler.write_episode(processed)
                source_group = input_handler._hdf5_data_group[name]
                output_name = f"demo_{output_handler.demo_count - 1}"
                output_group = output_handler._hdf5_data_group[output_name]
                for key, value in source_group.attrs.items():
                    if key not in ("num_samples", "seed", "success"):
                        output_group.attrs[key] = value
                
            except Exception as e:
                skipped_episodes.append((name, str(e)))
                print(f"\nWarning: Skipping episode '{name}' due to error: {str(e)}")
                continue
        
        if skipped_episodes:
            print(f"\n\nSummary: Skipped {len(skipped_episodes)} episode(s) due to errors:")
            for ep_name, error_msg in skipped_episodes:
                print(f"  - {ep_name}: {error_msg}")

    finally:
        input_handler.close()
        output_handler.flush()
        output_handler.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert recorded demonstration actions between IK and joint space."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./datasets/annotated_dataset.hdf5",
        help="Path to input dataset file."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./datasets/processed_annotated_dataset.hdf5",
        help="Path to save processed dataset file."
    )
    parser.add_argument(
        "--action_type",
        choices=["ik", "joint"],
        required=True,
        help="Target action representation: 'ik' or 'joint'."
    )
    parser.add_argument(
        "--output_env_name",
        type=str,
        default=None,
        help="Optional environment ID for the converted dataset; input ID is preserved by default.",
    )
    parser.add_argument(
        "--robot_type",
        choices=["OMY", "FFW_SG2"],
        required=True,
        help="Robot type: 'OMY' or 'FFW_SG2'."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    process_dataset(
        args.input_file,
        args.output_file,
        args.action_type,
        args.robot_type,
        args.output_env_name,
    )

if __name__ == "__main__":
    main()
