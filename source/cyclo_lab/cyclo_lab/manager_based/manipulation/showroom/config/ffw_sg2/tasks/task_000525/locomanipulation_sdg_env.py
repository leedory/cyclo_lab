"""Task000525 adapter for Isaac Lab's upstream locomanipulation_sdg pipeline."""

from __future__ import annotations

import json
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData

from isaaclab_mimic.locomanipulation_sdg.data_classes import LocomanipulationSDGInputData
from isaaclab_mimic.locomanipulation_sdg.envs.locomanipulation_sdg_env import (
    LocomanipulationSDGEnv,
)
from isaaclab_mimic.locomanipulation_sdg.envs.locomanipulation_sdg_env_cfg import (
    LocomanipulationSDGRecorderManagerCfg,
)
from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import (
    OccupancyMap,
    merge_occupancy_maps,
)
from isaaclab_mimic.locomanipulation_sdg.scene_utils import (
    HasOccupancyMap,
    HasPose,
    SceneBody,
)

from cyclo_lab.manager_based.actions.ffw_sg2 import make_ffw_sg2_joint_position_action_cfg
from cyclo_lab.manager_based.manipulation.common.ffw_sg2_mimic_action_cfg import (
    configure_ffw_sg2_mimic_ik_actions,
)
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_PUBLISHED_JOINT_NAMES,
    hdf5_contract_metadata,
)

from ...randomization.task_pose_events import randomize_robot_root
from .destination_mat import TASK000525_DESTINATION_MAT_DIMENSIONS_M
from .env_cfg import Task000525EnvCfg
from .layout import TASK000525_CAN_RADIUS_M
from .profiles import (
    TASK000525_PHYSICAL_TRAJECTORY_GENERATION,
    Task000525RandomizationCfg,
)
from .reset_events import randomize_coffee_can_center_regions
from .locomanipulation_sdg_contract import (
    DESTINATION_DOCKING_PARENT_POSE_WXYZ,
    SDG_FRAME_POSE_DIM,
    SDG_WORKING_ACTION_DIM,
    SHOWROOM_STATIC_OBSTACLE_AABBS,
    SOURCE_FIXTURE_POSE_WXYZ,
    STATIC_MAP_PREFILL_BUFFER_M,
)


class _ArticulationRootPose(HasPose):
    def __init__(self, articulation):
        self._articulation = articulation

    def get_pose(self) -> torch.Tensor:
        return torch.cat(
            (self._articulation.data.root_pos_w, self._articulation.data.root_quat_w),
            dim=-1,
        )


class _RigidObjectAsset(HasPose):
    """Isaac Lab 0.47-compatible pose adapter for a RigidObject."""

    def __init__(self, scene, entity_name: str):
        self.scene = scene
        self.entity_name = entity_name

    def get_pose(self) -> torch.Tensor:
        asset = self.scene[self.entity_name]
        return torch.cat((asset.data.root_pos_w, asset.data.root_quat_w), dim=-1)

    def set_pose(self, pose: torch.Tensor) -> None:
        asset = self.scene[self.entity_name]
        asset.write_root_pose_to_sim(pose[..., :7])


class _FixedFixture(HasPose, HasOccupancyMap):
    """A fixed pose plus a world-frame obstacle map for authored showroom USD."""

    def __init__(
        self,
        pose_wxyz: tuple[float, ...],
        obstacle_aabbs: tuple[tuple[object, ...], ...],
        *,
        device: str,
        prefill_buffer_m: float = 0.0,
    ):
        self._pose = torch.tensor(pose_wxyz, dtype=torch.float32, device=device)[None, :]
        self._obstacle_aabbs = obstacle_aabbs
        self._prefill_buffer_m = float(prefill_buffer_m)

    def get_pose(self) -> torch.Tensor:
        return self._pose

    def get_occupancy_map(self) -> OccupancyMap:
        maps = []
        for _name, min_x, min_y, max_x, max_y in self._obstacle_aabbs:
            boundary = np.asarray(
                (
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                ),
                dtype=np.float64,
            )
            maps.append(OccupancyMap.from_occupancy_boundary(boundary, resolution=0.05))
        merged = merge_occupancy_maps(maps)
        if self._prefill_buffer_m > 0.0:
            merged = merged.buffered_meters(self._prefill_buffer_m)
        return merged


@configclass
class Task000525LocomanipulationSDGEnvCfg(Task000525EnvCfg):
    """Task525 scene with hybrid EEF19 + body-velocity3 generation actions."""

    env_name: str = "Cyclo-Real-Showroom-Task000525-Locomanipulation-SDG-FFW-SG2-v0"
    recorders: LocomanipulationSDGRecorderManagerCfg = (
        LocomanipulationSDGRecorderManagerCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        self.env_name = (
            "Cyclo-Real-Showroom-Task000525-Locomanipulation-SDG-FFW-SG2-v0"
        )
        configure_ffw_sg2_mimic_ik_actions(
            self.actions, posture_bias_sides=("r",)
        )
        # Task525 manipulates only the right arm. Keep the inactive left arm on
        # its demonstrated joint targets instead of solving an underconstrained
        # Cartesian hold that can drift in the IK null space.
        self.actions.arm_l_action = make_ffw_sg2_joint_position_action_cfg(
            FFW_SG2_LEFT_ARM_JOINT_NAMES
        )


@configclass
class Task000525TrajectoryGenerationEnvCfg(
    Task000525LocomanipulationSDGEnvCfg
):
    """Physical Task525 trajectory generation from randomized valid starts."""

    env_name: str = (
        "Cyclo-Real-Showroom-Task000525-Trajectory-Generation-FFW-SG2-v0"
    )
    randomization: Task000525RandomizationCfg = (
        TASK000525_PHYSICAL_TRAJECTORY_GENERATION
    )

    def __post_init__(self):
        super().__post_init__()
        self.env_name = (
            "Cyclo-Real-Showroom-Task000525-Trajectory-Generation-FFW-SG2-v0"
        )

class Task000525LocomanipulationSDGEnv(LocomanipulationSDGEnv):
    """Keep upstream phase/cursor/path logic outside the SG2-specific adapter."""

    def __init__(self, cfg: Task000525LocomanipulationSDGEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        handler = getattr(self.recorder_manager, "_dataset_file_handler", None)
        data_group = getattr(handler, "_hdf5_data_group", None)
        if data_group is not None:
            metadata = {
                "schema_version": "cyclo_lab_hdf5_v1",
                "observation_semantics": "pre_step",
                "obs_last_action_semantics": "previous_step_action",
                "scene_state_semantics": "post_step",
                **hdf5_contract_metadata(self.cfg.actions),
            }
            for key, value in metadata.items():
                data_group.attrs[key] = (
                    json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                )
        robot = self.scene["robot"]
        self._robot_root = _ArticulationRootPose(robot)
        self._passive_joint_ids = torch.tensor(
            [robot.joint_names.index(name) for name in FFW_SG2_PUBLISHED_JOINT_NAMES[16:19]],
            dtype=torch.long,
            device=self.device,
        )
        self._action_joint_ids = torch.tensor(
            [robot.joint_names.index(name) for name in FFW_SG2_PUBLISHED_JOINT_NAMES[:19]],
            dtype=torch.long,
            device=self.device,
        )
        self._start_fixture = _FixedFixture(
            SOURCE_FIXTURE_POSE_WXYZ,
            SHOWROOM_STATIC_OBSTACLE_AABBS,
            device=self.device,
            prefill_buffer_m=STATIC_MAP_PREFILL_BUFFER_M,
        )
        self._end_fixture = _FixedFixture(
            DESTINATION_DOCKING_PARENT_POSE_WXYZ,
            (),
            device=self.device,
        )

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        """Restore a seed, then reapply Task525 generation pose randomization.

        Isaac Lab ``reset_to`` runs reset events first and then overwrites them
        with the HDF5 initial state. Reapplying only the existing Task525 root
        and B-region events here preserves the requested randomized episode
        start without duplicating their sampling logic.
        """

        obs, extras = super().reset_to(
            state,
            env_ids,
            seed=seed,
            is_relative=is_relative,
        )
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        randomization_applied = False

        robot_randomization = getattr(self.cfg.randomization, "robot_root", None)
        if robot_randomization is not None and robot_randomization.enabled:
            randomize_robot_root(
                self,
                ids,
                depth_x_max_m=robot_randomization.depth_x_max_m,
                lateral_y_max_m=robot_randomization.lateral_y_max_m,
                yaw_max_rad=robot_randomization.yaw_max_rad,
            )
            randomization_applied = True

        coffee_positions = self.cfg.randomization.coffee_positions
        if coffee_positions.enabled:
            randomize_coffee_can_center_regions(
                self,
                ids,
                layout_key=coffee_positions.layout_key,
            )
            randomization_applied = True

        if not randomization_applied:
            return obs, extras

        self.scene.write_data_to_sim()
        self.sim.forward()
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        # ``super().reset_to`` already opened a recorder episode with the seed
        # state. Replace that empty episode so initial_state records the actual
        # randomized root/can poses used by trajectory generation.
        self.recorder_manager.reset(ids)
        self.recorder_manager.record_post_reset(ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, extras

    @staticmethod
    def _episode_value(episode_data: EpisodeData, key: str, step: int) -> torch.Tensor:
        value = episode_data.data
        for part in key.split("/"):
            if part not in value:
                raise KeyError(
                    f"Task525 locomanipulation source is missing '{key}'. "
                    "Record a new Task525 seed with the mobile 22D observation contract."
                )
            value = value[part]
        return value[step]

    def load_input_data(
        self, episode_data: EpisodeData, step: int
    ) -> LocomanipulationSDGInputData | None:
        action = episode_data.get_action(step)
        if action is None:
            return None
        if action.shape[-1] != SDG_WORKING_ACTION_DIM:
            raise ValueError(
                f"Task525 source action must be 22D joint19+base3, got {tuple(action.shape)}"
            )

        # The local hybrid action uses these source joint commands for the
        # inactive left arm and passive head/lift terms.
        self._source_joint_action = action[:19].detach().clone()

        return LocomanipulationSDGInputData(
            left_hand_pose_target=self._episode_value(
                episode_data, "obs/left_eef_pose_world", step
            ),
            right_hand_pose_target=self._episode_value(
                episode_data, "obs/right_eef_pose_world", step
            ),
            left_hand_joint_positions_target=action[7:8],
            right_hand_joint_positions_target=action[15:16],
            base_pose=self._episode_value(
                episode_data, "obs/robot_root_pose_world", step
            ),
            object_pose=self._episode_value(
                episode_data, "obs/target_object_pose_world", step
            ),
            fixture_pose=torch.tensor(
                SOURCE_FIXTURE_POSE_WXYZ,
                dtype=action.dtype,
                device=action.device,
            ),
        )

    def _world_pose_to_current_root(self, pose: torch.Tensor) -> torch.Tensor:
        pose = pose.to(device=self.device, dtype=torch.float32).reshape(1, 7)
        root_pose = self._robot_root.get_pose()
        position, quaternion = math_utils.subtract_frame_transforms(
            root_pose[:, :3],
            root_pose[:, 3:],
            pose[:, :3],
            pose[:, 3:],
        )
        return torch.cat((position, quaternion), dim=-1)[0]

    def build_action_vector(
        self,
        left_hand_pose_target: torch.Tensor,
        right_hand_pose_target: torch.Tensor,
        left_hand_joint_positions_target: torch.Tensor,
        right_hand_joint_positions_target: torch.Tensor,
        base_velocity_target: torch.Tensor,
    ) -> torch.Tensor:
        action = torch.zeros(
            (self.num_envs, SDG_WORKING_ACTION_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        left_pose = self._world_pose_to_current_root(left_hand_pose_target)
        right_pose = self._world_pose_to_current_root(right_hand_pose_target)
        if left_pose.shape != (SDG_FRAME_POSE_DIM,) or right_pose.shape != (
            SDG_FRAME_POSE_DIM,
        ):
            raise ValueError("Task525 SDG hand targets must each be a 7D WXYZ pose")

        if not hasattr(self, "_source_joint_action"):
            raise RuntimeError("Task525 SDG source joint action is unavailable")
        action[0, 0:7] = self._source_joint_action[0:7].to(self.device)
        action[0, 7:8] = left_hand_joint_positions_target.to(self.device).reshape(1)
        action[0, 8:15] = right_pose
        action[0, 15:16] = right_hand_joint_positions_target.to(self.device).reshape(1)
        action[0, 16:19] = self._source_joint_action[16:19].to(self.device)
        action[0, 19:22] = base_velocity_target.to(self.device).reshape(3)
        if not torch.isfinite(action).all():
            raise ValueError("Task525 SDG action contains non-finite values")
        return action

    def _right_tcp_pose_world(self) -> torch.Tensor:
        sensor = self.scene["right_eef"]
        return torch.cat(
            (sensor.data.target_pos_w[:, 0, :], sensor.data.target_quat_w[:, 0, :]),
            dim=-1,
        )

    def evaluate_task525_carry_checkpoint(
        self,
        input_episode_data: EpisodeData,
        navigate_step: int,
        initial_object_pose: torch.Tensor,
    ) -> tuple[bool, str, dict[str, float]]:
        """Reject a failed pick before spending time on navigation."""

        object_pose = self.get_object().get_pose()
        right_tcp = self._right_tcp_pose_world()
        eef_object_distance = float(
            torch.linalg.vector_norm(right_tcp[0, :3] - object_pose[0, :3]).item()
        )
        object_displacement = float(
            torch.linalg.vector_norm(object_pose[0, :3] - initial_object_pose[0, :3]).item()
        )

        expected = input_episode_data.get_action(navigate_step)[:19].to(self.device)
        measured = self.scene["robot"].data.joint_pos[0, self._action_joint_ids]
        # Contact can keep either gripper away from its requested endpoint.
        posture_indices = [index for index in range(19) if index not in (7, 15)]
        home_joint_max_error = float(
            torch.max(torch.abs(measured[posture_indices] - expected[posture_indices])).item()
        )
        metrics = {
            "carry_eef_object_distance_m": eef_object_distance,
            "carry_object_displacement_m": object_displacement,
            "carry_home_joint_max_error_rad_or_m": home_joint_max_error,
        }
        if eef_object_distance > 0.080:
            return False, "carry_checkpoint: target can is no longer held by the right gripper", metrics
        if object_displacement < 0.200:
            return False, "carry_checkpoint: target can did not clear the cabinet", metrics
        if home_joint_max_error > 0.150:
            return False, "carry_checkpoint: robot did not return to the demonstrated carry/home posture", metrics
        return True, "", metrics

    def evaluate_task525_final_checkpoint(self) -> tuple[bool, str, dict[str, float]]:
        """Require a released, stable can fully inside the destination mat."""

        target = self.scene[self.cfg.target_object]
        mat = self.scene["destination_mat"]
        relative_position, _ = math_utils.subtract_frame_transforms(
            mat.data.root_pos_w,
            mat.data.root_quat_w,
            target.data.root_pos_w,
            target.data.root_quat_w,
        )
        local = relative_position[0]
        half_x = 0.5 * TASK000525_DESTINATION_MAT_DIMENSIONS_M[0]
        half_y = 0.5 * TASK000525_DESTINATION_MAT_DIMENSIONS_M[1]
        margin = TASK000525_CAN_RADIUS_M + 0.005
        inside_x = abs(float(local[0])) <= half_x - margin
        inside_y = abs(float(local[1])) <= half_y - margin
        supported_z = 0.020 <= float(local[2]) <= 0.090

        linear_speed = float(torch.linalg.vector_norm(target.data.root_lin_vel_w[0]).item())
        angular_speed = float(torch.linalg.vector_norm(target.data.root_ang_vel_w[0]).item())
        eef_distance = float(
            torch.linalg.vector_norm(
                self._right_tcp_pose_world()[0, :3] - target.data.root_pos_w[0]
            ).item()
        )
        root_pose = self._robot_root.get_pose()[0]
        metrics = {
            "final_can_mat_local_x_m": float(local[0]),
            "final_can_mat_local_y_m": float(local[1]),
            "final_can_mat_local_z_m": float(local[2]),
            "final_can_world_x_m": float(target.data.root_pos_w[0, 0]),
            "final_can_world_y_m": float(target.data.root_pos_w[0, 1]),
            "final_mat_world_x_m": float(mat.data.root_pos_w[0, 0]),
            "final_mat_world_y_m": float(mat.data.root_pos_w[0, 1]),
            "final_robot_world_x_m": float(root_pose[0]),
            "final_robot_world_y_m": float(root_pose[1]),
            "final_robot_world_qw": float(root_pose[3]),
            "final_robot_world_qz": float(root_pose[6]),
            "final_can_linear_speed_mps": linear_speed,
            "final_can_angular_speed_radps": angular_speed,
            "final_right_eef_can_distance_m": eef_distance,
        }
        if not (inside_x and inside_y and supported_z):
            return False, "final_checkpoint: target can is not fully supported inside the destination mat", metrics
        if linear_speed > 0.030 or angular_speed > 0.250:
            return False, "final_checkpoint: target can has not settled", metrics
        if eef_distance < 0.100:
            return False, "final_checkpoint: right gripper has not released the target can", metrics
        return True, "", metrics

    def get_base(self) -> HasPose:
        return self._robot_root

    def get_left_hand(self) -> HasPose:
        return SceneBody(self.scene, "robot", "arm_l_link7")

    def get_right_hand(self) -> HasPose:
        return SceneBody(self.scene, "robot", "arm_r_link7")

    def get_object(self) -> HasPose:
        return _RigidObjectAsset(self.scene, self.cfg.target_object)

    def get_start_fixture(self):
        return self._start_fixture

    def get_end_fixture(self):
        return self._end_fixture

    def get_obstacle_fixtures(self) -> list:
        # All fixed showroom furniture is already merged into the start map.
        # This intentionally disables upstream random fixture placement.
        return []
