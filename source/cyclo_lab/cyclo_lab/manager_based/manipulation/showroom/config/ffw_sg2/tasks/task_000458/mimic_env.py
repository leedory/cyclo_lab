"""Isaac Lab Mimic adapter for the showroom peanut take-out task."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.utils.math as PoseUtils

from cyclo_lab.manager_based.manipulation.common import FFWSG2MimicEnv

from ...platform.replay_state import prepare_sg2_position_replay_state
from . import takeout_terms


ACTION_DIM = 19
INACTIVE_JOINT_NAMES = (
    *(f"arm_l_joint{index}" for index in range(1, 8)),
    "gripper_l_joint1",
    "head_joint1",
    "head_joint2",
    "lift_joint",
)
INACTIVE_ACTION_INDICES = (*range(0, 8), *range(16, 19))
RIGHT_MIMIC_ACTION_INDICES = tuple(range(8, 16))


def _validate_action_layout() -> None:
    if len(INACTIVE_JOINT_NAMES) != len(INACTIVE_ACTION_INDICES):
        raise ValueError("inactive joint names and action slots differ in length")
    if set(INACTIVE_ACTION_INDICES).intersection(RIGHT_MIMIC_ACTION_INDICES):
        raise ValueError("inactive holds overlap the active right-arm Mimic slice")
    if set(INACTIVE_ACTION_INDICES).union(RIGHT_MIMIC_ACTION_INDICES) != set(
        range(ACTION_DIM)
    ):
        raise ValueError("Task458 action layout must cover exactly 19 dimensions")


_validate_action_layout()


class Task000458MimicEnv(FFWSG2MimicEnv):
    """Provide robot-relative objects and hold non-commanded SG2 joints."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._hold_inactive_joints = bool(cfg.hold_inactive_joint_targets)
        if self._hold_inactive_joints:
            self._initialize_inactive_joint_holds()

    def _ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if isinstance(env_ids, slice):
            return torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )[env_ids]
        return torch.as_tensor(
            env_ids, dtype=torch.long, device=self.device
        ).reshape(-1)

    def _initialize_inactive_joint_holds(self) -> None:
        robot = self.scene["robot"]
        name_to_index = {
            name: index for index, name in enumerate(robot.data.joint_names)
        }
        missing = [
            name for name in INACTIVE_JOINT_NAMES if name not in name_to_index
        ]
        if missing:
            raise RuntimeError(f"SG2 joints required for holding are missing: {missing}")
        self._inactive_joint_ids = torch.tensor(
            [name_to_index[name] for name in INACTIVE_JOINT_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self._inactive_joint_targets = torch.empty(
            (self.num_envs, len(INACTIVE_JOINT_NAMES)),
            dtype=robot.data.joint_pos.dtype,
            device=self.device,
        )
        self._capture_inactive_joint_holds(None)

    def _capture_inactive_joint_holds(
        self, env_ids: Sequence[int] | None
    ) -> None:
        if not self._hold_inactive_joints:
            return
        ids = self._ids(env_ids)
        values = self.scene["robot"].data.joint_pos[
            ids[:, None], self._inactive_joint_ids[None, :]
        ]
        if not torch.isfinite(values).all():
            raise RuntimeError("episode-start hold targets contain non-finite values")
        self._inactive_joint_targets[ids] = values

    def _finish_episode_reset(self, env_ids: Sequence[int] | None) -> None:
        ids = self._ids(env_ids)
        takeout_terms.reset_takeout_metric_state(
            self, ids, target_name=self.cfg.target_object
        )
        self._capture_inactive_joint_holds(ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)

    def reset(self, seed=None, env_ids=None, options=None):
        _, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        self._finish_episode_reset(env_ids)
        return self.obs_buf, extras

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        _, extras = super().reset_to(
            prepare_sg2_position_replay_state(state),
            env_ids,
            seed=seed,
            is_relative=is_relative,
        )
        self._finish_episode_reset(env_ids)
        return self.obs_buf, extras

    def _ensure_generation_success_tracking_buffers(self) -> None:
        if hasattr(self, "_task458_generation_grasp_stable"):
            return
        self._task458_generation_grasp_stable = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._task458_generation_released_after_takeout = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._task458_generation_final_target_outside_shelf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def reset_generation_success_tracking(self, env_id: int, success_term=None) -> None:
        """Reset the Task458 generation-only episode success tracker."""
        self._ensure_generation_success_tracking_buffers()
        index = int(env_id)
        self._task458_generation_grasp_stable[index] = False
        self._task458_generation_released_after_takeout[index] = False
        self._task458_generation_final_target_outside_shelf[index] = False

    def update_generation_success_tracking(self, env_id: int, success_term=None) -> bool:
        """Track subtask milestones with OR and final target-outside state.

        The generated episode is successful when:
        grasp_stable_ever AND released_after_takeout_ever AND final_target_outside_shelf.
        """
        self._ensure_generation_success_tracking_buffers()
        index = int(env_id)
        metric_params = self.cfg.seed_success_metric_params
        success_params = getattr(success_term, "params", None) or {}
        if "metric_params" in success_params:
            metric_params = success_params["metric_params"]
        metrics = takeout_terms._takeout_metrics(self, **metric_params)
        self._task458_generation_grasp_stable[index] = (
            self._task458_generation_grasp_stable[index]
            | metrics["grasp_stable"][index]
        )
        self._task458_generation_released_after_takeout[index] = (
            self._task458_generation_released_after_takeout[index]
            | metrics["released_after_takeout"][index]
        )
        self._task458_generation_final_target_outside_shelf[index] = metrics[
            "target_outside_shelf"
        ][index]
        return self.get_generation_success(env_id=index)

    def get_generation_success(self, env_id: int, success_term=None) -> bool:
        """Return Task458 generation success for the current episode."""
        self._ensure_generation_success_tracking_buffers()
        index = int(env_id)
        success = (
            self._task458_generation_grasp_stable[index]
            & self._task458_generation_released_after_takeout[index]
            & self._task458_generation_final_target_outside_shelf[index]
        )
        return bool(success.item())

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Keep the right Mimic command and hold all inactive axes."""
        action = super().target_eef_pose_to_action(
            target_eef_pose_dict,
            gripper_action_dict,
            action_noise_dict=action_noise_dict,
            env_id=env_id,
        )
        if not self._hold_inactive_joints:
            return action
        if action.shape != (1, ACTION_DIM):
            raise RuntimeError(
                f"expected one {ACTION_DIM}D action, got {tuple(action.shape)}"
            )
        action = action.clone()
        action[0, list(INACTIVE_ACTION_INDICES)] = self._inactive_joint_targets[
            int(env_id)
        ]
        return action

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        """Return all rigid-object poses in the robot-root frame."""
        if env_ids is None:
            env_ids = slice(None)
        scene_state = self.scene.get_state(is_relative=True)
        robot_root = scene_state["articulation"]["robot"]["root_pose"]
        root_pos = robot_root[env_ids, :3]
        root_quat = robot_root[env_ids, 3:7]
        object_poses = {}
        for name, state in scene_state["rigid_object"].items():
            position, quaternion = PoseUtils.subtract_frame_transforms(
                root_pos,
                root_quat,
                state["root_pose"][env_ids, :3],
                state["root_pose"][env_ids, 3:7],
            )
            object_poses[name] = PoseUtils.make_pose(
                position, PoseUtils.matrix_from_quat(quaternion)
            )
        return object_poses
