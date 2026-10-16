"""Isaac Lab Mimic adapter for WhiteRoom Task000100."""

from __future__ import annotations

import os
import math
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch

from cyclo_lab.manager_based.manipulation.common import FFWSG2MimicEnv

from . import mimic_terms, task_000300_mimic_terms
from .contact_failure import contact_failures, reset_contact_failures
from .gripper_tcp import CLOSED_REFERENCE_TCP_Z_M, gripper_dependent_tcp_z


# Calibrated to include all four clean source grasps while rejecting the large
# pre-grasp pushes seen in generated data.  The axial limit is approximately
# half the 53 mm bottle width, i.e. the bottle center must be inside the jaws.
TASK000300_CLOSE_REQUEST_THRESHOLD = 0.20
TASK000300_CLOSE_RADIAL_MAX_M = 0.023
TASK000300_CLOSE_AXIAL_MAX_M = 0.027
TASK000300_PREGRASP_DISPLACEMENT_MAX_M = 0.015
TASK000300_PREGRASP_UPRIGHT_MIN = math.cos(math.radians(10.0))
TASK000300_RELEASE_REQUEST_THRESHOLD = 0.50
TASK000300_FULLY_CLOSED_THRESHOLD = 0.50
TASK000300_RELEASE_CENTER_READY_M = 0.050


class Task000100MimicEnv(FFWSG2MimicEnv):
    """Expose WhiteRoom rigid objects in the randomized robot-root frame."""

    def get_generation_success(self, env_id: int, success_term) -> bool:
        """Use final containment and latched forbidden contacts for the label."""
        success = bool(mimic_terms.task_success(self, self.cfg.mimic_side)[env_id].item())
        if os.environ.get("CYCLO_TASK000100_LOG_OUTCOME") == "1":
            table, basket = contact_failures(self)
            delta = self.scene["navy_block"].data.root_pos_w[env_id] - self.scene["white_basket"].data.root_pos_w[env_id]
            print(
                f"TASK000100_OUTCOME env={env_id} "
                f"status={'ACCEPTED' if success else 'REJECTED'}"
                f" success={int(success)} table_contact={int(table[env_id].item())}"
                f" gripper_basket_contact={int(basket[env_id].item())}"
                f" block_basket_delta_m={delta.tolist()}"
                f" table_peak_n={self._task000100_table_peak_force_n[env_id].item():.6g}"
                f" basket_peak_n={self._task000100_basket_peak_force_n[env_id].item():.6g}",
                flush=True,
            )
        return success

    def update_generation_success_tracking(self, env_id: int, success_term) -> bool:
        """Sample contact history every action so early collisions remain failures."""
        return bool(mimic_terms.task_success(self, self.cfg.mimic_side)[env_id].item())

    def _finish_episode_reset(self, env_ids: Sequence[int] | None) -> None:
        reset_contact_failures(self, env_ids)
        mimic_terms.reset_metric_state(self, env_ids)
        self.obs_buf = self.observation_manager.compute(update_history=True)

    def reset(self, seed=None, env_ids=None, options=None):
        _, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        self._finish_episode_reset(env_ids)
        return self.obs_buf, extras

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        _, extras = super().reset_to(
            state, env_ids, seed=seed, is_relative=is_relative
        )
        self._finish_episode_reset(env_ids)
        return self.obs_buf, extras

    def get_object_poses(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        scene_state = self.scene.get_state(is_relative=True)
        robot_root = scene_state["articulation"]["robot"]["root_pose"]
        root_pos = robot_root[env_ids, :3]
        root_quat = robot_root[env_ids, 3:7]
        object_poses = {}
        for name, object_state in scene_state["rigid_object"].items():
            position, quaternion = PoseUtils.subtract_frame_transforms(
                root_pos,
                root_quat,
                object_state["root_pose"][env_ids, :3],
                object_state["root_pose"][env_ids, 3:7],
            )
            object_poses[name] = PoseUtils.make_pose(
                position, PoseUtils.matrix_from_quat(quaternion)
            )
        return object_poses


class Task000300MimicEnv(Task000100MimicEnv):
    """Use the upright bottle and shallow tray outcome for Task000300."""

    def _task000300_reset_close_gate(
        self, env_ids: Sequence[int] | None
    ) -> None:
        if not hasattr(self, "_task000300_close_permitted"):
            self._task000300_close_permitted = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._task000300_initial_bottle_xy = torch.zeros(
                (self.num_envs, 2), dtype=torch.float32, device=self.device
            )
            self._task000300_close_block_logged = torch.zeros_like(
                self._task000300_close_permitted
            )
            self._task000300_release_permitted = torch.zeros_like(
                self._task000300_close_permitted
            )
            self._task000300_fully_closed_seen = torch.zeros_like(
                self._task000300_close_permitted
            )
            self._task000300_release_block_logged = torch.zeros_like(
                self._task000300_close_permitted
            )
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            ).reshape(-1)
        )
        self._task000300_close_permitted[ids] = False
        self._task000300_close_block_logged[ids] = False
        self._task000300_release_permitted[ids] = False
        self._task000300_fully_closed_seen[ids] = False
        self._task000300_release_block_logged[ids] = False
        self._task000300_initial_bottle_xy[ids] = self.scene[
            "navy_block"
        ].data.root_pos_w[ids, :2]

    def _task000300_release_error(
        self, env_id: int
    ) -> tuple[torch.Tensor, float]:
        """Return actual bottle-to-tray XY error without accumulating it."""
        object_poses = self.get_object_poses([env_id])
        error_xy = (
            object_poses["white_basket"][0, :2, 3]
            - object_poses["navy_block"][0, :2, 3]
        )
        distance_m = float(torch.linalg.vector_norm(error_xy).item())
        return error_xy, distance_m

    def _task000300_close_readiness(
        self, eef_name: str, env_id: int
    ) -> tuple[bool, dict[str, float]]:
        eef_pose = self.get_robot_eef_pose(eef_name, [env_id])[0]
        bottle_pose = self.get_object_poses([env_id])["navy_block"][0]
        bottle_in_eef = eef_pose[:3, :3].transpose(0, 1) @ (
            bottle_pose[:3, 3] - eef_pose[:3, 3]
        )
        radial_m = torch.linalg.vector_norm(bottle_in_eef[:2])
        axial_abs_m = bottle_in_eef[2].abs()
        bottle_xy = self.scene["navy_block"].data.root_pos_w[env_id, :2]
        displacement_m = torch.linalg.vector_norm(
            bottle_xy - self._task000300_initial_bottle_xy[env_id]
        )
        upright = bottle_pose[2, 2]
        ready = bool(
            (
                (radial_m <= TASK000300_CLOSE_RADIAL_MAX_M)
                & (axial_abs_m <= TASK000300_CLOSE_AXIAL_MAX_M)
                & (
                    displacement_m
                    <= TASK000300_PREGRASP_DISPLACEMENT_MAX_M
                )
                & (upright >= TASK000300_PREGRASP_UPRIGHT_MIN)
            ).item()
        )
        diagnostics = {
            "radial_m": float(radial_m.item()),
            "axial_abs_m": float(axial_abs_m.item()),
            "displacement_m": float(displacement_m.item()),
            "upright": float(upright.item()),
        }
        return ready, diagnostics

    def _dynamic_eef_vector(self, eef_name: str, env_ids=None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        side = "r" if "right" in eef_name.lower() else "l"
        fixed_pose = self.obs_buf["policy"][f"{'right' if side == 'r' else 'left'}_eef_pose"][env_ids]
        gripper_index = self.scene["robot"].joint_names.index(f"gripper_{side}_joint1")
        gripper = self.scene["robot"].data.joint_pos[env_ids, gripper_index]
        dynamic_z = gripper_dependent_tcp_z(gripper)
        rotation = PoseUtils.matrix_from_quat(fixed_pose[:, 3:7])
        local_z_world = rotation[:, :, 2]
        result = fixed_pose.clone()
        result[:, :3] += local_z_world * (dynamic_z - CLOSED_REFERENCE_TCP_Z_M).unsqueeze(-1)
        return result

    def get_robot_eef_pose(self, eef_name: str, env_ids=None) -> torch.Tensor:
        dynamic_pose = self._dynamic_eef_vector(eef_name, env_ids)
        return PoseUtils.make_pose(
            dynamic_pose[:, :3], PoseUtils.matrix_from_quat(dynamic_pose[:, 3:7])
        )

    def target_eef_pose_to_action(
        self, target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0
    ) -> torch.Tensor:
        """Gate first closure on actual bottle containment, then convert the TCP."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        dynamic_target = target_eef_pose_dict[eef_name].clone()
        requested_gripper = gripper_action_dict[eef_name].reshape(-1)[0]
        gated_gripper_actions = dict(gripper_action_dict)
        if (
            float(requested_gripper.item())
            >= TASK000300_CLOSE_REQUEST_THRESHOLD
            and not bool(self._task000300_close_permitted[env_id].item())
        ):
            ready, diagnostics = self._task000300_close_readiness(
                eef_name, env_id
            )
            if ready:
                self._task000300_close_permitted[env_id] = True
                if os.environ.get("CYCLO_TASK000300_LOG_OUTCOME") == "1":
                    print(
                        f"TASK000300_CLOSE_PERMITTED env={env_id} "
                        f"sim_step={self._sim_step_counter} {diagnostics}",
                        flush=True,
                    )
            else:
                held_open = gripper_action_dict[eef_name].clone()
                held_open.zero_()
                gated_gripper_actions[eef_name] = held_open
                if (
                    os.environ.get("CYCLO_TASK000300_LOG_OUTCOME") == "1"
                    and not bool(
                        self._task000300_close_block_logged[env_id].item()
                    )
                ):
                    self._task000300_close_block_logged[env_id] = True
                    print(
                        f"TASK000300_CLOSE_BLOCKED env={env_id} "
                        f"sim_step={self._sim_step_counter} {diagnostics}",
                        flush=True,
                    )
        if (
            bool(self._task000300_close_permitted[env_id].item())
            and float(requested_gripper.item())
            >= TASK000300_FULLY_CLOSED_THRESHOLD
        ):
            self._task000300_fully_closed_seen[env_id] = True
        release_requested = (
            bool(self._task000300_close_permitted[env_id].item())
            and bool(self._task000300_fully_closed_seen[env_id].item())
            and float(requested_gripper.item())
            < TASK000300_RELEASE_REQUEST_THRESHOLD
        )
        if (
            release_requested
            and not bool(self._task000300_release_permitted[env_id].item())
        ):
            center_error_xy, center_distance_m = (
                self._task000300_release_error(env_id)
            )
            if center_distance_m <= TASK000300_RELEASE_CENTER_READY_M:
                self._task000300_release_permitted[env_id] = True
                if os.environ.get("CYCLO_TASK000300_LOG_OUTCOME") == "1":
                    print(
                        f"TASK000300_RELEASE_PERMITTED env={env_id} "
                        f"sim_step={self._sim_step_counter} "
                        f"center_error_xy_m={center_error_xy.tolist()} "
                        f"center_distance_m={center_distance_m:.6g}",
                        flush=True,
                    )
            else:
                held_closed = gripper_action_dict[eef_name].clone()
                held_closed.fill_(1.0)
                gated_gripper_actions[eef_name] = held_closed
                if (
                    os.environ.get("CYCLO_TASK000300_LOG_OUTCOME") == "1"
                    and not bool(
                        self._task000300_release_block_logged[env_id].item()
                    )
                ):
                    self._task000300_release_block_logged[env_id] = True
                    print(
                        f"TASK000300_RELEASE_BLOCKED env={env_id} "
                        f"sim_step={self._sim_step_counter} "
                        f"center_error_xy_m={center_error_xy.tolist()} "
                        f"center_distance_m={center_distance_m:.6g}",
                        flush=True,
                    )
        gripper = gated_gripper_actions[eef_name].reshape(-1)[0]
        dynamic_z = gripper_dependent_tcp_z(gripper)
        fixed_target = dynamic_target.clone()
        fixed_target[:3, 3] += dynamic_target[:3, 2] * (
            CLOSED_REFERENCE_TCP_Z_M - dynamic_z
        )
        targets = dict(target_eef_pose_dict)
        targets[eef_name] = fixed_target
        return super().target_eef_pose_to_action(
            targets,
            gated_gripper_actions,
            action_noise_dict=action_noise_dict,
            env_id=env_id,
        )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Expose fixed IK action targets back in the dynamic grasp-center frame."""
        targets = super().action_to_target_eef_pose(action)
        eef_name = list(targets)[0]
        gripper_index = 15 if "right" in eef_name.lower() else 7
        dynamic_z = gripper_dependent_tcp_z(action[:, gripper_index])
        dynamic_target = targets[eef_name].clone()
        dynamic_target[:, :3, 3] += dynamic_target[:, :3, 2] * (
            dynamic_z - CLOSED_REFERENCE_TCP_Z_M
        ).unsqueeze(-1)
        return {eef_name: dynamic_target}

    def _finish_episode_reset(self, env_ids: Sequence[int] | None) -> None:
        super()._finish_episode_reset(env_ids)
        task_000300_mimic_terms.reset_metric_state(self, env_ids)
        self._task000300_reset_close_gate(env_ids)
        self._task000300_contact_logged = False

    def get_generation_success(self, env_id: int, success_term) -> bool:
        success = bool(task_000300_mimic_terms.task_success(self, self.cfg.mimic_side)[env_id].item())
        if os.environ.get("CYCLO_TASK000300_LOG_OUTCOME") == "1":
            table, tray = contact_failures(self)
            delta = self.scene["navy_block"].data.root_pos_w[env_id] - self.scene["white_basket"].data.root_pos_w[env_id]
            print(
                f"TASK000300_OUTCOME env={env_id} "
                f"status={'ACCEPTED' if success else 'REJECTED'}"
                f" clean_close={int(self._task000300_close_permitted[env_id].item())}"
                f" table_contact={int(table[env_id].item())}"
                f" gripper_tray_contact={int(tray[env_id].item())}"
                f" bottle_tray_delta_m={delta.tolist()}"
                f" bottle_tray_center_xy_m={torch.linalg.vector_norm(delta[:2]).item():.6g}"
                f" table_peak_n={self._task000100_table_peak_force_n[env_id].item():.6g}"
                f" tray_peak_n={self._task000100_basket_peak_force_n[env_id].item():.6g}",
                flush=True,
            )
        return success

    def update_generation_success_tracking(self, env_id: int, success_term) -> bool:
        success = bool(task_000300_mimic_terms.task_success(self, self.cfg.mimic_side)[env_id].item())
        if os.environ.get("CYCLO_TASK000300_LOG_OUTCOME") == "1":
            _, tray_contact = contact_failures(self)
            if bool(tray_contact[env_id].item()) and not getattr(self, "_task000300_contact_logged", False):
                self._task000300_contact_logged = True
                bottle = self.scene["navy_block"].data.root_pos_w[env_id]
                tray = self.scene["white_basket"].data.root_pos_w[env_id]
                print(
                    f"TASK000300_FIRST_TRAY_CONTACT sim_step={self._sim_step_counter} "
                    f"bottle_tray_delta_m={(bottle - tray).tolist()} "
                    f"peak_n={self._task000100_basket_peak_force_n[env_id].item():.6g}",
                    flush=True,
                )
        return success
