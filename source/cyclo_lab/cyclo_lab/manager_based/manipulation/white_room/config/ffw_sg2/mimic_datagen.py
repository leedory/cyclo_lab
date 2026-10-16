"""Task000100 trajectory stitching for a fixed home pose and randomized objects."""

from copy import deepcopy
import os

import torch

from isaaclab.utils import math as math_utils
from isaaclab_mimic.datagen.data_generator import DataGenerator
from isaaclab_mimic.datagen.waypoint import WaypointSequence


TASK000300_PREGRASP_SETTLE_STEPS = 8
TASK000300_PRERELEASE_SETTLE_STEPS = 16
TASK000300_MAX_RELEASE_CENTER_CORRECTION_M = 0.20
TASK000300_RETURN_EEF_CLEARANCE_M = 0.060
# The measured Task000300 IK/grasp execution leaves the bottle consistently
# ahead of the Cartesian waypoint in the robot-root frame.  Compensate that
# fixed execution bias once, as part of the feed-forward retargeting; do not
# feed the measured error back into later waypoints (which caused overshoot).
TASK000300_RELEASE_EXECUTION_BIAS_XY_M = (0.001, 0.035)
TASK000300_RELEASE_EXECUTION_BIAS_BY_SOURCE = {
    1: (-0.002, 0.050),
    3: (0.015, 0.018),
}


def _stable_home_start_index(positions: torch.Tensor, tolerance_m: float = 0.002) -> int:
    """Return the first index of the final contiguous home-pose tail."""
    if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) < 2:
        raise ValueError("Task000300 return phase needs at least two 3D EEF positions")
    if tolerance_m <= 0.0:
        raise ValueError("Task000300 home-tail tolerance must be positive")

    distance_to_home = torch.linalg.vector_norm(positions - positions[-1], dim=1)
    outside_home = torch.nonzero(distance_to_home > tolerance_m).flatten()
    if len(outside_home) == 0:
        return 0
    return min(int(outside_home[-1].item()) + 1, len(positions) - 1)


def _gripper_value(waypoint) -> float:
    """Return one scalar gripper command from a Mimic waypoint."""
    value = waypoint.gripper_action.reshape(-1)[0]
    if not bool(torch.isfinite(value).item()):
        raise ValueError("Task000300 gripper action is not finite")
    return float(value.item())


def _first_gripper_close_index(trajectory, threshold: float = 0.5) -> int:
    """Return the first source waypoint whose gripper command is closed."""
    for index in range(len(trajectory)):
        if _gripper_value(trajectory[index]) >= threshold:
            return index
    raise ValueError("Task000300 grasp phase has no gripper-close waypoint")


def add_pregrasp_settle(
    trajectory, settle_steps: int = TASK000300_PREGRASP_SETTLE_STEPS
):
    """Hold the final open-gripper grasp pose before the source starts closing."""
    if settle_steps < 1:
        raise ValueError("Task000300 pregrasp settle steps must be positive")
    close_index = _first_gripper_close_index(trajectory)
    if close_index < 1:
        raise ValueError("Task000300 grasp phase closes before an open waypoint")
    waypoints = [deepcopy(trajectory[index]) for index in range(len(trajectory))]
    hold = waypoints[close_index - 1]
    waypoints[close_index:close_index] = [
        deepcopy(hold) for _ in range(settle_steps)
    ]
    trajectory.waypoint_sequences = [WaypointSequence(sequence=waypoints)]
    return trajectory


def _carry_release_start(trajectory) -> int:
    """Return the first waypoint after the stable closed-gripper interval."""
    if len(trajectory) < 10:
        raise ValueError("Task000300 carry phase is too short for clearance shaping")
    values = torch.tensor(
        [_gripper_value(trajectory[index]) for index in range(len(trajectory))],
        dtype=torch.float32,
    )
    peak = torch.max(values)
    if float(peak.item()) < 0.5:
        raise ValueError("Task000300 carry phase has no closed-gripper interval")

    closed = torch.nonzero(values >= 0.95 * peak).flatten()
    if len(closed) == 0:
        raise ValueError("Task000300 carry phase has no stable closed-gripper interval")
    release_start = int(closed[-1].item()) + 1
    if release_start >= len(trajectory):
        raise ValueError("Task000300 carry phase has no gripper-opening transition")
    return release_start


def _carry_clearance_offsets(trajectory, clearance_m: float = 0.012) -> tuple[list[float], int]:
    """Raise the carry phase and keep the clearance through release."""
    release_start = _carry_release_start(trajectory)

    ramp_up_end = min(max(1, int(round(release_start * 0.20))), release_start - 2)

    offsets = []
    for index in range(len(trajectory)):
        if index <= ramp_up_end:
            fraction = index / ramp_up_end
            lift_m = clearance_m * (3.0 * fraction**2 - 2.0 * fraction**3)
        else:
            # Do not descend while opening. The next subtask is anchored from
            # the actual post-release pose and returns to the seed home pose.
            lift_m = clearance_m
        offsets.append(lift_m)
    return offsets, release_start


def center_release_on_tray(
    trajectory,
    current_eef_pose: torch.Tensor,
    bottle_pose: torch.Tensor,
    tray_pose: torch.Tensor,
    release_start: int,
    env_id: int,
    source_demo_index: int,
    execution_bias_xy_m: tuple[float, float],
):
    """Blend the carry target so the grasped bottle reaches the tray center."""
    if not 0 < release_start < len(trajectory):
        raise ValueError("Task000300 release index is outside the carry trajectory")
    bottle_in_eef = torch.linalg.inv(current_eef_pose) @ bottle_pose
    predicted_release_bottle = trajectory[release_start].pose @ bottle_in_eef
    geometric_correction_xy = (
        tray_pose[:2, 3] - predicted_release_bottle[:2, 3]
    )
    execution_bias_xy = torch.tensor(
        execution_bias_xy_m,
        dtype=geometric_correction_xy.dtype,
        device=geometric_correction_xy.device,
    )
    correction_xy = geometric_correction_xy + execution_bias_xy
    if not bool(torch.isfinite(correction_xy).all().item()):
        raise ValueError("Task000300 tray-center correction is not finite")
    if (
        float(torch.linalg.vector_norm(correction_xy).item())
        > TASK000300_MAX_RELEASE_CENTER_CORRECTION_M
    ):
        raise ValueError("Task000300 tray-center correction exceeds 20 cm")
    if os.environ.get("CYCLO_TASK000300_LOG_OUTCOME") == "1":
        print(
            f"TASK000300_FEEDFORWARD_CENTER env={env_id} "
            f"source_demo={source_demo_index} "
            f"geometric_xy_m={geometric_correction_xy.tolist()} "
            f"execution_bias_xy_m={execution_bias_xy.tolist()} "
            f"correction_xy_m={correction_xy.tolist()}",
            flush=True,
        )

    # Keep the seed's height, orientation, and gripper timing, but make the
    # carry XY a single smooth path from the measured grasp pose to the
    # corrected release pose. Retaining the source XY wiggles after retargeting
    # made some trajectories cross the tray center and then return.
    start_xy = trajectory[0].pose[:2, 3].clone()
    release_target_xy = (
        trajectory[release_start].pose[:2, 3].clone() + correction_xy
    )
    for index in range(len(trajectory)):
        fraction = min(index / release_start, 1.0)
        weight = 3.0 * fraction**2 - 2.0 * fraction**3
        waypoint = trajectory[index]
        waypoint.pose = waypoint.pose.clone()
        waypoint.pose[:2, 3] = start_xy + (
            release_target_xy - start_xy
        ) * weight
    return trajectory


def add_prerelease_settle(
    trajectory,
    release_start: int,
    settle_steps: int = TASK000300_PRERELEASE_SETTLE_STEPS,
):
    """Hold the final closed, centered carry pose before opening the jaws."""
    if settle_steps < 1:
        raise ValueError("Task000300 prerelease settle steps must be positive")
    if not 0 < release_start < len(trajectory):
        raise ValueError("Task000300 release index is outside the carry trajectory")
    waypoints = [deepcopy(trajectory[index]) for index in range(len(trajectory))]
    hold = waypoints[release_start - 1]
    waypoints[release_start:release_start] = [
        deepcopy(hold) for _ in range(settle_steps)
    ]
    trajectory.waypoint_sequences = [WaypointSequence(sequence=waypoints)]
    return trajectory


def keep_return_above_tray(
    trajectory, tray_pose: torch.Tensor,
    clearance_m: float = TASK000300_RETURN_EEF_CLEARANCE_M,
):
    """Keep the open gripper above the tray until its return path clears it."""
    minimum_z = tray_pose[2, 3] + clearance_m
    for index in range(len(trajectory)):
        waypoint = trajectory[index]
        if waypoint.pose[2, 3] < minimum_z:
            waypoint.pose = waypoint.pose.clone()
            waypoint.pose[2, 3] = minimum_z
    return trajectory


def anchor_segment_to_current_pose(trajectory, current_pose, subtask_index: int):
    """Remove a phase-start jump, then blend back to the retargeted endpoint.

    Generic MimicGen transforms every waypoint in an object-referenced phase.
    Its first waypoint therefore moves with the randomized block or basket,
    even though this task's robot always starts from the same joint pose. At a
    carry transition, the basket transform can relocate the first carry
    waypoint away from the grasp pose. Keep the first target at the measured
    current EEF pose and fade that correction out before the phase endpoint.
    """
    count = len(trajectory)
    if count < 2:
        raise ValueError("Task000100 phase needs at least two EEF waypoints")
    positions = torch.stack([trajectory[index].pose[:3, 3] for index in range(count)])
    correction = current_pose[:3, 3] - positions[0]
    if not bool(torch.isfinite(correction).all().item()):
        raise ValueError("Task000100 EEF anchor correction is not finite")
    if float(torch.linalg.vector_norm(correction).item()) > 0.35:
        raise ValueError("Task000100 phase-start EEF jump exceeds 35 cm")

    # Preserve the seed's initial stationary frames. The first phase's source
    # arm does not move until the human starts the reach; randomization should
    # not turn those frames into an artificial backward approach.
    if subtask_index == 0:
        moved = torch.nonzero(torch.linalg.vector_norm(positions - positions[0], dim=1) > 0.005)
        fade_start = min(int(moved[0].item()), count - 2) if len(moved) else 0
    else:
        fade_start = 0

    # Match the object-relative source grasp by removing the start correction
    # before the first close command. This preserves the seed's approach and
    # gripper timing while avoiding a residual offset at contact.
    if subtask_index == 0:
        fade_end = _first_gripper_close_index(trajectory) - TASK000300_PREGRASP_SETTLE_STEPS
        if fade_end <= fade_start:
            raise ValueError("Task000300 grasp phase closes before anchor fading can start")
    # The final return phase in the denoised Task000300 seeds already contains
    # a stationary home tail. Stop fading the phase-start correction when that
    # recorded tail begins; otherwise the correction keeps moving the arm
    # during frames that should remain at home. This changes no waypoint count
    # or gripper timing and leaves the recorded tail poses untouched.
    elif subtask_index == 1:
        # A full-segment XYZ anchor leaves a residual Z correction at release
        # and made generated hands much lower than the seed. Restore the
        # object-relative seed pose by the middle of the closed carry phase.
        release_start = _carry_release_start(trajectory)
        fade_end = max(1, min(release_start - 1, int(round(release_start * 0.50))))
    elif subtask_index == 2:
        fade_end = _stable_home_start_index(positions)
        if fade_end <= fade_start:
            raise ValueError("Task000300 return phase has no motion before its stable home tail")
    else:
        fade_end = count - 1

    orientation_slerp_enabled = (
        os.environ.get("CYCLO_TASK000300_PHASE_ORIENTATION_SLERP") == "1"
    )
    if orientation_slerp_enabled:
        source_start_rotation = trajectory[0].pose[:3, :3]
        orientation_correction = (
            current_pose[:3, :3] @ source_start_rotation.transpose(0, 1)
        )
        correction_quaternion = math_utils.quat_from_matrix(
            orientation_correction
        )
        identity_quaternion = torch.zeros_like(correction_quaternion)
        identity_quaternion[0] = 1.0
        correction_angle_deg = float(
            torch.rad2deg(
                2.0
                * torch.acos(
                    torch.clamp(correction_quaternion[0].abs(), 0.0, 1.0)
                )
            ).item()
        )
        print(
            "TASK000300_PHASE_ORIENTATION_SLERP "
            f"subtask={subtask_index} fade_start={fade_start} "
            f"fade_end={fade_end} correction_deg={correction_angle_deg:.6g}",
            flush=True,
        )

    for index in range(count):
        if index <= fade_start:
            weight = 1.0
        elif index >= fade_end:
            weight = 0.0
        else:
            fraction = (index - fade_start) / (fade_end - fade_start)
            weight = 1.0 - (3.0 * fraction**2 - 2.0 * fraction**3)
        waypoint = trajectory[index]
        waypoint.pose = waypoint.pose.clone()
        waypoint.pose[:3, 3] += correction * weight
        if orientation_slerp_enabled:
            correction_step_quaternion = math_utils.quat_slerp(
                identity_quaternion,
                correction_quaternion.clone(),
                tau=float(weight),
            )
            correction_step_rotation = math_utils.matrix_from_quat(
                correction_step_quaternion
            )
            waypoint.pose[:3, :3] = (
                correction_step_rotation @ waypoint.pose[:3, :3]
            )
    return trajectory


class Task000100DataGenerator(DataGenerator):
    """Use the source motion while making each transformed phase continuous."""

    def generate_eef_subtask_trajectory(
        self, env_id, eef_name, subtask_ind, all_randomized_subtask_boundaries,
        runtime_subtask_constraints_dict, selected_src_demo_inds,
    ):
        trajectory = super().generate_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_ind=subtask_ind,
            all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
            selected_src_demo_inds=selected_src_demo_inds,
        )
        current_pose = self.env.get_robot_eef_pose(eef_name=eef_name, env_ids=[env_id])[0]
        return anchor_segment_to_current_pose(trajectory, current_pose, subtask_ind)


class Task000300DataGenerator(Task000100DataGenerator):
    """Keep the grasped bottle above the 2 cm tray rim through release."""

    def generate_eef_subtask_trajectory(
        self, env_id, eef_name, subtask_ind, all_randomized_subtask_boundaries,
        runtime_subtask_constraints_dict, selected_src_demo_inds,
    ):
        trajectory = super().generate_eef_subtask_trajectory(
            env_id=env_id,
            eef_name=eef_name,
            subtask_ind=subtask_ind,
            all_randomized_subtask_boundaries=all_randomized_subtask_boundaries,
            runtime_subtask_constraints_dict=runtime_subtask_constraints_dict,
            selected_src_demo_inds=selected_src_demo_inds,
        )
        if subtask_ind == 0:
            trajectory = add_pregrasp_settle(trajectory)
        elif subtask_ind == 1:
            source_demo_index = int(selected_src_demo_inds[eef_name])
            execution_bias_xy_m = (
                TASK000300_RELEASE_EXECUTION_BIAS_BY_SOURCE.get(
                    source_demo_index,
                    TASK000300_RELEASE_EXECUTION_BIAS_XY_M,
                )
            )
            lift_offsets, release_start = _carry_clearance_offsets(trajectory)
            for index, lift_m in enumerate(lift_offsets):
                waypoint = trajectory[index]
                waypoint.pose = waypoint.pose.clone()
                waypoint.pose[2, 3] += lift_m
            current_eef_pose = self.env.get_robot_eef_pose(
                eef_name=eef_name, env_ids=[env_id]
            )[0]
            object_poses = self.env.get_object_poses([env_id])
            trajectory = center_release_on_tray(
                trajectory,
                current_eef_pose=current_eef_pose,
                bottle_pose=object_poses["navy_block"][0],
                tray_pose=object_poses["white_basket"][0],
                release_start=release_start,
                env_id=env_id,
                source_demo_index=source_demo_index,
                execution_bias_xy_m=execution_bias_xy_m,
            )
            trajectory = add_prerelease_settle(
                trajectory, release_start=release_start
            )
        elif subtask_ind == 2:
            tray_pose = self.env.get_object_poses([env_id])["white_basket"][0]
            trajectory = keep_return_above_tray(
                trajectory, tray_pose=tray_pose
            )
        return trajectory

    async def generate(self, env_id, success_term, **kwargs):
        """Export only the final state result, not MimicGen's sticky OR result."""
        export_demo = kwargs.pop("export_demo", True)
        result = await super().generate(
            env_id=env_id,
            success_term=success_term,
            export_demo=False,
            **kwargs,
        )
        final_success = self.env.get_generation_success(env_id, success_term)
        result["success"] = final_success
        env_ids = torch.tensor([env_id], dtype=torch.int64, device=self.env.device)
        self.env.recorder_manager.set_success_to_episodes(
            env_ids,
            torch.tensor(
                [[final_success]], dtype=torch.bool, device=self.env.device
            ),
        )
        if export_demo:
            self.env.recorder_manager.export_episodes(env_ids)
        return result
