"""One local SG2 TCP IK estimate using Cyclo Lab's Isaac Lab DLS controller.

This does not step or move the robot. The Jacobian is valid at the measured
current pose, so distant targets are rejected instead of reported as solved.
"""

from __future__ import annotations

import torch

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils import math as math_utils

from seed_plan import IK_SOLVER_LABELS, joint_contract


MAX_LOCAL_DISTANCE_M = 0.50
TCP_OFFSET_M = (0.0, 0.0, -0.2)  # existing SG2 left_eef/right_eef and Mimic IK offset


def _estimate_local_dls(env, side: str, xyz_w: list[float]) -> dict:
    """Return a nearby 7-joint position candidate without changing env state."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    sensor_name = f"{side}_eef"
    if sensor_name not in env.scene.sensors:
        raise ValueError(f"Selected env has no {sensor_name} TCP sensor")
    robot = env.scene["robot"]
    sensor = env.scene[sensor_name]
    current_xyz_w = sensor.data.target_pos_w[0:1, 0, :]
    current_quat_w = sensor.data.target_quat_w[0:1, 0, :]
    target_xyz_w = torch.tensor(xyz_w, device=current_xyz_w.device, dtype=current_xyz_w.dtype).reshape(1, 3)
    distance = torch.linalg.vector_norm(target_xyz_w - current_xyz_w).item()
    if distance > MAX_LOCAL_DISTANCE_M:
        raise ValueError(
            f"The target is {distance:.3f} m from the current TCP. This local IK preview accepts "
            f"targets within {MAX_LOCAL_DISTANCE_M:.2f} m. Add a closer intermediate point."
        )

    short = "l" if side == "left" else "r"
    joint_ids, joint_names = robot.find_joints([f"arm_{short}_joint[1-7]"], preserve_order=True)
    body_ids, _ = robot.find_bodies(f"arm_{short}_link7")
    if len(joint_ids) != 7 or len(body_ids) != 1:
        raise ValueError("Selected robot does not expose the expected SG2 7D arm/link7")
    body_index = body_ids[0] - 1 if robot.is_fixed_base else body_ids[0]
    jacobian_joint_ids = joint_ids if robot.is_fixed_base else [index + 6 for index in joint_ids]
    full_jacobian_w = robot.root_physx_view.get_jacobians()[0:1, body_index, :, :]
    indices = torch.tensor(jacobian_joint_ids, device=full_jacobian_w.device, dtype=torch.long)
    jacobian_w = full_jacobian_w.index_select(-1, indices).clone()
    root_quat_w = robot.data.root_quat_w[0:1]
    root_pos_w = robot.data.root_pos_w[0:1]
    rotation_b_w = math_utils.matrix_from_quat(math_utils.quat_inv(root_quat_w))
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, :3, :] = torch.bmm(rotation_b_w, jacobian_w[:, :3, :])
    jacobian_b[:, 3:, :] = torch.bmm(rotation_b_w, jacobian_w[:, 3:, :])
    offset = torch.tensor(TCP_OFFSET_M, device=jacobian_b.device, dtype=jacobian_b.dtype).reshape(1, 3)
    # Same TCP offset correction as DifferentialInverseKinematicsAction.
    jacobian_b[:, :3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset), jacobian_b[:, 3:, :])
    current_xyz_b, current_quat_b = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, current_xyz_w, current_quat_w
    )
    target_xyz_b, _ = math_utils.subtract_frame_transforms(
        root_pos_w, root_quat_w, target_xyz_w, current_quat_w
    )
    controller = DifferentialIKController(
        DifferentialIKControllerCfg(
            command_type="pose", ik_method="dls", ik_params={"lambda_val": 0.05}, use_relative_mode=False
        ),
        num_envs=1,
        device=robot.device,
    )
    controller.set_command(torch.cat((target_xyz_b, current_quat_b), dim=1))
    joint_pos = robot.data.joint_pos[0:1, joint_ids]
    raw_candidate = controller.compute(current_xyz_b, current_quat_b, jacobian_b, joint_pos)
    if not bool(torch.isfinite(raw_candidate).all().item()):
        raise ValueError("IK produced a non-finite joint candidate")
    # The selected env's reset pose can intentionally sit outside Isaac Lab's
    # contracted soft range. UI q_des input is governed by the SG2 URDF limits,
    # so use the same contract for candidates instead of rejecting the reset pose.
    _, urdf_limits = joint_contract()
    raw_delta = raw_candidate - joint_pos
    joint_step_scale = 1.0
    for index, name in enumerate(joint_names):
        lower, upper = urdf_limits[name]
        current_value = float(joint_pos[0, index].item())
        delta_value = float(raw_delta[0, index].item())
        if delta_value > 0:
            joint_step_scale = min(joint_step_scale, max(0.0, (upper - current_value) / delta_value))
        elif delta_value < 0:
            joint_step_scale = min(joint_step_scale, max(0.0, (lower - current_value) / delta_value))
    joint_step_scale = min(1.0, joint_step_scale)
    candidate = joint_pos + joint_step_scale * raw_delta
    for index, name in enumerate(joint_names):
        lower, upper = urdf_limits[name]
        candidate[:, index].clamp_(lower, upper)
    delta = candidate - joint_pos
    predicted_delta = torch.bmm(jacobian_b[:, :3, :], delta.unsqueeze(-1)).squeeze(-1)
    linear_residual = torch.linalg.vector_norm(target_xyz_b - current_xyz_b - predicted_delta).item()
    limited = joint_step_scale < 1.0 - 1.0e-6
    candidate_values = candidate[0].cpu().tolist()
    current_values = joint_pos[0].cpu().tolist()
    margins = [
        min(value - urdf_limits[name][0], urdf_limits[name][1] - value)
        for name, value in zip(joint_names, candidate_values)
    ]
    return {
        "side": side,
        "solver": "local_dls",
        "solver_label": IK_SOLVER_LABELS["local_dls"],
        "frame": "world",
        "target_xyz_m": target_xyz_w[0].cpu().tolist(),
        "current_tcp_xyz_m": current_xyz_w[0].cpu().tolist(),
        "current_orientation_held": True,
        "distance_m": distance,
        "converged": None,
        "iterations": 1,
        "position_residual_m": linear_residual,
        "orientation_residual_rad": None,
        "linear_residual_m": linear_residual,
        "joint_step_scale": joint_step_scale,
        "urdf_limited": limited,
        "max_joint_delta_rad": max(
            abs(value - current) for value, current in zip(candidate_values, current_values)
        ),
        "min_joint_limit_margin_rad": min(margins),
        "reference_posture": None,
        "q_des": dict(zip(joint_names, candidate_values)),
        "note": (
            "This q_des candidate uses the current-pose Jacobian. Joint deltas were scaled to fit URDF limits. "
            "Calculate the next candidate from the pose held after Preview Motion."
            if limited else
            "This q_des candidate uses the current-pose Jacobian. Check the measured TCP after Preview Motion."
        ),
    }


def estimate_q_des(
    env, side: str, xyz_w: list[float], solver: str = "iterative_lm"
) -> dict:
    """Dispatch one UI request to the selected side-effect-free solver."""
    if solver not in IK_SOLVER_LABELS:
        raise ValueError(f"Unsupported IK solver: {solver}")
    if solver == "local_dls":
        return _estimate_local_dls(env, side, xyz_w)
    from ik_solvers import estimate_advanced_q_des

    return estimate_advanced_q_des(env, side, xyz_w, solver)
