"""Nonlinear SG2 TCP IK solvers used by the waypoint seed UI."""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
import torch

from isaaclab.utils import math as math_utils

from seed_plan import IK_SOLVER_LABELS, LAB_ROOT, joint_contract


MAX_LOCAL_DISTANCE_M = 0.50
STAGED_SG2_URDF_FILE = LAB_ROOT / ".runtime/joint_waypoint_seed_ui/ffw_sg2_follower.urdf"
MAX_ITERATIONS = 100
MAX_JOINT_STEP_RAD = 0.08
JOINT_LIMIT_MARGIN_RAD = 0.01
POSITION_TOLERANCE_M = 0.003
ORIENTATION_TOLERANCE_RAD = math.radians(2.0)


def _tcp_offset(sensor) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    target_frames = getattr(sensor.cfg, "target_frames", ())
    offset = getattr(target_frames[0], "offset", None) if target_frames else None
    if offset is None:
        return (0.0, 0.0, -0.2), (1.0, 0.0, 0.0, 0.0)
    position = tuple(float(value) for value in offset.pos)
    quaternion = tuple(float(value) for value in offset.rot)
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError("The selected TCP sensor has an invalid frame offset")
    return position, quaternion


def _pin_se3(pin, position, quaternion_wxyz):
    quaternion = pin.Quaternion(*[float(value) for value in quaternion_wxyz])
    return pin.SE3(
        quaternion.matrix(), np.asarray(position, dtype=np.float64).reshape(3)
    )


@lru_cache(maxsize=8)
def _pin_model(
    side: str,
    offset_position: tuple[float, float, float],
    offset_quaternion: tuple[float, float, float, float],
):
    try:
        import pinocchio as pin
    except ImportError as exc:
        raise ValueError("Pinocchio is unavailable in the Isaac runner") from exc
    if not STAGED_SG2_URDF_FILE.is_file():
        raise ValueError(
            "The staged SG2 URDF is missing. Disconnect and reconnect the environment."
        )
    model = pin.buildModelFromUrdf(str(STAGED_SG2_URDF_FILE))
    short = "l" if side == "left" else "r"
    link_frame_id = model.getFrameId(f"arm_{short}_link7")
    if link_frame_id >= len(model.frames):
        raise ValueError("SG2 URDF does not contain the selected arm link7 frame")
    link_frame = model.frames[link_frame_id]
    tcp_offset = _pin_se3(pin, offset_position, offset_quaternion)
    tcp_frame_id = model.addFrame(
        pin.Frame(
            f"waypoint_ui_{side}_tcp",
            link_frame.parent,
            link_frame_id,
            link_frame.placement * tcp_offset,
            pin.FrameType.OP_FRAME,
        )
    )
    return pin, model, tcp_frame_id


def _pin_configuration(pin, model, robot):
    q = pin.neutral(model)
    measured = robot.data.joint_pos[0].detach().cpu().numpy()
    for name, value in zip(robot.joint_names, measured):
        joint_id = model.getJointId(name)
        if joint_id == 0:
            continue
        joint = model.joints[joint_id]
        if joint.nq == 1:
            q[joint.idx_q] = float(value)
        elif joint.nq == 2 and joint.nv == 1:
            q[joint.idx_q : joint.idx_q + 2] = (
                math.cos(float(value)),
                math.sin(float(value)),
            )
    return q


def _evaluate(pin, model, data, q, tcp_frame_id, velocity_indices, target_pose):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    pose = data.oMf[tcp_frame_id]
    error = pin.log6(pose.inverse() * target_pose).vector
    jacobian = pin.computeFrameJacobian(
        model, data, q, tcp_frame_id, pin.ReferenceFrame.LOCAL
    )[:, velocity_indices]
    position_residual = float(np.linalg.norm(target_pose.translation - pose.translation))
    orientation_residual = float(
        np.linalg.norm(pin.log3(pose.rotation.T @ target_pose.rotation))
    )
    objective = position_residual**2 + (0.1 * orientation_residual) ** 2
    return pose, error, jacobian, position_residual, orientation_residual, objective


def _make_context(env, side: str, xyz_w: list[float]) -> dict:
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    sensor_name = f"{side}_eef"
    if sensor_name not in env.scene.sensors:
        raise ValueError(f"Selected env has no {sensor_name} TCP sensor")
    robot = env.scene["robot"]
    sensor = env.scene[sensor_name]
    current_xyz_w = sensor.data.target_pos_w[0:1, 0, :]
    current_quaternion_w = sensor.data.target_quat_w[0:1, 0, :]
    target_xyz_w = torch.tensor(
        xyz_w, device=current_xyz_w.device, dtype=current_xyz_w.dtype
    ).reshape(1, 3)
    distance = float(torch.linalg.vector_norm(target_xyz_w - current_xyz_w).item())
    if distance > MAX_LOCAL_DISTANCE_M:
        raise ValueError(
            f"The target is {distance:.3f} m from the current TCP. Add a closer "
            f"intermediate point within {MAX_LOCAL_DISTANCE_M:.2f} m."
        )
    short = "l" if side == "left" else "r"
    joint_ids, joint_names = robot.find_joints(
        [f"arm_{short}_joint[1-7]"], preserve_order=True
    )
    if len(joint_ids) != 7:
        raise ValueError("Selected robot does not expose the expected SG2 7D arm")
    current_xyz_b, current_quaternion_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w[0:1],
        robot.data.root_quat_w[0:1],
        current_xyz_w,
        current_quaternion_w,
    )
    target_xyz_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w[0:1],
        robot.data.root_quat_w[0:1],
        target_xyz_w,
        current_quaternion_w,
    )
    return {
        "side": side,
        "robot": robot,
        "sensor": sensor,
        "joint_ids": joint_ids,
        "joint_names": tuple(joint_names),
        "current_xyz_w": current_xyz_w,
        "target_xyz_w": target_xyz_w,
        "current_xyz_b": current_xyz_b,
        "current_quaternion_b": current_quaternion_b,
        "target_xyz_b": target_xyz_b,
        "distance": distance,
    }


def estimate_advanced_q_des(env, side: str, xyz_w: list[float], solver: str) -> dict:
    """Solve a nonlinear, posture-biased, or bound-constrained TCP IK request."""
    if solver not in ("iterative_lm", "posture_lm", "constrained_qp"):
        raise ValueError(f"Unsupported advanced IK solver: {solver}")
    context = _make_context(env, side, xyz_w)
    robot = context["robot"]
    offset_position, offset_quaternion = _tcp_offset(context["sensor"])
    pin, model, tcp_frame_id = _pin_model(
        side, offset_position, offset_quaternion
    )
    data = model.createData()
    q = _pin_configuration(pin, model, robot)
    arm_joint_ids = [model.getJointId(name) for name in context["joint_names"]]
    if any(joint_id == 0 for joint_id in arm_joint_ids):
        raise ValueError("The staged SG2 URDF is missing an arm joint")
    position_indices = np.asarray(
        [model.joints[joint_id].idx_q for joint_id in arm_joint_ids], dtype=int
    )
    velocity_indices = np.asarray(
        [model.joints[joint_id].idx_v for joint_id in arm_joint_ids], dtype=int
    )
    reference = q[position_indices].copy()

    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    current_model_pose = data.oMf[tcp_frame_id]
    current_root_pose = _pin_se3(
        pin,
        context["current_xyz_b"][0].detach().cpu().numpy(),
        context["current_quaternion_b"][0].detach().cpu().numpy(),
    )
    # Calibrate the fixed URDF-model frame to Isaac's articulation-root frame
    # from the measured current TCP. This does not mutate or step the env.
    root_from_model = current_root_pose * current_model_pose.inverse()
    target_root_pose = _pin_se3(
        pin,
        context["target_xyz_b"][0].detach().cpu().numpy(),
        context["current_quaternion_b"][0].detach().cpu().numpy(),
    )
    target_model_pose = root_from_model.inverse() * target_root_pose

    _, limits = joint_contract()
    lower = np.asarray(
        [limits[name][0] + JOINT_LIMIT_MARGIN_RAD for name in context["joint_names"]]
    )
    upper = np.asarray(
        [limits[name][1] - JOINT_LIMIT_MARGIN_RAD for name in context["joint_names"]]
    )
    damping = 0.05
    touched_limit = False
    converged = False
    iteration = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        _, error, jacobian, pos_error, rot_error, objective = _evaluate(
            pin, model, data, q, tcp_frame_id, velocity_indices, target_model_pose
        )
        if pos_error <= POSITION_TOLERANCE_M and rot_error <= ORIENTATION_TOLERANCE_RAD:
            converged = True
            break
        arm_position = q[position_indices]

        if solver == "constrained_qp":
            try:
                from scipy.optimize import lsq_linear
            except ImportError as exc:
                raise ValueError("SciPy is unavailable in the Isaac runner") from exc
            # The TCP task stays dominant. These small secondary terms only
            # choose among redundant 7-DoF solutions instead of accepting a
            # visibly inaccurate TCP pose to preserve posture.
            posture_weight = 1.0e-4
            center_weight = 1.0e-7
            center = 0.5 * (lower + upper)
            # This bounded least-squares form is the same convex quadratic
            # objective, solved without an optional qpsolvers backend:
            #   ||J dq - error||^2 + wp||q+dq-q_ref||^2
            #                           + wc||q+dq-q_center||^2.
            identity = np.eye(7)
            system = np.vstack(
                (
                    jacobian,
                    math.sqrt(posture_weight) * identity,
                    math.sqrt(center_weight) * identity,
                )
            )
            target = np.concatenate(
                (
                    error,
                    math.sqrt(posture_weight) * (reference - arm_position),
                    math.sqrt(center_weight) * (center - arm_position),
                )
            )
            qp_result = lsq_linear(
                system,
                target,
                bounds=(
                    np.maximum(-MAX_JOINT_STEP_RAD, lower - arm_position),
                    np.minimum(MAX_JOINT_STEP_RAD, upper - arm_position),
                ),
                method="trf",
                tol=1.0e-8,
                lsmr_tol="auto",
                max_iter=50,
            )
            delta = qp_result.x
            if not qp_result.success or not np.all(np.isfinite(delta)):
                raise ValueError("Constrained QP could not find a finite joint update")
            touched_limit |= bool(np.any(qp_result.active_mask != 0))
        else:
            task_matrix = jacobian @ jacobian.T + damping**2 * np.eye(6)
            delta = jacobian.T @ np.linalg.solve(task_matrix, error)
            if solver == "posture_lm":
                jacobian_pinv = jacobian.T @ np.linalg.solve(task_matrix, np.eye(6))
                null_projector = np.eye(7) - jacobian_pinv @ jacobian
                delta += 0.12 * null_projector @ (reference - arm_position)
            largest = float(np.max(np.abs(delta)))
            if largest > MAX_JOINT_STEP_RAD:
                delta *= MAX_JOINT_STEP_RAD / largest

        accepted = False
        for line_scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            proposed = arm_position + line_scale * delta
            clipped = np.clip(proposed, lower, upper)
            touched_limit |= not np.allclose(proposed, clipped, atol=1.0e-10)
            candidate_q = q.copy()
            candidate_q[position_indices] = clipped
            *_, candidate_objective = _evaluate(
                pin,
                model,
                data,
                candidate_q,
                tcp_frame_id,
                velocity_indices,
                target_model_pose,
            )
            if candidate_objective < objective - 1.0e-12:
                q = candidate_q
                damping = max(0.005, damping * 0.7)
                accepted = True
                break
        if not accepted:
            damping = min(1.0, damping * 2.0)
            if damping >= 1.0:
                break

    _, _, _, pos_error, rot_error, _ = _evaluate(
        pin, model, data, q, tcp_frame_id, velocity_indices, target_model_pose
    )
    converged = converged or (
        pos_error <= POSITION_TOLERANCE_M
        and rot_error <= ORIENTATION_TOLERANCE_RAD
    )
    if not converged:
        raise ValueError(
            f"{IK_SOLVER_LABELS[solver]} did not converge after {iteration} iterations: "
            f"position residual {pos_error:.4f} m, orientation residual "
            f"{math.degrees(rot_error):.2f} deg. Add a closer intermediate target."
        )

    candidate_values = q[position_indices]
    candidate = torch.tensor(
        candidate_values,
        device=robot.device,
        dtype=robot.data.joint_pos.dtype,
    ).reshape(1, 7)
    current_values = robot.data.joint_pos[0, context["joint_ids"]].detach().cpu().numpy()
    margins = [
        min(value - limits[name][0], limits[name][1] - value)
        for name, value in zip(context["joint_names"], candidate_values)
    ]
    notes = {
        "iterative_lm": (
            "Nonlinear LM with FK/Jacobian recomputation, line search, per-step "
            "limits, and URDF joint margins."
        ),
        "posture_lm": (
            "Nonlinear LM with a null-space bias toward the measured starting posture."
        ),
        "constrained_qp": (
            "Iterative bounded QP with explicit joint-step and URDF-limit bounds plus posture continuity."
        ),
    }
    return {
        "side": side,
        "solver": solver,
        "solver_label": IK_SOLVER_LABELS[solver],
        "frame": "world",
        "target_xyz_m": context["target_xyz_w"][0].cpu().tolist(),
        "current_tcp_xyz_m": context["current_xyz_w"][0].cpu().tolist(),
        "current_orientation_held": True,
        "distance_m": context["distance"],
        "converged": True,
        "iterations": iteration,
        "position_residual_m": pos_error,
        "orientation_residual_rad": rot_error,
        "linear_residual_m": pos_error,
        "joint_step_scale": 1.0,
        "urdf_limited": touched_limit,
        "max_joint_delta_rad": float(np.max(np.abs(candidate_values - current_values))),
        "min_joint_limit_margin_rad": float(min(margins)),
        "reference_posture": "current_measured" if solver != "iterative_lm" else None,
        "q_des": dict(zip(context["joint_names"], candidate[0].cpu().tolist())),
        "note": notes[solver],
    }
