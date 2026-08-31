"""SG2 showroom state preparation for position-command replay."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def prepare_sg2_position_replay_state(
    state: Mapping,
    articulation_name: str = "robot",
) -> dict:
    """Return a reset state with the SG2 articulation initially at rest.

    Isaac Lab's generic ``reset_to()`` also copies ``joint_velocity`` into the
    actuator velocity target. Position-command episodes start from rest, so a
    recorded instantaneous velocity must not become a persistent command.
    """

    articulation_states = dict(state["articulation"])
    robot_state = dict(articulation_states[articulation_name])
    root_velocity = robot_state["root_velocity"]
    joint_velocity = robot_state["joint_velocity"]
    if not isinstance(root_velocity, torch.Tensor) or not isinstance(
        joint_velocity, torch.Tensor
    ):
        raise TypeError("robot root_velocity and joint_velocity must be tensors")

    robot_state["root_velocity"] = torch.zeros_like(root_velocity)
    robot_state["joint_velocity"] = torch.zeros_like(joint_velocity)
    articulation_states[articulation_name] = robot_state
    prepared = dict(state)
    prepared["articulation"] = articulation_states
    return prepared


def restore_sg2_replay_root_pose(
    env,
    root_pose: torch.Tensor,
    env_ids,
    articulation_name: str = "robot",
) -> None:
    """Restore the recorded stationary root pose after one replay step."""

    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)
    if root_pose.shape != (len(ids), 7):
        raise ValueError(
            f"root_pose must have shape ({len(ids)}, 7), got {tuple(root_pose.shape)}"
        )

    root_pose_w = root_pose.clone()
    root_pose_w[:, :3] += env.scene.env_origins[ids]
    robot = env.scene[articulation_name]
    robot.write_root_pose_to_sim(root_pose_w, env_ids=ids)
    robot.write_root_velocity_to_sim(
        torch.zeros((len(ids), 6), dtype=root_pose.dtype, device=env.device),
        env_ids=ids,
    )
