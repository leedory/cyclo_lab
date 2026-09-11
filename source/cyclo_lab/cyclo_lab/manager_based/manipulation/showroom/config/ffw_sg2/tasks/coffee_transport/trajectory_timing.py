"""Temporal alignment helpers for recorded coffee trajectories."""

from __future__ import annotations

import numpy as np


def scene_pose_at_pre_step(
    initial_pose: np.ndarray,
    post_step_poses: np.ndarray,
    step: int,
) -> np.ndarray:
    """Return the scene pose aligned with pre-step observation/action ``step``.

    Isaac Lab records observations and actions before physics, but scene states
    after physics. Consequently, pre-step row 0 matches ``initial_state`` and
    pre-step row ``t`` matches post-step state row ``t - 1`` for ``t > 0``.
    """

    if step < 0:
        raise IndexError(f"pre-step index must be non-negative, got {step}")
    if step == 0:
        return initial_pose
    state_index = step - 1
    if state_index >= len(post_step_poses):
        raise IndexError(
            f"pre-step index {step} requires post-step state {state_index}, "
            f"but only {len(post_step_poses)} states are available"
        )
    return post_step_poses[state_index]
