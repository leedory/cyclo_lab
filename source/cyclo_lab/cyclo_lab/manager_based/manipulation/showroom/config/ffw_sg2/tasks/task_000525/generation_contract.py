"""Runtime trajectory-generation contract for Task000525.

This file describes the frames actually consumed by the Task525 local
trajectory generator.  It is intentionally descriptive: changing a reference
frame changes the generated motion and requires a separate replay validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Task000525RetargetFrame(str, Enum):
    """Reference-frame policies used by the generated EEF targets."""

    INITIAL_TARGET_OBJECT = "initial_target_object"
    INITIAL_OBJECT_TO_CURRENT_ROOT_BLEND = (
        "initial_target_object_to_current_robot_root_blend"
    )
    CURRENT_ROBOT_ROOT = "current_robot_root"
    RECORDED_BASE_TO_CURRENT_ROOT = "recorded_base_to_current_robot_root"


@dataclass(frozen=True)
class Task000525PhaseContract:
    """One source phase and its runtime retarget policy."""

    phase_id: int
    key: str
    start_marker: str
    eef_reference: Task000525RetargetFrame
    description: str


TASK000525_GENERATION_PHASES = (
    Task000525PhaseContract(
        0,
        "grasp_can",
        "B",
        Task000525RetargetFrame.INITIAL_TARGET_OBJECT,
        "Grasp against fixed source/target can frames captured at episode reset.",
    ),
    Task000525PhaseContract(
        1,
        "clear_cabinet_and_return_carry_home",
        "F",
        Task000525RetargetFrame.INITIAL_OBJECT_TO_CURRENT_ROOT_BLEND,
        "Blend from can-relative pull-out to robot-root-relative carry/home.",
    ),
    Task000525PhaseContract(
        2,
        "navigate_to_table",
        "G",
        Task000525RetargetFrame.CURRENT_ROBOT_ROOT,
        "Hold both hands robot-root-relative during planned base motion.",
    ),
    Task000525PhaseContract(
        3,
        "place_release_and_return_home",
        "arrival",
        Task000525RetargetFrame.RECORDED_BASE_TO_CURRENT_ROOT,
        "Replay the recorded place, release, and empty return at the current base.",
    ),
)

TASK000525_QUALITY_EVALUATORS = (
    "evaluate_task525_carry_checkpoint",
    "evaluate_task525_final_checkpoint",
)
