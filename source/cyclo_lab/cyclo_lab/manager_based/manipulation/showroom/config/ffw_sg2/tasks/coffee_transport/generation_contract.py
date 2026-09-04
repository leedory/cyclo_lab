"""Shared trajectory-generation semantics for the temporary coffee tasks.

This module is descriptive. Runtime boundary signals and trajectory generation
will consume this contract once seed collection is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CoffeeTransportRetargetFrame(str, Enum):
    """Reference frames used while retargeting one recorded can cycle."""

    EPISODE_HOME = "episode_home"
    INITIAL_CAN = "initial_can"
    INITIAL_CAN_TO_EPISODE_HOME_BLEND = "initial_can_to_episode_home_blend"
    DESTINATION_SUPPORT = "destination_support"


@dataclass(frozen=True)
class CoffeeCanCycleContract:
    """Fixed identity and gripper assignment for one can cycle."""

    order: int
    can_name: str
    arm: str


@dataclass(frozen=True)
class CoffeeTransportPhaseContract:
    """One source-trajectory phase and its terminal boundary signal."""

    key: str
    can_name: str | None
    arm: str | None
    end_signal: str
    eef_reference: CoffeeTransportRetargetFrame
    description: str


COFFEE_CAN_CYCLES = (
    CoffeeCanCycleContract(0, "coffee_can_right", "right"),
    CoffeeCanCycleContract(1, "coffee_can_center", "right"),
    CoffeeCanCycleContract(2, "coffee_can_left", "left"),
)


def _can_phases(cycle: CoffeeCanCycleContract) -> tuple[CoffeeTransportPhaseContract, ...]:
    name = cycle.can_name
    arm = cycle.arm
    return (
        CoffeeTransportPhaseContract(
            f"grasp_{name}",
            name,
            arm,
            f"{name}_grasp_stable",
            CoffeeTransportRetargetFrame.INITIAL_CAN,
            "Approach the named can and establish a stable grasp.",
        ),
        CoffeeTransportPhaseContract(
            f"clear_{name}_to_carry_home",
            name,
            arm,
            f"{name}_carry_home_stable",
            CoffeeTransportRetargetFrame.INITIAL_CAN_TO_EPISODE_HOME_BLEND,
            "Clear the cabinet and settle the active arm at its carry HOME.",
        ),
        CoffeeTransportPhaseContract(
            f"transport_{name}",
            name,
            arm,
            f"{name}_transport_ready",
            CoffeeTransportRetargetFrame.EPISODE_HOME,
            "Hold the can robot-root-relative while lift or base reaches the destination.",
        ),
        CoffeeTransportPhaseContract(
            f"place_{name}",
            name,
            arm,
            f"{name}_placed_stable",
            CoffeeTransportRetargetFrame.DESTINATION_SUPPORT,
            "Replay the inverse pickup path, release, and verify stable placement.",
        ),
        CoffeeTransportPhaseContract(
            f"recover_{name}",
            None,
            arm,
            f"{name}_cycle_home_ready",
            CoffeeTransportRetargetFrame.EPISODE_HOME,
            "Return the empty arm and lift or base to the same episode HOME.",
        ),
    )


COFFEE_TRANSPORT_PHASES = (
    CoffeeTransportPhaseContract(
        "normalize_start",
        None,
        None,
        "episode_home_ready",
        CoffeeTransportRetargetFrame.EPISODE_HOME,
        "Move the randomized lift or base to the cached episode HOME before grasping.",
    ),
    *tuple(phase for cycle in COFFEE_CAN_CYCLES for phase in _can_phases(cycle)),
)
