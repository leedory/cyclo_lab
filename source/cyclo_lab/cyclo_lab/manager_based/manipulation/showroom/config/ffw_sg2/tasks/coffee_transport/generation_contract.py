"""Seed and generation phase contract for the coffee-transport task family.

Before each can, automation holds the fixed HOME until a fresh arm-specific
A3 tact immediately transfers operator control. Operators then teleoperate only the two
source-side segments for that can. Pressing G after extracting a grasped can
transfers control to automation until the next can is ready at the nominal
robot posture.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PhaseExecutor(str, Enum):
    OPERATOR = "operator"
    AUTOMATION = "automation"


class CoffeeTransportRetargetFrame(str, Enum):
    """Reference frames used while retargeting one recorded can cycle."""

    EPISODE_BASE_HOME = "episode_base_home"
    INITIAL_CAN = "initial_can"
    INITIAL_CAN_TO_CARRY_HOME_BLEND = "initial_can_to_carry_home_blend"
    TASK_AXIS = "task_axis"
    DESTINATION_SUPPORT = "destination_support"


@dataclass(frozen=True)
class CoffeeCanCycleContract:
    """Fixed identity and gripper assignment for one can cycle."""

    order: int
    can_name: str
    arm: str

    @property
    def operator_label(self) -> str:
        return f"{self.order + 1}"


@dataclass(frozen=True)
class CoffeeTransportPhaseContract:
    """One explicit state in seed acquisition or trajectory generation."""

    phase_id: int
    key: str
    cycle_order: int | None
    can_name: str | None
    arm: str | None
    executor: PhaseExecutor
    start_signal: str
    end_signal: str
    eef_reference: CoffeeTransportRetargetFrame
    reverse_of: str | None
    description: str


COFFEE_CAN_CYCLES = (
    CoffeeCanCycleContract(0, "coffee_can_right", "right"),
    CoffeeCanCycleContract(1, "coffee_can_left", "left"),
)


def _phase(
    phase_id: int,
    key: str,
    cycle: CoffeeCanCycleContract | None,
    executor: PhaseExecutor,
    start_signal: str,
    end_signal: str,
    reference: CoffeeTransportRetargetFrame,
    description: str,
    *,
    reverse_of: str | None = None,
) -> CoffeeTransportPhaseContract:
    return CoffeeTransportPhaseContract(
        phase_id=phase_id,
        key=key,
        cycle_order=None if cycle is None else cycle.order,
        can_name=None if cycle is None else cycle.can_name,
        arm=None if cycle is None else cycle.arm,
        executor=executor,
        start_signal=start_signal,
        end_signal=end_signal,
        eef_reference=reference,
        reverse_of=reverse_of,
        description=description,
    )


def _can_phases(
    cycle: CoffeeCanCycleContract,
    first_phase_id: int,
) -> tuple[CoffeeTransportPhaseContract, ...]:
    """Build the two teleop and eight automatic states for one can."""

    name = cycle.can_name
    pick_key = f"operator_pick_{name}"
    carry_key = f"operator_extract_{name}_until_g"
    definitions = (
        (
            pick_key,
            PhaseExecutor.OPERATOR,
            f"{name}_cycle_ready",
            f"{name}_body_grasp_stable",
            CoffeeTransportRetargetFrame.INITIAL_CAN,
            None,
            "Move from nominal to the can and close on the cylindrical body.",
        ),
        (
            carry_key,
            PhaseExecutor.OPERATOR,
            f"{name}_body_grasp_stable",
            f"{name}_g_pressed_after_extract",
            CoffeeTransportRetargetFrame.INITIAL_CAN_TO_CARRY_HOME_BLEND,
            None,
            "Extract the held can clear of the source, then press G once.",
        ),
        (
            f"auto_{name}_return_home",
            PhaseExecutor.AUTOMATION,
            f"{name}_g_pressed_after_extract",
            f"{name}_home_stable",
            CoffeeTransportRetargetFrame.INITIAL_CAN_TO_CARRY_HOME_BLEND,
            None,
            (
                "Blend both arms and lift to the fixed episode HOME, preserve the "
                "closed active gripper and head, then wait for measured stability."
            ),
        ),
        (
            f"auto_{name}_axis_to_destination",
            PhaseExecutor.AUTOMATION,
            f"{name}_home_stable",
            f"{name}_axis_at_destination",
            CoffeeTransportRetargetFrame.TASK_AXIS,
            None,
            "Hold the closed grasp while lift or base reaches the destination.",
        ),
        (
            f"auto_{name}_inverse_carry_closed",
            PhaseExecutor.AUTOMATION,
            f"{name}_axis_at_destination",
            f"{name}_destination_grasp_pose",
            CoffeeTransportRetargetFrame.DESTINATION_SUPPORT,
            carry_key,
            (
                "Replay the buffered operator-extract plus automatic-HOME path "
                "backward with the gripper closed."
            ),
        ),
        (
            f"auto_{name}_release",
            PhaseExecutor.AUTOMATION,
            f"{name}_destination_grasp_pose",
            f"{name}_gripper_open",
            CoffeeTransportRetargetFrame.DESTINATION_SUPPORT,
            None,
            "Open only the active gripper at the mirrored grasp pose.",
        ),
        (
            f"auto_{name}_inverse_approach_open",
            PhaseExecutor.AUTOMATION,
            f"{name}_gripper_open",
            f"{name}_arm_nominal_at_destination",
            CoffeeTransportRetargetFrame.DESTINATION_SUPPORT,
            pick_key,
            "Replay the approach backward with the active gripper forced open, then command exact destination-side HOME and require measured stability.",
        ),
        (
            f"auto_{name}_validate_placement",
            PhaseExecutor.AUTOMATION,
            f"{name}_arm_nominal_at_destination",
            f"{name}_placement_checked",
            CoffeeTransportRetargetFrame.DESTINATION_SUPPORT,
            None,
            (
                "Observe placement stability for the bounded return wait; the final "
                "N gate remains authoritative if the interim check has not passed."
            ),
        ),
        (
            f"auto_{name}_axis_to_source",
            PhaseExecutor.AUTOMATION,
            f"{name}_placement_checked",
            f"{name}_axis_at_source",
            CoffeeTransportRetargetFrame.TASK_AXIS,
            None,
            "Return lift or base after the bounded placement observation.",
        ),
        (
            f"auto_{name}_cycle_complete",
            PhaseExecutor.AUTOMATION,
            f"{name}_axis_at_source",
            f"{name}_cycle_complete",
            CoffeeTransportRetargetFrame.EPISODE_BASE_HOME,
            None,
            "Hold nominal joints at the source axis and wait for a fresh arm-specific A3 tact.",
        ),
    )
    return tuple(
        _phase(
            first_phase_id + offset,
            key,
            cycle,
            executor,
            start_signal,
            end_signal,
            reference,
            description,
            reverse_of=reverse_of,
        )
        for offset, (
            key,
            executor,
            start_signal,
            end_signal,
            reference,
            reverse_of,
            description,
        ) in enumerate(definitions)
    )


COFFEE_TRANSPORT_PHASES = (
    _phase(
        0,
        "normalize_episode_start",
        None,
        PhaseExecutor.AUTOMATION,
        "episode_reset",
        "first_can_cycle_ready",
        CoffeeTransportRetargetFrame.EPISODE_BASE_HOME,
        (
            "For generated episodes, move the randomized lift or base to nominal "
            "before enabling the first teleoperated-equivalent pickup."
        ),
    ),
    *tuple(
        phase
        for cycle in COFFEE_CAN_CYCLES
        for phase in _can_phases(cycle, 1 + 10 * cycle.order)
    ),
)

COFFEE_OPERATOR_PHASES = tuple(
    phase
    for phase in COFFEE_TRANSPORT_PHASES
    if phase.executor is PhaseExecutor.OPERATOR
)
COFFEE_AUTOMATION_PHASES = tuple(
    phase
    for phase in COFFEE_TRANSPORT_PHASES
    if phase.executor is PhaseExecutor.AUTOMATION
)
