"""Pure helpers for replaying coffee-can grasps with measured TCP tracking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class GraspTrackingCfg:
    """Approved measured-position gates around the seed grasp reference."""

    position_tolerance_m: float = 0.010
    stable_steps: int = 5
    preclose_timeout_steps: int = 90
    closed_alignment_timeout_steps: int = 60
    minimum_close_excursion: float = 0.100
    significant_close_delta: float = 0.005
    maximum_quiet_gap_steps: int = 10


@dataclass(frozen=True)
class CloseSegment:
    """The substantive close ramp and its open/closed hold values."""

    onset_step: int
    grasp_step: int
    open_value: float
    closed_value: float


DEFAULT_GRASP_TRACKING = GraspTrackingCfg()


def _finite(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError("gripper values must be a non-empty finite sequence")
    return result


def infer_close_segment(
    gripper_values: Sequence[float],
    *,
    operator_start_step: int,
    grasp_step: int,
    g_step: int,
    cfg: GraspTrackingCfg = DEFAULT_GRASP_TRACKING,
) -> CloseSegment:
    """Infer the final closing ramp without mistaking open-gripper settling for it.

    The clean seed reaches a stable closed plateau at ``grasp_step``. Walking the
    significant gripper changes backwards from that marker separates the final
    close ramp from earlier tact/activation settling by its quiet gap.
    """

    values = _finite(gripper_values)
    if not 0 <= operator_start_step < grasp_step < g_step <= len(values):
        raise ValueError("expected operator_start < grasp < G within gripper values")

    plateau_end = min(g_step, grasp_step + cfg.stable_steps + 2)
    closed_value = float(median(values[grasp_step:plateau_end]))
    prefix_end = min(grasp_step, operator_start_step + 10)
    initial_open = float(median(values[operator_start_step:prefix_end]))
    if abs(closed_value - initial_open) < cfg.minimum_close_excursion:
        raise ValueError("active gripper has no clear close excursion")

    direction = 1.0 if closed_value > initial_open else -1.0
    significant_steps = [
        step
        for step in range(operator_start_step + 1, grasp_step + 1)
        if direction * (values[step] - values[step - 1])
        >= cfg.significant_close_delta
    ]
    if not significant_steps:
        raise ValueError("active gripper has no substantive close ramp")

    onset = significant_steps[-1]
    for previous, current in zip(
        reversed(significant_steps[:-1]), reversed(significant_steps[1:])
    ):
        if current - previous > cfg.maximum_quiet_gap_steps:
            break
        onset = previous

    open_window_start = max(operator_start_step, onset - cfg.stable_steps)
    open_value = float(median(values[open_window_start:onset]))
    if direction * (closed_value - open_value) < cfg.minimum_close_excursion:
        raise ValueError("inferred close ramp does not reach a closed plateau")
    return CloseSegment(
        onset_step=onset,
        grasp_step=grasp_step,
        open_value=open_value,
        closed_value=closed_value,
    )


def translation_alignment_passes(
    error_xyz_m: Sequence[float],
    cfg: GraspTrackingCfg = DEFAULT_GRASP_TRACKING,
) -> bool:
    """Return whether every can-relative TCP translation axis is within 1 cm."""

    error = _finite(error_xyz_m)
    if len(error) != 3:
        raise ValueError("translation error must contain exactly x, y, z")
    return all(abs(value) <= cfg.position_tolerance_m for value in error)
