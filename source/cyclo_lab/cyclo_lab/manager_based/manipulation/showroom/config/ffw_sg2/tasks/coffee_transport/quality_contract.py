"""Pure quality gates for coffee body grasps and final placements."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QualityDecision:
    """A deterministic decision with stable machine-readable failure codes."""

    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class BodyCarryQualityCfg:
    """Diagnostic quality for a body grasp; G does not block on this score."""

    eef_distance_min_m: float = 0.080
    eef_distance_source_tolerance_m: float = 0.025
    eef_distance_hard_max_m: float = 0.100
    tcp_z_relative_to_can_range_m: tuple[float, float] = (-0.030, 0.050)
    can_displacement_min_m: float = 0.200
    home_joint_max_error: float = 0.150
    can_linear_speed_max_mps: float = 0.050
    can_angular_speed_max_radps: float = 0.500
    stable_steps: int = 5


@dataclass(frozen=True)
class PlacementQualityCfg:
    """Gate before lift/base is allowed to return toward the source."""

    center_position_tolerance_m: float = 0.006
    support_height_tolerance_m: float = 0.010
    upright_max_tilt_rad: float = math.radians(10.0)
    can_linear_speed_max_mps: float = 0.030
    can_angular_speed_max_radps: float = 0.250
    released_eef_distance_min_m: float = 0.100
    stable_steps: int = 8


DEFAULT_BODY_CARRY_QUALITY = BodyCarryQualityCfg()
DEFAULT_PLACEMENT_QUALITY = PlacementQualityCfg()


def _require_finite(**metrics: float) -> None:
    invalid = tuple(name for name, value in metrics.items() if not math.isfinite(value))
    if invalid:
        raise ValueError(f"Coffee quality metrics must be finite: {invalid}")


def generated_carry_distance_limit_m(
    source_distance_m: float,
    cfg: BodyCarryQualityCfg = DEFAULT_BODY_CARRY_QUALITY,
) -> float:
    """Return the source-relative limit, capped by the body-grasp hard bound."""

    _require_finite(source_distance_m=source_distance_m)
    if source_distance_m < 0.0:
        raise ValueError("Source EEF-to-can distance must be non-negative.")
    source_relative = max(
        cfg.eef_distance_min_m,
        source_distance_m + cfg.eef_distance_source_tolerance_m,
    )
    return min(cfg.eef_distance_hard_max_m, source_relative)


def evaluate_body_carry(
    *,
    eef_object_distance_m: float,
    source_eef_object_distance_m: float,
    tcp_z_relative_to_can_m: float,
    can_displacement_m: float,
    home_joint_max_error: float,
    can_linear_speed_mps: float,
    can_angular_speed_radps: float,
    cfg: BodyCarryQualityCfg = DEFAULT_BODY_CARRY_QUALITY,
    require_stability: bool = True,
) -> QualityDecision:
    """Evaluate one carry frame, optionally treating motion as diagnostic only."""

    _require_finite(
        eef_object_distance_m=eef_object_distance_m,
        source_eef_object_distance_m=source_eef_object_distance_m,
        tcp_z_relative_to_can_m=tcp_z_relative_to_can_m,
        can_displacement_m=can_displacement_m,
        home_joint_max_error=home_joint_max_error,
        can_linear_speed_mps=can_linear_speed_mps,
        can_angular_speed_radps=can_angular_speed_radps,
    )
    failures = []
    distance_limit = generated_carry_distance_limit_m(
        source_eef_object_distance_m, cfg
    )
    if eef_object_distance_m > distance_limit:
        failures.append("eef_too_far_from_can")
    body_z_min, body_z_max = cfg.tcp_z_relative_to_can_range_m
    if not body_z_min <= tcp_z_relative_to_can_m <= body_z_max:
        failures.append("tcp_outside_body_grasp_band")
    if can_displacement_m < cfg.can_displacement_min_m:
        failures.append("can_not_clear_of_source")
    if home_joint_max_error > cfg.home_joint_max_error:
        failures.append("carry_home_not_reached")
    if require_stability:
        if can_linear_speed_mps > cfg.can_linear_speed_max_mps:
            failures.append("can_linear_motion_not_stable")
        if can_angular_speed_radps > cfg.can_angular_speed_max_radps:
            failures.append("can_angular_motion_not_stable")
    return QualityDecision(tuple(failures))


def upright_cosine_from_wxyz(quaternion_wxyz: tuple[float, float, float, float]) -> float:
    """Return the can local +Z dot world +Z for a normalized WXYZ quaternion."""

    w, x, y, z = quaternion_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("Can quaternion must have a finite positive norm.")
    x /= norm
    y /= norm
    return 1.0 - 2.0 * (x * x + y * y)


def evaluate_placement(
    *,
    center_x_m: float,
    center_y_m: float,
    origin_z_m: float,
    expected_origin_z_m: float,
    upright_cosine: float,
    can_linear_speed_mps: float,
    can_angular_speed_radps: float,
    active_eef_distance_m: float,
    center_region_xyxy_m: tuple[float, float, float, float],
    cfg: PlacementQualityCfg = DEFAULT_PLACEMENT_QUALITY,
) -> QualityDecision:
    """Require named-region containment, support, uprightness, settling, and release."""

    _require_finite(
        center_x_m=center_x_m,
        center_y_m=center_y_m,
        origin_z_m=origin_z_m,
        expected_origin_z_m=expected_origin_z_m,
        upright_cosine=upright_cosine,
        can_linear_speed_mps=can_linear_speed_mps,
        can_angular_speed_radps=can_angular_speed_radps,
        active_eef_distance_m=active_eef_distance_m,
    )
    x_min, x_max, y_min, y_max = center_region_xyxy_m
    if not x_min <= x_max or not y_min <= y_max:
        raise ValueError("Placement center region bounds must be ordered.")

    failures = []
    tolerance = cfg.center_position_tolerance_m
    if not (
        x_min - tolerance <= center_x_m <= x_max + tolerance
        and y_min - tolerance <= center_y_m <= y_max + tolerance
    ):
        failures.append("can_center_outside_named_destination_region")
    if abs(origin_z_m - expected_origin_z_m) > cfg.support_height_tolerance_m:
        failures.append("can_not_on_destination_support")
    if upright_cosine < math.cos(cfg.upright_max_tilt_rad):
        failures.append("can_not_upright")
    if can_linear_speed_mps > cfg.can_linear_speed_max_mps:
        failures.append("can_linear_motion_not_settled")
    if can_angular_speed_radps > cfg.can_angular_speed_max_radps:
        failures.append("can_angular_motion_not_settled")
    if active_eef_distance_m < cfg.released_eef_distance_min_m:
        failures.append("active_gripper_not_released")
    return QualityDecision(tuple(failures))
