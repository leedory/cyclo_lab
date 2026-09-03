"""Measured shelf geometry and reviewed layout candidates for Task000525.

Every rectangle in this module is a domain for the coffee can's *center point*.
The physical can extends 33.5 mm beyond a sampled center in every lateral
direction, so the helpers report body clearances separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


TASK000525_CAN_NAMES = (
    "coffee_can_black",
    "coffee_can_brown",
    "coffee_can_green",
    "coffee_can_orange",
)
TASK000525_REGION_KEYS = ("A", "B", "C", "D")
TASK000525_CAN_RADIUS_M = 0.0335
TASK000525_CAN_HEIGHT_M = 0.133
TASK000525_CAN_ORIGIN_ABOVE_BOTTOM_M = 0.045


@dataclass(frozen=True)
class ShelfBounds:
    """World-aligned top support bounds measured from cabinet_02 C03_LATERAL."""

    x_min_back_m: float
    x_max_front_m: float
    y_min_m: float
    y_max_m: float
    top_z_m: float

    @property
    def depth_x_m(self) -> float:
        return self.x_max_front_m - self.x_min_back_m

    @property
    def width_y_m(self) -> float:
        return self.y_max_m - self.y_min_m


TASK000525_SHELF_BOUNDS = ShelfBounds(
    x_min_back_m=-2.4061267375946045,
    x_max_front_m=-2.1061267852783203,
    y_min_m=0.3780692815780638,
    y_max_m=1.1694384813308714,
    top_z_m=1.3051289319992065,
)
TASK000525_CAN_UPRIGHT_ORIGIN_Z_M = (
    TASK000525_SHELF_BOUNDS.top_z_m + TASK000525_CAN_ORIGIN_ABOVE_BOTTOM_M
)


@dataclass(frozen=True)
class ShelfLayoutParameters:
    """Margins around and between four can-center sampling rectangles."""

    key: str
    korean_name: str
    lateral_low_y_margin_mm: float
    lateral_high_y_margin_mm: float
    between_regions_mm: float
    back_low_x_margin_mm: float
    front_high_x_margin_mm: float
    recommended: bool = False


@dataclass(frozen=True)
class CenterSamplingRegion:
    region_key: str
    x_min_back_m: float
    x_max_front_m: float
    y_min_m: float
    y_max_m: float
    default_position_m: tuple[float, float, float]

    @property
    def depth_x_m(self) -> float:
        return self.x_max_front_m - self.x_min_back_m

    @property
    def width_y_m(self) -> float:
        return self.y_max_m - self.y_min_m


TASK000525_LAYOUT_CANDIDATES = {
    "A": ShelfLayoutParameters(
        key="A",
        korean_name="reference-fit",
        lateral_low_y_margin_mm=140.0,
        lateral_high_y_margin_mm=140.0,
        between_regions_mm=77.0,
        back_low_x_margin_mm=90.0,
        front_high_x_margin_mm=50.0,
    ),
    "B": ShelfLayoutParameters(
        key="B",
        korean_name="user-approved-forward",
        lateral_low_y_margin_mm=120.0,
        lateral_high_y_margin_mm=120.0,
        between_regions_mm=100.0,
        back_low_x_margin_mm=120.0,
        front_high_x_margin_mm=30.0,
        recommended=True,
    ),
    "C": ShelfLayoutParameters(
        key="C",
        korean_name="wide-random-range",
        lateral_low_y_margin_mm=60.0,
        lateral_high_y_margin_mm=60.0,
        between_regions_mm=77.0,
        back_low_x_margin_mm=55.0,
        front_high_x_margin_mm=45.0,
    ),
}

# Approved by the user after reviewing the A/B/C metric and showroom renders.
TASK000525_SELECTED_LAYOUT_KEY = "B"

# Rough authoring markers from /root/Documents/525_placement_draft.usd, sorted
# from low Y to high Y.  They are evidence, not production spawn positions.
TASK000525_REFERENCE_MARKER_POSITIONS_M = (
    (-2.216540264104232, 0.5495617451280137, 1.365386247634888),
    (-2.2279987430666397, 0.6863371347670837, 1.3653862476348932),
    (-2.227560435497610, 0.8580572001639317, 1.3653862476348877),
    (-2.236959169975144, 0.9914186200516855, 1.3653862476348873),
)


def build_sampling_regions(
    parameters: ShelfLayoutParameters,
) -> tuple[CenterSamplingRegion, ...]:
    """Build four equal center domains in low-Y to high-Y order."""

    bounds = TASK000525_SHELF_BOUNDS
    low_y_margin_m = parameters.lateral_low_y_margin_mm / 1000.0
    high_y_margin_m = parameters.lateral_high_y_margin_mm / 1000.0
    gap_m = parameters.between_regions_mm / 1000.0
    back_margin_m = parameters.back_low_x_margin_mm / 1000.0
    front_margin_m = parameters.front_high_x_margin_mm / 1000.0
    region_width_y_m = (
        bounds.width_y_m - low_y_margin_m - high_y_margin_m - 3.0 * gap_m
    ) / 4.0
    region_depth_x_m = bounds.depth_x_m - back_margin_m - front_margin_m
    if region_width_y_m <= 0.0 or region_depth_x_m <= 0.0:
        raise ValueError(f"Layout {parameters.key} does not fit inside the shelf")

    x_min = bounds.x_min_back_m + back_margin_m
    x_max = bounds.x_max_front_m - front_margin_m
    regions = []
    for index, region_key in enumerate(TASK000525_REGION_KEYS):
        y_min = bounds.y_min_m + low_y_margin_m + index * (
            region_width_y_m + gap_m
        )
        y_max = y_min + region_width_y_m
        regions.append(
            CenterSamplingRegion(
                region_key=region_key,
                x_min_back_m=x_min,
                x_max_front_m=x_max,
                y_min_m=y_min,
                y_max_m=y_max,
                default_position_m=(
                    (x_min + x_max) / 2.0,
                    (y_min + y_max) / 2.0,
                    TASK000525_CAN_UPRIGHT_ORIGIN_Z_M,
                ),
            )
        )
    return tuple(regions)


def candidate_sampling_regions(key: str) -> tuple[CenterSamplingRegion, ...]:
    try:
        parameters = TASK000525_LAYOUT_CANDIDATES[key.upper()]
    except KeyError as exc:
        choices = ", ".join(TASK000525_LAYOUT_CANDIDATES)
        raise ValueError(f"Unknown Task000525 layout {key!r}; choose {choices}") from exc
    return build_sampling_regions(parameters)


def selected_sampling_regions() -> tuple[CenterSamplingRegion, ...]:
    return candidate_sampling_regions(TASK000525_SELECTED_LAYOUT_KEY)


def guaranteed_clearances_mm(parameters: ShelfLayoutParameters) -> dict[str, float]:
    radius_mm = TASK000525_CAN_RADIUS_M * 1000.0
    return {
        "between_can_surfaces": parameters.between_regions_mm - 2.0 * radius_mm,
        "can_to_low_y_edge": parameters.lateral_low_y_margin_mm - radius_mm,
        "can_to_high_y_edge": parameters.lateral_high_y_margin_mm - radius_mm,
        "can_to_back_edge": parameters.back_low_x_margin_mm - radius_mm,
        "can_to_front_edge": parameters.front_high_x_margin_mm - radius_mm,
    }


def reference_y_rms_error_mm(parameters: ShelfLayoutParameters) -> float:
    regions = build_sampling_regions(parameters)
    errors = [
        (region.default_position_m[1] - marker[1]) * 1000.0
        for region, marker in zip(regions, TASK000525_REFERENCE_MARKER_POSITIONS_M)
    ]
    return sqrt(sum(error * error for error in errors) / len(errors))
