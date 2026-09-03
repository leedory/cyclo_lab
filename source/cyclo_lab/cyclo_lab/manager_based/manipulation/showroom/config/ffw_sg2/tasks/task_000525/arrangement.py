"""Task525 coffee-can identity, region, and manipulation-side policy."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .layout import TASK000525_CAN_NAMES, TASK000525_REGION_KEYS


TASK000525_TARGET_OBJECT = "coffee_can_orange"
TASK000525_DISTRACTOR_OBJECTS = tuple(
    name for name in TASK000525_CAN_NAMES if name != TASK000525_TARGET_OBJECT
)
TASK000525_REGION_TO_SIDE = {
    "A": "left",
    "B": "left",
    "C": "right",
    "D": "right",
}


def validate_region_key(region_key: str) -> str:
    """Return a normalized A-D region key or raise a useful error."""

    normalized = str(region_key).upper()
    if normalized not in TASK000525_REGION_KEYS:
        choices = ", ".join(TASK000525_REGION_KEYS)
        raise ValueError(f"Unknown Task525 region {region_key!r}; choose {choices}")
    return normalized


def manipulation_side_for_region(region_key: str) -> str:
    """Return the fixed arm policy: A/B left and C/D right."""

    return TASK000525_REGION_TO_SIDE[validate_region_key(region_key)]


@dataclass(frozen=True)
class CoffeeArrangement:
    """Assignment of the four stable coffee-can entities to spatial regions."""

    target_region: str
    region_to_object: dict[str, str]

    def __post_init__(self) -> None:
        target_region = validate_region_key(self.target_region)
        if set(self.region_to_object) != set(TASK000525_REGION_KEYS):
            raise ValueError("Task525 arrangement must assign every A-D region exactly once")
        if set(self.region_to_object.values()) != set(TASK000525_CAN_NAMES):
            raise ValueError("Task525 arrangement must assign every coffee-can entity exactly once")
        if self.region_to_object[target_region] != TASK000525_TARGET_OBJECT:
            raise ValueError("Task525 target region must contain coffee_can_orange")
        object.__setattr__(self, "target_region", target_region)

    @property
    def manipulation_side(self) -> str:
        return manipulation_side_for_region(self.target_region)

    @property
    def object_to_region(self) -> dict[str, str]:
        return {
            object_name: region_key
            for region_key, object_name in self.region_to_object.items()
        }


def make_coffee_arrangement(
    target_region: str,
    *,
    shuffle_distractors: bool = False,
    rng: Random | None = None,
) -> CoffeeArrangement:
    """Place orange in one region and assign distractors to the remaining regions."""

    target_region = validate_region_key(target_region)
    distractors = list(TASK000525_DISTRACTOR_OBJECTS)
    if shuffle_distractors:
        (rng or Random()).shuffle(distractors)
    remaining_regions = [
        region_key
        for region_key in TASK000525_REGION_KEYS
        if region_key != target_region
    ]
    region_to_object = dict(zip(remaining_regions, distractors))
    region_to_object[target_region] = TASK000525_TARGET_OBJECT
    return CoffeeArrangement(target_region, region_to_object)
