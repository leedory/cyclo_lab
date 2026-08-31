"""Task-local coffee-can rigid-object configurations."""

from __future__ import annotations

from copy import deepcopy

from cyclo_lab.assets.object import COFFEE_CAN_CFG

from .layout import candidate_sampling_regions


def make_task000525_coffee_can_cfg(region):
    """Instantiate one appearance variant at its reviewed default center."""

    appearance = region.object_name.removeprefix("coffee_can_")
    cfg = deepcopy(COFFEE_CAN_CFG)
    cfg.prim_path = f"{{ENV_REGEX_NS}}/{region.object_name}"
    cfg.spawn.variants = {"appearance": appearance}
    cfg.spawn.semantic_tags = [
        ("class", "coffee_can"),
        ("instance", region.object_name),
        ("appearance", appearance),
    ]
    cfg.init_state.pos = region.default_position_m
    cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    return cfg


def iter_task000525_coffee_can_cfgs(layout_key: str):
    """Yield the four scene entities in low-Y to high-Y region order."""

    for region in candidate_sampling_regions(layout_key):
        yield region.object_name, make_task000525_coffee_can_cfg(region)
