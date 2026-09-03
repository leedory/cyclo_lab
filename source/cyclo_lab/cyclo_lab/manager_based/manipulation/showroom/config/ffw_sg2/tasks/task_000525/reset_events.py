"""Task000525 reset events for independent per-region coffee-can sampling."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING

import torch

from .arrangement import make_coffee_arrangement, validate_region_key
from .layout import TASK000525_REGION_KEYS, candidate_sampling_regions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_coffee_can_center_regions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    layout_key: str,
    target_region: str | None = None,
    sample_positions: bool = True,
    shuffle_distractors: bool = False,
) -> None:
    """Place orange in one A-D region and distribute the three distractors.

    Entity identities and appearances remain stable. Root poses determine
    which region each entity occupies, so target observations continue to
    refer to coffee_can_orange.
    """

    configured_region = target_region or getattr(
        env, "_task525_target_region", "C"
    )
    configured_region = validate_region_key(configured_region)
    regions = {
        region.region_key: region
        for region in candidate_sampling_regions(layout_key)
    }
    resolved: dict[int, dict[str, object]] = {}

    for env_id_tensor in env_ids.reshape(-1):
        env_id = int(env_id_tensor.item())
        arrangement_seed = int(
            torch.randint(0, 2**31 - 1, (1,), device=env.device).item()
        )
        arrangement = make_coffee_arrangement(
            configured_region,
            shuffle_distractors=shuffle_distractors,
            rng=Random(arrangement_seed),
        )
        for region_key in TASK000525_REGION_KEYS:
            region = regions[region_key]
            object_name = arrangement.region_to_object[region_key]
            asset = env.scene[object_name]
            root_pose = asset.data.default_root_state[
                env_id : env_id + 1, :7
            ].clone()
            if sample_positions:
                sample = torch.rand((1, 2), device=asset.device)
                x = region.x_min_back_m + sample[0, 0] * (
                    region.x_max_front_m - region.x_min_back_m
                )
                y = region.y_min_m + sample[0, 1] * (
                    region.y_max_m - region.y_min_m
                )
            else:
                x, y, _ = region.default_position_m
            root_pose[:, 0] = x + env.scene.env_origins[env_id, 0]
            root_pose[:, 1] = y + env.scene.env_origins[env_id, 1]
            root_pose[:, 2] = (
                region.default_position_m[2]
                + env.scene.env_origins[env_id, 2]
            )
            one_env_id = env_id_tensor.reshape(1)
            asset.write_root_pose_to_sim(root_pose, env_ids=one_env_id)
            asset.write_root_velocity_to_sim(
                torch.zeros(
                    (1, 6), dtype=root_pose.dtype, device=asset.device
                ),
                env_ids=one_env_id,
            )
        resolved[env_id] = {
            "target_region": arrangement.target_region,
            "manipulation_side": arrangement.manipulation_side,
            "region_to_object": dict(arrangement.region_to_object),
        }

    existing = dict(getattr(env, "_task525_arrangements", {}))
    existing.update(resolved)
    env._task525_arrangements = existing
