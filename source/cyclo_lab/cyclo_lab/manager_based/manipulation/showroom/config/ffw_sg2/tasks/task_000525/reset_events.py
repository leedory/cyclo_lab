"""Task000525 reset events for independent per-region coffee-can sampling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .layout import candidate_sampling_regions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_coffee_can_center_regions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    layout_key: str,
) -> None:
    """Sample each upright can center uniformly inside its own rectangle.

    Appearance, region assignment, upright orientation, and support height are
    unchanged.  Only world X/Y are sampled, independently for every can and
    environment.
    """

    for region in candidate_sampling_regions(layout_key):
        asset = env.scene[region.object_name]
        samples = torch.rand((len(env_ids), 2), device=asset.device)
        root_pose = asset.data.default_root_state[env_ids, :7].clone()
        root_pose[:, 0] = (
            region.x_min_back_m
            + samples[:, 0] * (region.x_max_front_m - region.x_min_back_m)
            + env.scene.env_origins[env_ids, 0]
        )
        root_pose[:, 1] = (
            region.y_min_m
            + samples[:, 1] * (region.y_max_m - region.y_min_m)
            + env.scene.env_origins[env_ids, 1]
        )
        root_pose[:, 2] = (
            region.default_position_m[2] + env.scene.env_origins[env_ids, 2]
        )
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(
            torch.zeros(
                (len(env_ids), 6), dtype=root_pose.dtype, device=asset.device
            ),
            env_ids=env_ids,
        )
