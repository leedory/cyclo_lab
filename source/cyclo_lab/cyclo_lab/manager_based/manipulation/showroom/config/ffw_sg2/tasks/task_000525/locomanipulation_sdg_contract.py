"""Task000525 constants shared by the locomanipulation SDG adapter.

The destination pose is an experiment-only docking parent frame.  It is not a
production-approved table approach pose.
"""

from __future__ import annotations

import math


SDG_WORKING_ACTION_DIM = 22
SDG_FRAME_POSE_DIM = 7
SDG_GRIPPER_DIM = 1
SDG_BASE_ACTION_DIM = 3

SOURCE_FIXTURE_POSE_WXYZ = (
    -2.100731923587504,
    0.8710273489268139,
    0.8048447767505422,
    1.0,
    0.0,
    0.0,
    0.0,
)
CANDIDATE_BASE_GOAL_XYYAW = (-1.27, -0.66, 0.0)
DESTINATION_DOCKING_PARENT_POSE_WXYZ = (
    -0.640648076412496,
    -0.7551893883136659,
    0.8048447767505422,
    0.0,
    0.0,
    0.0,
    1.0,
)

SG2_DEPLOYED_FOOTPRINT_RADIUS_M = 0.44
SDG_PLANNING_MARGIN_M = 0.025
UPSTREAM_FINAL_BUFFER_M = 0.15
STATIC_MAP_PREFILL_BUFFER_M = (
    SG2_DEPLOYED_FOOTPRINT_RADIUS_M
    + SDG_PLANNING_MARGIN_M
    - UPSTREAM_FINAL_BUFFER_M
)

SHOWROOM_STATIC_OBSTACLE_AABBS = (
    ("bar_stool_1", 0.8216584060, -2.4823148398, 1.2943184322, -2.0095616887),
    ("bar_stool_2", 1.5404205962, -1.7287156575, 1.9983644064, -1.2707940832),
    ("bar_table", 1.2821440704, -2.4512196831, 1.8561440654, -1.7510950021),
    ("central_dining_set", -0.7882308864, -1.4717081895, 0.6138585902, -0.0763553491),
    ("kolbjorn_cabinet_02", -2.2876594955, 0.4671557233, -1.9138043517, 1.2748989746),
    ("kolbjorn_cabinet_1", -2.2876594955, 1.2773309794, -1.9138043517, 2.0850742308),
    ("right_lounge_chair_back", 0.1185759306, 1.6424744089, 0.8065983653, 2.4766281803),
    ("right_side_table_front", 1.0402440478, 1.5537955384, 1.5353380491, 2.0486007850),
    ("small_chair", -2.2834509999, -1.3897648506, -1.9230000466, -0.9392102115),
    ("vihals_low_cabinet", -2.2856502979, -0.9336060004, -1.7959557026, 0.4613977953),
)


def wrap_to_pi(angle: float) -> float:
    """Normalize a yaw angle to [-pi, pi]."""

    return math.atan2(math.sin(angle), math.cos(angle))


def expected_body_direction(command_index: int) -> tuple[int, float]:
    """Map +vx/+vy/+wz to the measured body x/body y/yaw sign check."""

    if command_index not in (0, 1, 2):
        raise ValueError(f"base command index must be 0, 1, or 2, got {command_index}")
    return command_index, 1.0
