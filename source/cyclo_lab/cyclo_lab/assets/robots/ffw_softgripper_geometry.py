# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure geometry helpers for FFW soft-gripper collision proxies."""

from __future__ import annotations

import numpy as np


# Candidate 21 bends the SG2 fingertip 1.50 mm over its measured mesh span.
FFW_SOFTGRIPPER_TIP_DEFLECTION_FRACTION = 1.50 / 96.60673142


def banana_points(
    points: np.ndarray, finger: str
) -> tuple[np.ndarray, int, int, float]:
    """Bend a distal mesh gently toward the gripper's grasping center."""

    spans = np.ptp(points, axis=0)
    long_axis = int(np.argmax(spans))
    long_span = float(spans[long_axis])
    if long_span <= 0.0:
        raise RuntimeError("Distal gripper mesh has zero longitudinal span")

    cross_axes = [axis for axis in range(3) if axis != long_axis]
    width_axis = min(
        cross_axes,
        key=lambda axis: abs(float(points[:, axis].min() + points[:, axis].max()))
        / max(float(spans[axis]), 1.0e-12),
    )
    normal_axis = next(axis for axis in cross_axes if axis != width_axis)
    if finger == "l2":
        inward_sign = 1.0
    elif finger == "r2":
        inward_sign = -1.0
    else:
        raise RuntimeError(f"Unsupported distal finger label: {finger}")

    lower = float(points[:, long_axis].min())
    longitudinal_t = np.clip((points[:, long_axis] - lower) / long_span, 0.0, 1.0)
    tip_deflection = FFW_SOFTGRIPPER_TIP_DEFLECTION_FRACTION * long_span
    deformed = points.copy()
    deformed[:, normal_axis] += inward_sign * tip_deflection * longitudinal_t**2
    return deformed, long_axis, normal_axis, tip_deflection
