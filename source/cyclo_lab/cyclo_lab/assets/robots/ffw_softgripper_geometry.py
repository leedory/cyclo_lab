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


def softgripper_source_points(points: np.ndarray) -> tuple[np.ndarray, int]:
    """Use the soft-gripper USD mesh without changing its finger shape."""

    spans = np.ptp(points, axis=0)
    long_axis = int(np.argmax(spans))
    long_span = float(spans[long_axis])
    if long_span <= 0.0:
        raise RuntimeError("Distal gripper mesh has zero longitudinal span")

    return points.copy(), long_axis
