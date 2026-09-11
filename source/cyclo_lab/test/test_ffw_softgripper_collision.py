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

"""Unit tests for the shared FFW soft-gripper collision approximation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).parents[1]
    / "cyclo_lab"
    / "assets"
    / "robots"
    / "ffw_softgripper_geometry.py"
)
_SPEC = importlib.util.spec_from_file_location("ffw_softgripper_geometry", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
FFW_SOFTGRIPPER_TIP_DEFLECTION_FRACTION = (
    _MODULE.FFW_SOFTGRIPPER_TIP_DEFLECTION_FRACTION
)
_banana_points = _MODULE.banana_points


@pytest.mark.parametrize(("finger", "expected_sign"), [("l2", 1.0), ("r2", -1.0)])
def test_banana_points_bends_only_toward_grasp(finger: str, expected_sign: float):
    finger_length = 96.60673142
    points = np.asarray(
        [
            [-4.0, -2.0, 0.0],
            [4.0, -2.0, 0.0],
            [-4.0, 2.0, 0.0],
            [4.0, 2.0, 0.0],
            [-4.0, -2.0, finger_length],
            [4.0, -2.0, finger_length],
            [-4.0, 2.0, finger_length],
            [4.0, 2.0, finger_length],
        ],
        dtype=np.float64,
    )

    deformed, long_axis, normal_axis, tip_deflection = _banana_points(points, finger)

    assert long_axis == 2
    assert normal_axis == 1
    assert tip_deflection == pytest.approx(1.5)
    assert deformed[:4] == pytest.approx(points[:4])
    assert deformed[4:, normal_axis] == pytest.approx(
        points[4:, normal_axis] + expected_sign * 1.5
    )
    assert FFW_SOFTGRIPPER_TIP_DEFLECTION_FRACTION == pytest.approx(
        1.5 / finger_length
    )
