"""Unit tests for validated SG2 camera-profile loading."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import yaml

from cyclo_lab.robot_specs.ffw.sg2 import (
    DEFAULT_SG2_CAMERA_PROFILE,
    load_sg2_camera_profile,
)


class TestSG2CameraProfiles(unittest.TestCase):
    def test_default_1050_profile(self) -> None:
        profile = load_sg2_camera_profile()

        self.assertEqual(DEFAULT_SG2_CAMERA_PROFILE, "1050")
        self.assertEqual(profile.profile_id, "sg2-1050-2026-09-03")
        self.assertEqual(profile.robot_hostname, "ffw-SNPR48A1050")
        self.assertEqual(
            (profile.camera("head").height, profile.camera("head").width),
            (376, 672),
        )
        self.assertEqual(profile.camera("head").intrinsic_matrix[0], 365.1824645996094)
        self.assertEqual(profile.camera("head").serial, "11295797")
        self.assertEqual(profile.camera("wrist_left").serial, "335122270624")
        self.assertEqual(profile.camera("wrist_right").serial, "335122272052")
        self.assertEqual(len(profile.source_sha256), 64)

    def test_missing_camera_role_is_rejected(self) -> None:
        source = Path(__file__).parents[1] / "cyclo_lab/robot_specs/ffw/sg2/profiles/1050.yaml"
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        invalid = deepcopy(document)
        invalid["cameras"].pop("wrist_right")

        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_path = Path(temporary_directory) / "invalid.yaml"
            invalid_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain exactly"):
                load_sg2_camera_profile(invalid_path)

    def test_nonfinite_intrinsic_is_rejected(self) -> None:
        source = Path(__file__).parents[1] / "cyclo_lab/robot_specs/ffw/sg2/profiles/1050.yaml"
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        invalid = deepcopy(document)
        invalid["cameras"]["head"]["k"][0] = float("nan")

        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_path = Path(temporary_directory) / "invalid.yaml"
            invalid_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be finite"):
                load_sg2_camera_profile(invalid_path)

    def test_duplicate_wrist_serial_is_rejected(self) -> None:
        source = Path(__file__).parents[1] / "cyclo_lab/robot_specs/ffw/sg2/profiles/1050.yaml"
        invalid = yaml.safe_load(source.read_text(encoding="utf-8"))
        invalid["cameras"]["wrist_right"]["serial"] = invalid["cameras"]["wrist_left"]["serial"]

        with tempfile.TemporaryDirectory() as temporary_directory:
            invalid_path = Path(temporary_directory) / "invalid.yaml"
            invalid_path.write_text(yaml.safe_dump(invalid), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different physical serials"):
                load_sg2_camera_profile(invalid_path)


if __name__ == "__main__":
    unittest.main()
