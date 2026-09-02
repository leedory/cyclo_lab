"""Pure regression tests for Task525 canonical camera pixels."""

import ast
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[3]
COMMON = (
    REPO
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "tasks"
    / "task_000525"
    / "policy_staging_common.py"
)
AUDIT = COMMON.with_name("audit_lerobot_v30.py")
EXPORT = COMMON.with_name("export_native_policy_staging.py")


def load_common():
    spec = spec_from_file_location("task_000525_policy_staging_common", COMMON)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTask525CameraContract(unittest.TestCase):
    def test_canonical_frames_keep_exact_head_and_portrait_wrist_shapes(self):
        module = load_common()
        expected = {
            "cam_head": (376, 672),
            "cam_wrist_left": (640, 480),
            "cam_wrist_right": (640, 480),
        }

        self.assertEqual(module.CANONICAL_CAMERA_SHAPES, expected)
        self.assertEqual(module.CAMERA_ROTATION_DEG, dict.fromkeys(expected, 0))
        for camera, shape in expected.items():
            frame = np.zeros((*shape, 3), dtype=np.uint8)
            actual = module.canonicalize_camera_frame(camera, frame)
            self.assertEqual(actual.shape, (*shape, 3))

    def test_transposed_frames_are_rejected_not_rotated(self):
        module = load_common()

        with self.assertRaisesRegex(
            module.Task525PolicyDataError,
            "rotation and resizing are not permitted",
        ):
            module.canonicalize_camera_frame(
                "cam_wrist_left", np.zeros((480, 640, 3), dtype=np.uint8)
            )
        with self.assertRaisesRegex(
            module.Task525PolicyDataError,
            "rotation and resizing are not permitted",
        ):
            module.canonicalize_camera_frame(
                "cam_head", np.zeros((480, 640, 3), dtype=np.uint8)
            )

    def test_export_writer_and_final_audit_use_per_camera_shapes(self):
        export_source = EXPORT.read_text(encoding="utf-8")
        self.assertIn("height, width = CANONICAL_CAMERA_SHAPES[source_camera]", export_source)
        self.assertIn("float(fps), (width, height)", export_source)

        audit_source = AUDIT.read_text(encoding="utf-8")
        tree = ast.parse(audit_source)
        contract = None
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if any(
                isinstance(target, ast.Name)
                and target.id == "CAMERA_SHAPES_H_W"
                for target in node.targets
            ):
                contract = ast.literal_eval(node.value)
                break
        self.assertEqual(
            contract,
            {
                "cam_left_head": (376, 672),
                "cam_left_wrist": (640, 480),
                "cam_right_wrist": (640, 480),
            },
        )
        self.assertIn("height, width = CAMERA_SHAPES_H_W[camera]", audit_source)
        self.assertIn("probe_video(path, 15, CAMERA_SHAPES_H_W[camera])", audit_source)


if __name__ == "__main__":
    unittest.main()
