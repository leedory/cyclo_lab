"""Pure regression tests for Task525 canonical camera pixels."""

import ast
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
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
STAGING_AUDIT = COMMON.with_name("audit_policy_staging.py")
EXPORT = COMMON.with_name("export_native_policy_staging.py")
REPLAY = COMMON.with_name("replay_visual_policy_staging.py")


def load_common():
    spec = spec_from_file_location("task_000525_policy_staging_common", COMMON)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_staging_audit():
    module_name = "task_000525_audit_policy_staging"
    spec = spec_from_file_location(module_name, STAGING_AUDIT)
    assert spec is not None and spec.loader is not None
    sys.path.insert(0, str(STAGING_AUDIT.parent))
    try:
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
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

    def test_camera_subset_preserves_requested_order_and_rejects_bad_lists(self):
        module = load_common()

        self.assertEqual(
            module.select_camera_map(("cam_head", "cam_wrist_right")),
            {
                "cam_head": "cam_left_head",
                "cam_wrist_right": "cam_right_wrist",
            },
        )
        with self.assertRaisesRegex(module.Task525PolicyDataError, "non-empty"):
            module.select_camera_map(())
        with self.assertRaisesRegex(module.Task525PolicyDataError, "duplicates"):
            module.select_camera_map(("cam_head", "cam_head"))
        with self.assertRaisesRegex(module.Task525PolicyDataError, "unsupported"):
            module.select_camera_map(("cam_chin",))

    def test_native_and_visual_staging_apply_selected_camera_contract(self):
        export_source = EXPORT.read_text(encoding="utf-8")
        replay_source = REPLAY.read_text(encoding="utf-8")

        for source in (export_source, replay_source):
            self.assertIn('"--camera-names"', source)
            self.assertIn('nargs="+"', source)
            self.assertIn("select_camera_map(args.camera_names)", source)
            self.assertIn('"camera_map": camera_map', source)
            self.assertIn('"camera_shapes_h_w": camera_shapes', source)
            self.assertIn('"camera_rotation_deg": camera_rotations', source)
        self.assertIn("for camera_name in camera_names", replay_source)
        self.assertIn("for source_camera in camera_map", replay_source)
        self.assertIn('"camera_map",', replay_source)

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


class TestTask525PolicyStagingAudit(unittest.TestCase):
    def make_manifest(self, module, root: Path, source_cameras):
        camera_map = {
            camera: module.CANONICAL_CAMERA_MAP[camera]
            for camera in source_cameras
        }
        episode_directory = root / "videos" / "episode_000000"
        episode_directory.mkdir(parents=True)
        videos = {}
        for output_camera in camera_map.values():
            relative = (
                Path("videos") / "episode_000000" / f"{output_camera}.mp4"
            )
            (root / relative).write_bytes(output_camera.encode())
            videos[output_camera] = str(relative)
        return {
            "camera_map": camera_map,
            "camera_shapes_h_w": {
                camera: list(module.CANONICAL_CAMERA_SHAPES[camera])
                for camera in source_cameras
            },
            "camera_rotation_deg": {
                camera: module.CAMERA_ROTATION_DEG[camera]
                for camera in source_cameras
            },
            "episodes": [{"episode_index": 0, "videos": videos}],
        }

    def valid_appearance(self):
        sampled = {
            "coffee_can_black": "brown",
            "coffee_can_brown": "green",
            "coffee_can_green": "black",
        }
        return {
            "coffee_can_distractor_appearance": {
                "protected_target": {
                    "object_name": "coffee_can_orange",
                    "appearance": "orange",
                },
                "distractor_mapping": {
                    object_name: {
                        "authored_appearance": object_name.removeprefix(
                            "coffee_can_"
                        ),
                        "sampled_appearance": appearance,
                        "bound_material_path": (
                            f"/World/envs/env_0/{object_name}/Looks/{appearance}"
                        ),
                    }
                    for object_name, appearance in sampled.items()
                },
            }
        }

    def test_accepts_exact_head_only_native_and_augmented_camera_contract(self):
        module = load_staging_audit()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            native_root = root / "native"
            augmented_root = root / "augmented"
            native = self.make_manifest(module, native_root, ("cam_head",))
            augmented = self.make_manifest(module, augmented_root, ("cam_head",))

            self.assertEqual(
                module.validate_matching_camera_contracts(
                    native_root, native, augmented_root, augmented
                ),
                {"cam_head": "cam_left_head"},
            )

            other_root = root / "other"
            other = self.make_manifest(module, other_root, ("cam_wrist_left",))
            with self.assertRaisesRegex(module.AuditError, "do not match exactly"):
                module.validate_matching_camera_contracts(
                    native_root, native, other_root, other
                )

    def test_rejects_video_map_or_files_outside_head_only_subset(self):
        module = load_staging_audit()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.make_manifest(module, root, ("cam_head",))
            bad_map = deepcopy(manifest)
            bad_map["episodes"][0]["videos"]["cam_left_wrist"] = (
                "videos/episode_000000/cam_left_wrist.mp4"
            )
            with self.assertRaisesRegex(module.AuditError, "video map"):
                module.validate_camera_contract(root, bad_map, "native")

            (root / "videos" / "episode_000000" / "extra.mp4").write_bytes(b"x")
            with self.assertRaisesRegex(module.AuditError, "video files"):
                module.validate_camera_contract(root, manifest, "native")

    def test_requires_protected_orange_and_exact_distractor_material_permutation(self):
        module = load_staging_audit()
        valid = self.valid_appearance()
        module.validate_distractor_appearance(valid)

        invalid_cases = []
        unprotected = deepcopy(valid)
        unprotected["coffee_can_distractor_appearance"]["protected_target"][
            "appearance"
        ] = "black"
        invalid_cases.append((unprotected, "protect canonical orange"))

        missing = deepcopy(valid)
        del missing["coffee_can_distractor_appearance"]["distractor_mapping"][
            "coffee_can_green"
        ]
        invalid_cases.append((missing, "cover exactly"))

        not_permuted = deepcopy(valid)
        not_permuted["coffee_can_distractor_appearance"]["distractor_mapping"][
            "coffee_can_green"
        ]["sampled_appearance"] = "brown"
        not_permuted["coffee_can_distractor_appearance"]["distractor_mapping"][
            "coffee_can_green"
        ]["bound_material_path"] = (
            "/World/envs/env_0/coffee_can_green/Looks/brown"
        )
        invalid_cases.append((not_permuted, "exact black/brown/green permutation"))

        wrong_authored = deepcopy(valid)
        wrong_authored["coffee_can_distractor_appearance"]["distractor_mapping"][
            "coffee_can_black"
        ]["authored_appearance"] = "green"
        invalid_cases.append((wrong_authored, "authored appearance"))

        wrong_binding = deepcopy(valid)
        wrong_binding["coffee_can_distractor_appearance"]["distractor_mapping"][
            "coffee_can_black"
        ]["bound_material_path"] = "/World/wrong/Looks/brown"
        invalid_cases.append((wrong_binding, "bound material path"))

        for evidence, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(module.AuditError, message):
                    module.validate_distractor_appearance(evidence)


if __name__ == "__main__":
    unittest.main()
