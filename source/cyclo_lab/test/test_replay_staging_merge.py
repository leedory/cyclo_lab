"""Pure filesystem tests for replay-staging merging."""

from __future__ import annotations

import ast
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "data_converter"
    / "merge_replay_staging.py"
)


CONVERTER_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "data_converter"
    / "replay_staging_to_lerobot_v30.py"
)


AUDIT_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "tasks"
    / "task_000525"
    / "audit_lerobot_v30.py"
)


TASK458_STAGING_PRODUCER_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "data_converter"
    / "isaac_hdf5_to_replay_staging.py"
)


TASK525_STAGING_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "tasks"
    / "task_000525"
    / "policy_staging_common.py"
)


def read_producer_string_constant(path: Path, constant_name: str) -> str:
    """Read a producer-owned string contract without importing runtime dependencies."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if not any(
            isinstance(target, ast.Name) and target.id == constant_name
            for target in targets
        ):
            continue
        if isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            return statement.value.value
    raise AssertionError(f"missing string constant {constant_name} in {path}")


TASK458_PRODUCER_ACTION_SEMANTICS = read_producer_string_constant(
    TASK458_STAGING_PRODUCER_PATH, "ACTION_SEMANTICS"
)
TASK525_PRODUCER_ACTION_SEMANTICS = read_producer_string_constant(
    TASK525_STAGING_CONTRACT_PATH, "POLICY_ACTION_SEMANTICS"
)


def load_module():
    spec = spec_from_file_location("merge_replay_staging", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_converter_module():
    spec = spec_from_file_location(
        "replay_staging_to_lerobot_v30", CONVERTER_MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_audit_module():
    spec = spec_from_file_location("task525_audit_lerobot_v30", AUDIT_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReplayStagingMergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()
        cls.converter = load_converter_module()

    def make_staging(
        self,
        root: Path,
        name: str,
        source_episode: str,
        *,
        action_semantics: str = TASK525_PRODUCER_ACTION_SEMANTICS,
        include_camera_geometry: bool = True,
    ) -> Path:
        staging = root / name
        array = staging / "policy_arrays" / "episode_000000.npz"
        video = staging / "videos" / "episode_000000" / "cam_left_head.mp4"
        array.parent.mkdir(parents=True)
        video.parent.mkdir(parents=True)
        array.write_bytes((name + "-array").encode())
        video.write_bytes((name + "-video").encode())
        record = {
            "episode_index": 0,
            "source_episode": source_episode,
            "length": 3,
            "arrays": str(array.relative_to(staging)),
            "array_sha256": self.module.sha256(array),
            "videos": {"cam_left_head": str(video.relative_to(staging))},
            "state_names": ["joint"],
            "action_names": ["joint"],
        }
        manifest = {
            "schema": self.module.SCHEMA,
            "action_semantics": action_semantics,
            "fps": 30,
            "camera_map": {"cam_head": "cam_left_head"},
            "task_instruction": "test",
            "source_hdf": name + ".hdf5",
            "source_hdf_sha256": name,
            "randomization_profile": name,
            "episode_count": 1,
            "total_frames": 3,
            "episodes": [record],
        }
        if include_camera_geometry:
            manifest.update(
                {
                    "camera_shapes_h_w": {"cam_head": [376, 672]},
                    "camera_rotation_deg": {"cam_head": 0},
                }
            )
        (staging / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return staging

    def test_merges_without_decoding_media(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.make_staging(root, "first", "demo_0")
            second = self.make_staging(root, "second", "demo_1")
            output = root / "merged"

            result = self.module.merge([first, second], output)

            self.assertEqual(result["episodes"], 2)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["episode_count"], 2)
            self.assertEqual(
                [record["episode_index"] for record in manifest["episodes"]],
                [0, 1],
            )
            self.assertEqual(sum(result["transfer_modes"].values()), 4)
            self.assertEqual(
                manifest["camera_map"], {"cam_head": "cam_left_head"}
            )
            self.assertEqual(
                manifest["camera_shapes_h_w"], {"cam_head": [376, 672]}
            )
            self.assertEqual(manifest["camera_rotation_deg"], {"cam_head": 0})
            self.assertEqual(
                set(manifest["episodes"][0]["videos"]), {"cam_left_head"}
            )
            self.assertEqual(
                manifest["action_semantics"], TASK525_PRODUCER_ACTION_SEMANTICS
            )
            converted_manifest = self.converter.load_manifest(output, 2)
            self.assertEqual(
                converted_manifest["action_semantics"],
                TASK525_PRODUCER_ACTION_SEMANTICS,
            )

    def test_supported_semantics_are_exactly_the_current_producer_contracts(self):
        producer_semantics = {
            TASK458_PRODUCER_ACTION_SEMANTICS,
            TASK525_PRODUCER_ACTION_SEMANTICS,
        }
        self.assertEqual(self.module.SUPPORTED_ACTION_SEMANTICS, producer_semantics)
        self.assertEqual(self.converter.SUPPORTED_ACTION_SEMANTICS, producer_semantics)

    def test_task458_semantics_and_optional_camera_geometry_are_preserved(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self.make_staging(
                root,
                "first",
                "demo_0",
                action_semantics=TASK458_PRODUCER_ACTION_SEMANTICS,
                include_camera_geometry=False,
            )
            second = self.make_staging(
                root,
                "second",
                "demo_1",
                action_semantics=TASK458_PRODUCER_ACTION_SEMANTICS,
                include_camera_geometry=False,
            )
            output = root / "merged"

            self.module.merge([first, second], output)

            manifest = self.converter.load_manifest(output, 2)
            self.assertEqual(
                manifest["action_semantics"], TASK458_PRODUCER_ACTION_SEMANTICS
            )
            self.assertNotIn("camera_shapes_h_w", manifest)
            self.assertNotIn("camera_rotation_deg", manifest)

    def test_rejects_mixed_producer_semantics_before_materializing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            task458 = self.make_staging(
                root,
                "task458",
                "demo_0",
                action_semantics=TASK458_PRODUCER_ACTION_SEMANTICS,
                include_camera_geometry=False,
            )
            task525 = self.make_staging(
                root,
                "task525",
                "demo_1",
                action_semantics=TASK525_PRODUCER_ACTION_SEMANTICS,
                include_camera_geometry=False,
            )
            output = root / "merged"

            with self.assertRaisesRegex(self.module.MergeError, "action_semantics"):
                self.module.merge([task458, task525], output)
            self.assertFalse(output.exists())

    def test_rejects_name_changes_in_later_episode_before_materializing(self):
        for field in ("state_names", "action_names"):
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = self.make_staging(root, "source", "demo_0")
                manifest_path = source / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                second_record = dict(manifest["episodes"][0])
                second_record["episode_index"] = 1
                second_record[field] = ["changed_joint"]
                manifest["episodes"].append(second_record)
                manifest["episode_count"] = 2
                manifest["total_frames"] = 6
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                output = root / "merged"

                with self.assertRaisesRegex(
                    self.module.MergeError, rf"{field.removesuffix('_names')} names change"
                ):
                    self.module.merge([source], output)
                self.assertFalse(output.exists())

    def test_compares_state_and_action_names_across_inputs_before_materializing(self):
        for field in ("state_names", "action_names"):
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                root = Path(temporary)
                first = self.make_staging(root, "first", "demo_0")
                second = self.make_staging(root, "second", "demo_1")
                manifest_path = second / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["episodes"][0][field] = ["changed_joint"]
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                output = root / "merged"

                with self.assertRaisesRegex(
                    self.module.MergeError,
                    rf"incompatible {field.removesuffix('_names')} joint names",
                ):
                    self.module.merge([first, second], output)
                self.assertFalse(output.exists())

    def test_can_filter_by_source_episode(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.make_staging(root, "source", "demo_0")
            output = root / "filtered"
            with self.assertRaisesRegex(self.module.MergeError, "selection is empty"):
                self.module.merge(
                    [source],
                    output,
                    exclude_source_episodes={"demo_0"},
                )
            self.assertFalse(output.exists())


class Task525LerobotAuditContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_audit_module()

    def test_accepts_pick_18d_head_only_contract(self):
        selected = self.module.JOINT_NAMES[:18]
        self.assertEqual(
            self.module.validate_expected_subset(
                selected, self.module.JOINT_NAMES, "expected state names"
            ),
            tuple(selected),
        )
        self.assertEqual(
            self.module.validate_expected_subset(
                ["cam_left_head"], self.module.CAMERAS, "expected cameras"
            ),
            ("cam_left_head",),
        )
        self.module.validate_video_check_records(
            [
                {
                    "episode_index": 0,
                    "cameras": {
                        "cam_left_head": {
                            "height": 376,
                            "width": 672,
                            "fps": 15.0,
                            "frames": 7,
                        }
                    },
                }
            ],
            ["cam_left_head"],
            1,
            7,
            label="checks",
        )

    def test_rejects_noncanonical_order_and_extra_camera(self):
        with self.assertRaisesRegex(self.module.AuditError, "canonical order"):
            self.module.validate_expected_subset(
                ["head_joint2", "head_joint1"],
                self.module.JOINT_NAMES,
                "expected state names",
            )
        with self.assertRaisesRegex(self.module.AuditError, "camera set mismatch"):
            self.module.validate_video_check_records(
                [
                    {
                        "episode_index": 0,
                        "cameras": {
                            "cam_left_head": {
                                "height": 376,
                                "width": 672,
                                "fps": 15.0,
                                "frames": 7,
                            },
                            "cam_left_wrist": {},
                        },
                    }
                ],
                ["cam_left_head"],
                1,
                7,
                label="checks",
            )


if __name__ == "__main__":
    unittest.main()
