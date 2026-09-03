"""Static and pure-function contracts for action-replay augmentation."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


REPO = Path(__file__).resolve().parents[3]
REPLAY = (
    REPO
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "tasks"
    / "task_000458"
    / "replay_with_randomization.py"
)
CONVERTER = (
    REPO
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "data_converter"
    / "replay_staging_to_lerobot_v30.py"
)
PROFILE = (
    REPO
    / "source"
    / "cyclo_lab"
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "task_000458"
    / "profiles.py"
)


def load_converter():
    spec = spec_from_file_location("replay_staging_to_lerobot_v30", CONVERTER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReplayAugmentationContractTest(unittest.TestCase):
    def test_action_replay_is_profile_driven_and_steps_physics(self):
        source = REPLAY.read_text(encoding="utf-8")
        self.assertIn("presence_events.randomize_non_target_presence(", source)
        self.assertNotIn(
            "appearance_events.randomize_non_target_presence(", source
        )
        self.assertIn('"--randomization-profile",', source)
        self.assertIn("prepare_sg2_position_replay_state(", source)
        self.assertIn("restore_sg2_replay_root_pose(", source)
        self.assertIn("env.reset_to(initial_state", source)
        self.assertIn("env.step(torch.as_tensor(action_batch", source)
        self.assertIn("source initial_state + source actions -> Isaac physics", source)

    def test_augment_profile_does_not_move_target_or_robot(self):
        source = PROFILE.read_text(encoding="utf-8")
        block = source.split("TASK000458_AUGMENT_RANDOM =", 1)[1]
        self.assertIn("presence=PresenceRandomizationCfg", block)
        self.assertIn("lighting=LightingRandomizationCfg", block)
        self.assertIn("shelf=ShelfAppearanceRandomizationCfg", block)
        self.assertIn("wall=WallAppearanceRandomizationCfg", block)
        self.assertIn("camera=CameraRandomizationCfg", block)
        self.assertNotIn("target_pose=", block)
        self.assertNotIn("robot_root=", block)

    def test_name_based_action_reorder(self):
        module = load_converter()
        source = ["a", "lift", "head_1", "head_2"]
        target = ["a", "head_1", "head_2", "lift"]
        self.assertEqual(module.build_reorder_indices(source, target), [0, 2, 3, 1])

    def test_name_based_action_reorder_rejects_mismatch(self):
        module = load_converter()
        with self.assertRaises(module.ConversionError):
            module.build_reorder_indices(["a", "b"], ["a", "c"])

    def test_state_and_action_projection_are_independent_and_ordered(self):
        module = load_converter()
        with tempfile.TemporaryDirectory() as temp_dir:
            staging = Path(temp_dir)
            arrays_path = staging / "episode.npz"
            np.savez(
                arrays_path,
                observation_state=np.asarray(
                    [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
                    dtype=np.float32,
                ),
                action=np.asarray(
                    [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
                    dtype=np.float32,
                ),
                timestamp_s=np.asarray([0.0, 1.0 / 15.0], dtype=np.float64),
            )
            record = {
                "arrays": arrays_path.name,
                "array_sha256": module.sha256(arrays_path),
                "length": 2,
                "state_names": ["s0", "s1", "s2"],
                "action_names": ["a0", "a1", "a2"],
            }

            state, action, _, state_names, action_names = module.load_episode_arrays(
                staging,
                record,
                requested_state_names=["s2", "s0"],
                requested_action_names=["a1"],
            )

        np.testing.assert_array_equal(
            state,
            np.asarray([[12.0, 10.0], [22.0, 20.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            action,
            np.asarray([[101.0], [201.0]], dtype=np.float32),
        )
        self.assertEqual(state_names, ["s2", "s0"])
        self.assertEqual(action_names, ["a1"])

    def test_projection_validation_rejects_empty_duplicate_or_missing_names(self):
        module = load_converter()
        with self.assertRaisesRegex(module.ConversionError, "must not be empty"):
            module.build_projection_indices(["a"], [], field="action")
        with self.assertRaisesRegex(module.ConversionError, "duplicates"):
            module.build_projection_indices(
                ["a", "b"], ["a", "a"], field="action"
            )
        with self.assertRaisesRegex(module.ConversionError, "not found in source"):
            module.build_projection_indices(["a", "b"], ["c"], field="action")

    def test_projection_defaults_to_all_source_names(self):
        module = load_converter()
        names, indices = module.build_projection_indices(
            ["second", "first"], None, field="state"
        )
        self.assertEqual(names, ["second", "first"])
        self.assertEqual(indices, [0, 1])

    def test_cli_accepts_independent_projection_lists(self):
        module = load_converter()
        argv = [
            "converter",
            "--staging", "/tmp/staging",
            "--output", "/tmp/output",
            "--repo-id", "robotis/smoke",
            "--state-names", "s2", "s0",
            "--action-names", "a1",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = module.parse_args()
        self.assertEqual(args.state_names, ["s2", "s0"])
        self.assertEqual(args.action_names, ["a1"])

    def test_write_dataset_propagates_projected_names_and_shapes(self):
        module = load_converter()
        captured = {}

        class FakeEpisodeData:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeWriter:
            def __init__(self, config):
                self.config = config

            def write_from_episodes(self, episodes):
                captured["episodes"] = episodes
                (Path(self.config.output_dir) / "meta").mkdir(parents=True)
                return True

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            staging = root / "staging"
            output = root / "dataset"
            staging.mkdir()
            arrays_path = staging / "episode.npz"
            np.savez(
                arrays_path,
                observation_state=np.asarray(
                    [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
                    dtype=np.float32,
                ),
                action=np.asarray(
                    [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
                    dtype=np.float32,
                ),
                timestamp_s=np.asarray([0.0, 1.0 / 15.0], dtype=np.float64),
            )
            record = {
                "episode_index": 0,
                "length": 2,
                "arrays": arrays_path.name,
                "array_sha256": module.sha256(arrays_path),
                "state_names": ["s0", "s1", "s2"],
                "action_names": ["a0", "a1", "a2"],
                "videos": {"cam_left_head": "head.mp4"},
            }
            manifest = {
                "schema": module.STAGING_SCHEMA,
                "action_semantics": "pre_step_raw_absolute_joint_position_command",
                "observation_semantics": "current_state",
                "episode_count": 1,
                "source_episode_count": 1,
                "fps": 15,
                "camera_map": {"head": "cam_left_head"},
                "source_hdf": "/source.hdf5",
                "source_hdf_sha256": "source-hash",
                "repeats": 1,
                "randomization_profile": "test.profile",
                "render_contract": {"test": True},
                "task": "test task",
                "episodes": [record],
            }
            (staging / "manifest.json").write_text(json.dumps(manifest))

            base_module = types.ModuleType("cyclo_data.converter.base_converter")
            base_module.EpisodeData = FakeEpisodeData
            writer_module = types.ModuleType("cyclo_data.converter.to_lerobot_v30")
            writer_module.RosbagToLerobotV30Converter = FakeWriter
            writer_module.V30ConversionConfig = FakeConfig
            fake_modules = {
                "cyclo_data": types.ModuleType("cyclo_data"),
                "cyclo_data.converter": types.ModuleType("cyclo_data.converter"),
                "cyclo_data.converter.base_converter": base_module,
                "cyclo_data.converter.to_lerobot_v30": writer_module,
            }
            video_check = {
                "path": str(staging / "head.mp4"),
                "codec": "h264",
                "frames": 2,
                "width": 672,
                "height": 376,
                "fps": 15.0,
                "sha256": "video-hash",
            }
            with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
                module, "verify_video", return_value=video_check
            ):
                module.write_dataset(
                    staging,
                    output,
                    "robotis/smoke",
                    "ffw_sg2_rev1",
                    root,
                    1,
                    state_names=["s2", "s0"],
                    action_names=["a1"],
                )

            episode = captured["episodes"][0]
            np.testing.assert_array_equal(
                np.asarray(episode.observation_state),
                np.asarray([[12.0, 10.0], [22.0, 20.0]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                np.asarray(episode.action),
                np.asarray([[101.0], [201.0]], dtype=np.float32),
            )
            self.assertEqual(episode.observation_state_names, ["s2", "s0"])
            self.assertEqual(episode.action_names, ["a1"])
            provenance = json.loads(
                (output / "meta" / module.SIDECAR_NAME).read_text()
            )
            self.assertEqual(provenance["state_names"], ["s2", "s0"])
            self.assertEqual(provenance["action_names"], ["a1"])
            self.assertEqual(provenance["source_state_names"], ["s0", "s1", "s2"])
            self.assertEqual(provenance["source_action_names"], ["a0", "a1", "a2"])

    def test_sg2_act_camera_contract_accepts_only_canonical_shapes(self):
        module = load_converter()

        expected = {
            "cam_left_head": (376, 672),
            "cam_left_wrist": (640, 480),
            "cam_right_wrist": (640, 480),
        }
        self.assertEqual(module.CAMERA_SHAPE_CONTRACTS["ffw_sg2_rev1"], expected)
        for camera, (height, width) in expected.items():
            self.assertIsNone(
                module.validate_camera_shape(
                    "ffw_sg2_rev1",
                    camera,
                    input_height=height,
                    input_width=width,
                )
            )

    def test_sg2_act_camera_contract_rejects_transposed_or_legacy_shapes(self):
        module = load_converter()

        with self.assertRaisesRegex(
            module.ConversionError, "rotation and resizing are not permitted"
        ):
            module.validate_camera_shape(
                "ffw_sg2_rev1",
                "cam_left_wrist",
                input_height=480,
                input_width=640,
            )
        with self.assertRaisesRegex(
            module.ConversionError, "rotation and resizing are not permitted"
        ):
            module.validate_camera_shape(
                "ffw_sg2_rev1",
                "cam_left_head",
                input_height=480,
                input_width=640,
            )


if __name__ == "__main__":
    unittest.main()
