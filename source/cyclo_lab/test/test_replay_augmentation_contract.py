"""Static and pure-function contracts for action-replay augmentation."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest


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

    def test_sg2_act_camera_contract_preserves_captured_pixel_layout(self):
        module = load_converter()

        for camera in (
            "cam_left_head",
            "cam_left_wrist",
            "cam_right_wrist",
        ):
            self.assertEqual(
                module.required_camera_rotation(
                    "ffw_sg2_rev1",
                    camera,
                    input_height=480,
                    input_width=640,
                ),
                0,
            )

    def test_sg2_act_camera_contract_rejects_rotated_wrist_shape(self):
        module = load_converter()

        with self.assertRaisesRegex(
            module.ConversionError, "camera shape cannot satisfy"
        ):
            module.required_camera_rotation(
                "ffw_sg2_rev1",
                "cam_left_wrist",
                input_height=640,
                input_width=480,
            )


if __name__ == "__main__":
    unittest.main()
