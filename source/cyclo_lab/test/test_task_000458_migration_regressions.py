"""Regression contracts retained across the SG2 showroom file migration."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import unittest


REPO = Path(__file__).resolve().parents[3]
ROOT = (
    Path(__file__).resolve().parents[1]
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
)
TASK = ROOT / "tasks" / "task_000458"
RANDOMIZATION = ROOT / "randomization"


class Task000458MigrationRegressionTest(unittest.TestCase):
    def test_generation_profile_retains_enabled_axes(self):
        source = (RANDOMIZATION / "cfg.py").read_text()
        profile = (TASK / "profiles.py").read_text()
        compact_profile = "".join(profile.split())
        for class_name in (
            "TargetPoseRandomizationCfg",
            "PresenceRandomizationCfg",
            "LightingRandomizationCfg",
            "ShelfAppearanceRandomizationCfg",
            "WallAppearanceRandomizationCfg",
            "CameraRandomizationCfg",
        ):
            block = source.split(f"class {class_name}", 1)[1].split("@dataclass", 1)[0]
            self.assertIn("enabled: bool = False", block, class_name)
            self.assertIn(f"{class_name}(enabled=True", compact_profile, class_name)

        robot_block = source.split("class RobotRootRandomizationCfg", 1)[1].split(
            "@dataclass", 1
        )[0]
        self.assertIn("enabled: bool = False", robot_block)
        self.assertIn("yaw_max_rad: float = math.radians(1.0)", robot_block)

    def test_all_generation_axes_are_compiled_by_one_builder(self):
        builder = (RANDOMIZATION / "event_cfg.py").read_text()
        mimic = (TASK / "mimic_cfg.py").read_text()
        for config_name in (
            "target_pose",
            "robot_root",
            "presence",
            "lighting",
            "shelf",
            "wall",
            "camera",
        ):
            self.assertIn(f"profile.{config_name}", builder)
        self.assertNotIn("_configure_randomization_events", mimic)
        self.assertNotIn(".params.update(", mimic)

    def test_target_pose_and_shared_pose_events_remain_continuous(self):
        target_event = (RANDOMIZATION / "task_pose_events.py").read_text()
        shared_event = (RANDOMIZATION / "events.py").read_text()
        self.assertIn("torch.rand((len(ids), 2)", target_event)
        self.assertIn("torch.rand((len(env_ids), 3)", shared_event)
        self.assertNotIn("pose_bank", target_event)
        self.assertNotIn("pose_bank", shared_event)

    def test_production_task_files_have_no_experiment_ids_or_variants(self):
        paths = (
            TASK / "mimic_env.py",
            TASK / "mimic_cfg.py",
            TASK / "takeout_terms.py",
            RANDOMIZATION / "cfg.py",
            RANDOMIZATION / "task_pose_events.py",
            RANDOMIZATION / "appearance_events.py",
        )
        for path in paths:
            self.assertIsNone(re.search(r"\bE\d{2}\b", path.read_text()), path)
        self.assertFalse(list(TASK.glob("*registration.py")))
        self.assertFalse(list(TASK.glob("*cascade*.py")))
        self.assertFalse(list(TASK.glob("*pose_bank*.py")))
        self.assertFalse(list(TASK.glob("mimic_env_v*.py")))

    def test_generated_hdf_has_no_post_hoc_filter(self):
        common = (ROOT / "tasks" / "common.py").read_text()
        generator = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "mimic"
            / "generate_dataset.py"
        ).read_text()
        self.assertIn("ActionStateRecorderManagerCfg", common)
        self.assertNotIn("terminal_automatic_metric", generator)
        self.assertNotIn("stamp_generation_hdf_metadata", generator)
        self.assertFalse((TASK / "recorder_terms.py").exists())

    def test_hdf_identifier_and_lerobot_guard_follow_new_ids(self):
        helper = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "data_converter"
            / "hdf5_task_metadata.py"
        ).read_text()
        converter = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "data_converter"
            / "isaaclab2lerobot.py"
        ).read_text()
        self.assertIn("def is_task_hdf(", helper)
        self.assertIn('is_task_hdf(source_hdf["data"], 458)', converter)
        self.assertIn("raw Mimic actions are hybrid EEF/joint commands", converter)
        self.assertIn("joint_pos_target[1:]", converter)

    def test_mimic_splice_defaults_are_conservative(self):
        source = (TASK / "mimic_cfg.py").read_text()
        compact = "".join(source.split())
        self.assertIn("generation_interpolate_from_last_target_pose=False", compact)
        self.assertEqual(compact.count("subtask_term_offset_range=(0,0)"), 3)
        self.assertEqual(compact.count("num_interpolation_steps=10"), 3)
        self.assertEqual(compact.count("action_noise=0.0"), 3)

    def test_generation_reset_event_order_is_explicit(self):
        tree = ast.parse((TASK / "mimic_cfg.py").read_text())
        event_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "Task458GenerationEventsCfg"
        )
        names = [
            target.id
            for node in event_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        expected = [
            "refresh_shelf_support",
            "randomize_target_pose",
            "randomize_robot_root",
            "randomize_non_target_presence",
            "randomize_lighting",
            "randomize_shelf_appearance",
            "randomize_wall_color",
            "randomize_cameras",
        ]
        self.assertEqual([name for name in names if name in expected], expected)

    def test_inactive_joint_hold_does_not_overlap_right_mimic_slice(self):
        source = (TASK / "mimic_env.py").read_text()
        self.assertIn(
            "INACTIVE_ACTION_INDICES = (*range(0, 8), *range(16, 19))", source
        )
        self.assertIn(
            "RIGHT_MIMIC_ACTION_INDICES = tuple(range(8, 16))", source
        )
        self.assertIn("self._capture_inactive_joint_holds(ids)", source)

    def test_mimic_reset_sanitizes_position_replay_state(self):
        source = (TASK / "mimic_env.py").read_text()
        self.assertIn(
            "from ...platform.replay_state import prepare_sg2_position_replay_state",
            source,
        )
        self.assertIn("prepare_sg2_position_replay_state(state)", source)

    def test_wall_randomization_uses_one_sample_for_two_task_walls(self):
        source = (RANDOMIZATION / "appearance_events.py").read_text()
        self.assertIn("ShowroomShell/BackWall", source)
        self.assertIn("ShowroomShell/LeftWall", source)
        self.assertIn("for wall_suffix in WALL_PRIM_SUFFIXES", source)
        self.assertEqual(
            source.count("colors = lower + torch.rand((len(ids), 3)"), 1
        )


if __name__ == "__main__":
    unittest.main()
