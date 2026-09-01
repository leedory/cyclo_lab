"""Static contracts for the organized Task000525 implementation."""

from pathlib import Path
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = (
    SOURCE_ROOT
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
)
TASK = ROOT / "tasks" / "task_000525"
REPO = SOURCE_ROOT.parents[1]
SCRIPTS = REPO / "scripts" / "sim2real" / "imitation_learning"


class Task000525ScaffoldTest(unittest.TestCase):
    def test_reference_files_are_explicit(self):
        for name in (
            "README.md",
            "spec.py",
            "layout.py",
            "destination_mat.py",
            "reset_events.py",
            "appearance_events.py",
            "profiles.py",
            "env_cfg.py",
            "generation_contract.py",
            "locomanipulation_sdg_contract.py",
            "locomanipulation_sdg_env.py",
            "online_dijkstra.py",
            "robot_stability.py",
        ):
            self.assertTrue((TASK / name).is_file(), name)

    def test_task_profile_owns_all_task_specific_randomization_axes(self):
        profiles = (TASK / "profiles.py").read_text(encoding="utf-8")
        env = (TASK / "env_cfg.py").read_text(encoding="utf-8")
        self.assertIn("class Task000525RandomizationCfg", profiles)
        self.assertIn("coffee_positions: CoffeeRegionRandomizationCfg", profiles)
        self.assertIn("coffee_visual_yaw: CoffeeVisualYawRandomizationCfg", profiles)
        self.assertIn("TASK000525_PHYSICAL_TRAJECTORY_GENERATION", profiles)
        self.assertIn("TASK000525_VISUAL_REPLAY_AUGMENTATION", profiles)
        self.assertNotIn("randomize_coffee_positions: bool", env)
        self.assertNotIn("randomize_coffee_visual_yaw: bool", env)
        self.assertIn("self.randomization.coffee_positions", env)
        self.assertIn("self.randomization.coffee_visual_yaw", env)

    def test_runtime_frame_contract_matches_local_generator(self):
        contract = (TASK / "generation_contract.py").read_text(encoding="utf-8")
        for frame in (
            "INITIAL_TARGET_OBJECT",
            "INITIAL_OBJECT_TO_CURRENT_ROOT_BLEND",
            "CURRENT_ROBOT_ROOT",
            "RECORDED_BASE_TO_CURRENT_ROOT",
        ):
            self.assertIn(frame, contract)
        self.assertNotIn("MimicSubtaskDraft", contract)

    def test_generation_is_outcome_named_and_task_local(self):
        registration = (ROOT / "__init__.py").read_text(encoding="utf-8")
        launcher = (SCRIPTS / "run_task000525_trajectory_generation.sh").read_text(
            encoding="utf-8"
        )
        local_generator = (
            SCRIPTS / "tasks" / "task_000525" / "generate_trajectories.py"
        )
        self.assertTrue(local_generator.is_file())
        self.assertIn(
            "Cyclo-Real-Showroom-Task000525-Trajectory-Generation-FFW-SG2-v0",
            registration,
        )
        self.assertIn("Task000525TrajectoryGenerationEnvCfg", registration)
        self.assertIn("generate_trajectories.py", launcher)
        self.assertNotIn("third_party/IsaacLab", launcher)

    def test_visual_profile_is_strictly_non_physical(self):
        profiles = (TASK / "profiles.py").read_text(encoding="utf-8")
        visual = profiles.split(
            "TASK000525_VISUAL_REPLAY_AUGMENTATION =", 1
        )[1]
        self.assertIn("coffee_visual_yaw=CoffeeVisualYawRandomizationCfg(enabled=True)", visual)
        self.assertNotIn("RobotRootRandomizationCfg", visual)
        self.assertNotIn("CoffeeRegionRandomizationCfg(enabled=True)", visual)

    def test_retired_mimic_aliases_are_removed(self):
        registration = (ROOT / "__init__.py").read_text(encoding="utf-8")
        profiles = (TASK / "profiles.py").read_text(encoding="utf-8")
        adapter = (TASK / "locomanipulation_sdg_env.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(
            "Cyclo-Real-Showroom-Task000525-Mimic-Generate-RootRandomized-FFW-SG2-v0",
            registration,
        )
        self.assertNotIn(
            "Task000525RootRandomizedLocomanipulationSDGEnvCfg", adapter
        )
        self.assertNotIn("TASK000525_TRAJECTORY_MIMIC_GENERATION", profiles)
        self.assertFalse(
            (SCRIPTS / "run_task000525_root_randomized_mimicgen.sh").exists()
        )
        self.assertFalse((TASK / "mimic_cfg.py").exists())


if __name__ == "__main__":
    unittest.main()
