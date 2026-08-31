"""Static contracts for the Task000525 fixed ivory destination mat."""

from pathlib import Path
import unittest


SOURCE = Path(__file__).resolve().parents[1]
PACKAGE = SOURCE / "cyclo_lab"
TASK = (
    PACKAGE
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "task_000525"
)
ASSET_DATA = SOURCE / "data" / "object"


class Task000525DestinationMatTest(unittest.TestCase):
    def test_mat_asset_exists(self):
        self.assertTrue((ASSET_DATA / "ivory_table_mat.usd").is_file())

    def test_reviewed_pose_and_clearances_are_explicit(self):
        source = (TASK / "destination_mat.py").read_text(encoding="utf-8")
        for value in (
            "-0.860735354423523",
            "-0.6178486049175262",
            "0.7511681447029115",
            "0.25802019238471985",
            "0.085",
        ):
            self.assertIn(value, source)
        self.assertIn("TASK000525_DESTINATION_MAT_DIMENSIONS_M", source)

    def test_joint_targets_table_and_mat_rigid_bodies(self):
        source = (TASK / "destination_mat.py").read_text(encoding="utf-8")
        self.assertIn("UsdPhysics.FixedJoint.Define", source)
        self.assertIn("joint.CreateBody0Rel().SetTargets([table_path])", source)
        self.assertIn("joint.CreateBody1Rel().SetTargets([mat_path])", source)
        self.assertIn("joint.CreateJointEnabledAttr().Set(True)", source)
        self.assertIn("joint.CreateExcludeFromArticulationAttr().Set(True)", source)

    def test_table_proxy_alignment_is_not_hard_coded_at_runtime(self):
        source = (TASK / "destination_mat.py").read_text(encoding="utf-8")
        self.assertNotIn("_align_table_proxy_to_visual_top", source)
        self.assertNotIn("TASK000525_TABLE_PROXY_VISUAL_OFFSET_M", source)

    def test_environment_spawns_mat_and_authors_joint_before_physics(self):
        source = (TASK / "env_cfg.py").read_text(encoding="utf-8")
        self.assertIn(
            "self.scene.destination_mat = make_task000525_destination_mat_cfg()",
            source,
        )
        self.assertIn("func=attach_task000525_destination_mat_to_table", source)
        self.assertIn('mode="prestartup"', source)

    def test_instruction_names_ivory_mat(self):
        source = (TASK / "spec.py").read_text(encoding="utf-8")
        self.assertIn("can on the ivory mat on the table", source)


if __name__ == "__main__":
    unittest.main()
