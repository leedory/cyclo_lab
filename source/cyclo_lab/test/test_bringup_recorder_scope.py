"""Keep runtime recorder suppression scoped away from demo collection."""

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BRINGUP_PATH = REPOSITORY_ROOT / "scripts" / "sim2real" / "bringup.py"
RECORD_DEMOS_PATH = (
    REPOSITORY_ROOT
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "recorder"
    / "record_demos.py"
)


class BringupRecorderScopeTest(unittest.TestCase):
    def test_generic_bringup_uses_an_empty_recorder(self):
        source = BRINGUP_PATH.read_text(encoding="utf-8")
        self.assertIn("from isaaclab.managers import RecorderManagerBaseCfg", source)
        self.assertIn("env_cfg.recorders = RecorderManagerBaseCfg()", source)

    def test_demo_entry_point_still_installs_streaming_recorder(self):
        source = RECORD_DEMOS_PATH.read_text(encoding="utf-8")
        self.assertIn("env_cfg.recorders.dataset_export_dir_path = output_dir", source)
        self.assertIn(
            "env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)",
            source,
        )
        self.assertNotIn("RecorderManagerBaseCfg", source)


if __name__ == "__main__":
    unittest.main()
