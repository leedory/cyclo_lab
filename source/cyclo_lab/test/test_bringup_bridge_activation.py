"""Regression checks for optional bridge activation hooks in generic bringup."""

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BRINGUP_PATH = REPOSITORY_ROOT / "scripts" / "sim2real" / "bringup.py"


class TestBringupBridgeActivation(unittest.TestCase):
    def test_activation_hook_is_optional_for_non_sg2_bridges(self):
        source = BRINGUP_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'control_activation = getattr(bridge, "begin_control_activation", None)',
            source,
        )
        self.assertIn("if callable(control_activation):", source)
        self.assertIn("control_activation()", source)
        self.assertNotIn("bridge.begin_control_activation()", source)

    def test_sg2_bridge_uses_task_owned_trajectory_groups(self):
        source = BRINGUP_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'getattr(\n                env_cfg, "sim2real_active_trajectory_groups", None',
            source,
        )

    def test_ui_session_requires_sg2_and_forces_cameras(self):
        source = BRINGUP_PATH.read_text(encoding="utf-8")

        self.assertIn('"--ui-session"', source)
        self.assertIn('"--session-status-file"', source)
        self.assertIn(
            'DEFAULT_UI_SESSION_STATUS_FILE = "/tmp/cyclo_lab_ui_session.json"',
            source,
        )
        self.assertIn('if args_cli.bridge != "ffw_sg2":', source)
        self.assertIn('parser.error("--ui-session requires --bridge ffw_sg2.")', source)
        self.assertIn("args_cli.enable_cameras = True", source)

    def test_ui_session_attaches_every_bridge_camera_before_validation(self):
        source = BRINGUP_PATH.read_text(encoding="utf-8")

        attach_index = source.index(
            'enable_ui_session_camera = getattr(env_cfg, "enable_ui_session_camera", None)'
        )
        select_index = source.index(
            "camera_names = _set_camera_sensors_enabled(env_cfg, args_cli.enable_cameras)"
        )
        validate_index = source.index(
            "set(FFW_SG2_CAMERA_TOPICS).difference(camera_names)"
        )
        self.assertLess(attach_index, select_index)
        self.assertLess(select_index, validate_index)
        self.assertIn(
            'getattr(env_cfg, "enable_ui_session_camera", None)', source
        )
        self.assertIn('elif args_cli.camera_view == "operator":', source)
        self.assertIn('if args_cli.ui_session or args_cli.camera_view != "operator":', source)

    def test_ui_session_wording_is_simulation_specific(self):
        source = BRINGUP_PATH.read_text(encoding="utf-8")

        self.assertIn("UI-launched simulation session; policy actions are active", source)
        self.assertNotIn("UI session robot actions", source)

    def test_ui_session_avoids_keyboard_and_auto_activates_after_reset(self):
        source = BRINGUP_PATH.read_text(encoding="utf-8")

        listener_guard_index = source.index("if not args_cli.ui_session:")
        listener_import_index = source.index("from pynput.keyboard import Listener")
        self.assertLess(listener_guard_index, listener_import_index)
        self.assertGreaterEqual(source.count("_begin_bridge_control_activation(bridge)"), 3)
        self.assertIn("session.begin_reset(reset_source)", source)
        self.assertIn("session.finish_reset()", source)

    def test_ui_session_reports_sequences_and_forces_post_reset_heartbeat(self):
        source = BRINGUP_PATH.read_text(encoding="utf-8")

        self.assertIn("observation_sequence += 1", source)
        self.assertIn("if camera_batch_published:", source)
        self.assertIn("camera_sequence += 1", source)
        self.assertIn("force_heartbeat_after_reset = True", source)
        self.assertIn(
            "force_heartbeat_after_reset and camera_batch_published", source
        )
        self.assertIn("force=force_session_heartbeat", source)
        self.assertIn("session.stopping()", source)
        self.assertIn("session.stopped()", source)
        self.assertIn("session.fail(exc)", source)


if __name__ == "__main__":
    unittest.main()
