"""CPU-only contract tests for the LeRobot episode action/state viewer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO
    / "scripts/sim2real/imitation_learning/tasks/coffee_transport/"
    "view_lerobot_episode.py"
)


def load_viewer():
    spec = importlib.util.spec_from_file_location("coffee_episode_viewer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CoffeeEpisodeViewerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.viewer = load_viewer()

    def test_signal_selection_defaults_to_named_union(self):
        selected = self.viewer.select_signals(
            [],
            ("arm_l_joint1", "lift_joint"),
            ("arm_l_joint1", "lift_joint"),
        )
        self.assertEqual(selected, ("arm_l_joint1", "lift_joint"))
        self.assertEqual(
            self.viewer.select_signals(
                ("lift_joint", "lift_joint"),
                ("arm_l_joint1", "lift_joint"),
                ("arm_l_joint1", "lift_joint"),
            ),
            ("lift_joint",),
        )

    def test_unknown_signal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown signals"):
            self.viewer.select_signals(
                ("not_a_signal",), ("lift_joint",), ("lift_joint",)
            )

    def test_video_path_uses_episode_chunk_file_and_timestamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = (
                "videos/observation.images.rgb.cam_left_head/"
                "chunk-002/file-003.mp4"
            )
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"placeholder")
            camera = "observation.images.rgb.cam_left_head"
            info = {
                "video_path": (
                    "videos/{video_key}/chunk-{chunk_index:03d}/"
                    "file-{file_index:03d}.mp4"
                ),
                "features": {camera: {"info": {"video.fps": 15.0}}},
            }
            metadata = {
                f"videos/{camera}/chunk_index": 2,
                f"videos/{camera}/file_index": 3,
                f"videos/{camera}/from_timestamp": 4.0,
            }
            resolved, frame, fps = self.viewer._resolve_video(
                root, info, metadata, camera
            )
            self.assertEqual(resolved, path)
            self.assertEqual(frame, 60)
            self.assertEqual(fps, 15.0)

    def test_info_requires_lerobot_v3(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "meta").mkdir()
            (root / "meta/info.json").write_text(
                json.dumps({"codebase_version": "v2.1"}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "v3.0"):
                self.viewer._read_info(root)


if __name__ == "__main__":
    unittest.main()
