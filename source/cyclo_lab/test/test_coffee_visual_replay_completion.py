"""CPU-only regressions for Coffee visual replay completion and resume."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / (
    "scripts/sim2real/imitation_learning/tasks/coffee_transport/"
    "visual_replay_completion.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("visual_replay_completion", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExitCalled(Exception):
    def __init__(self, code: int):
        self.code = code


class TestCoffeeVisualReplayCompletion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def make_checkpoint(self, output: Path, count: int, batch_size: int = 5) -> dict:
        episodes = []
        batches = []
        for index in range(count):
            arrays = output / "policy_arrays" / f"episode_{index:06d}.npz"
            video = output / "videos" / f"episode_{index:06d}" / "cam_left_head.mp4"
            arrays.parent.mkdir(parents=True, exist_ok=True)
            video.parent.mkdir(parents=True, exist_ok=True)
            arrays.write_bytes(f"array-{index}".encode())
            video.write_bytes(f"video-{index}".encode())
            episodes.append(
                {
                    "episode_index": index,
                    "length": index + 1,
                    "arrays": str(arrays.relative_to(output)),
                    "array_sha256": self.module._sha256(arrays),
                    "videos": {"cam_left_head": str(video.relative_to(output))},
                }
            )
        for start in range(0, count, batch_size):
            batches.append({"output_indices": list(range(start, start + batch_size))})
        return {"episodes": episodes, "batches": batches}

    def test_incomplete_clean_exit_is_forced_nonzero(self):
        calls = []
        stderr = io.StringIO()

        def exit_fn(code: int):
            calls.append(code)
            raise ExitCalled(code)

        guard = self.module.VisualReplayCompletionGuard(
            exit_fn=exit_fn, stderr=stderr, register=False
        )
        guard.arm(Path("manifest.json"), 400)
        with self.assertRaises(ExitCalled):
            guard.fail_if_incomplete()
        self.assertEqual(calls, [self.module.INCOMPLETE_EXIT_CODE])
        self.assertIn("expected_outputs=400", stderr.getvalue())

    def test_completed_run_keeps_clean_exit(self):
        calls = []
        guard = self.module.VisualReplayCompletionGuard(
            exit_fn=calls.append, stderr=io.StringIO(), register=False
        )
        guard.arm(Path("manifest.json"), 400)
        guard.mark_complete()
        guard.fail_if_incomplete()
        self.assertEqual(calls, [])

    def test_105_episode_checkpoint_resumes_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.make_checkpoint(output, 105)
            completed = self.module.validate_resume_checkpoint(
                output, manifest, batch_size=5, expected_outputs=400
            )
        self.assertEqual(completed, 105)

    def test_resume_rejects_missing_or_changed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.make_checkpoint(output, 5)
            (output / manifest["episodes"][3]["arrays"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                self.module.validate_resume_checkpoint(
                    output, manifest, batch_size=5, expected_outputs=400
                )

    def test_resume_rejects_gap_or_final_fields_on_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.make_checkpoint(output, 5)
            manifest["episodes"][4]["episode_index"] = 7
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                self.module.validate_resume_checkpoint(
                    output, manifest, batch_size=5, expected_outputs=400
                )

            manifest = self.make_checkpoint(output, 5)
            manifest["episode_count"] = 5
            with self.assertRaisesRegex(ValueError, "final field"):
                self.module.validate_resume_checkpoint(
                    output, manifest, batch_size=5, expected_outputs=400
                )

    def test_finalize_requires_exact_ordered_count(self):
        incomplete = {
            "episodes": [{"episode_index": index, "length": 1} for index in range(105)]
        }
        with self.assertRaisesRegex(ValueError, "expected 400"):
            self.module.finalize_manifest(
                incomplete, expected_outputs=400, elapsed_s=1.0
            )

        complete = {
            "episodes": [{"episode_index": index, "length": 2} for index in range(400)]
        }
        self.module.finalize_manifest(
            complete, expected_outputs=400, elapsed_s=3.5
        )
        self.assertEqual(complete["episode_count"], 400)
        self.assertEqual(complete["total_frames"], 800)
        self.assertEqual(complete["elapsed_s"], 3.5)


if __name__ == "__main__":
    unittest.main()
