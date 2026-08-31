"""Pure filesystem tests for replay-staging merging."""

from __future__ import annotations

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


def load_module():
    spec = spec_from_file_location("merge_replay_staging", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReplayStagingMergeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def make_staging(self, root: Path, name: str, source_episode: str) -> Path:
        staging = root / name
        array = staging / "policy_arrays" / "episode_000000.npz"
        video = staging / "videos" / "episode_000000" / "cam.mp4"
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
            "videos": {"cam": str(video.relative_to(staging))},
            "state_names": ["joint"],
            "action_names": ["joint"],
        }
        manifest = {
            "schema": self.module.SCHEMA,
            "action_semantics": self.module.ACTION_SEMANTICS,
            "fps": 30,
            "camera_map": {"cam": "cam"},
            "task_instruction": "test",
            "source_hdf": name + ".hdf5",
            "source_hdf_sha256": name,
            "randomization_profile": name,
            "episode_count": 1,
            "total_frames": 3,
            "episodes": [record],
        }
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


if __name__ == "__main__":
    unittest.main()
