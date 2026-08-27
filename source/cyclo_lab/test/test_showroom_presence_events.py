"""Offline contracts for showroom object-presence sampling."""

from __future__ import annotations

from pathlib import Path
import unittest

import torch

from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.presence_events import (
    SimulationVisibilityPresenceController,
    sample_present_masks,
)

EVENT_CFG = (
    Path(__file__).resolve().parents[1]
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "randomization"
    / "event_cfg.py"
)


class SamplePresentMasksTest(unittest.TestCase):
    def test_zero_probability_keeps_every_pair_present(self):
        masks = sample_present_masks(
            object_names=("a", "b"),
            env_count=16,
            disappearance_probability=0.0,
            device="cpu",
        )
        self.assertTrue(all(bool(mask.all()) for mask in masks.values()))

    def test_one_probability_makes_every_pair_absent(self):
        masks = sample_present_masks(
            object_names=("a", "b"),
            env_count=16,
            disappearance_probability=1.0,
            device="cpu",
        )
        self.assertTrue(all(not bool(mask.any()) for mask in masks.values()))

    def test_seed_is_reproducible(self):
        torch.manual_seed(458)
        first = sample_present_masks(
            object_names=("a", "b", "c"),
            env_count=16,
            disappearance_probability=0.5,
            device="cpu",
        )
        torch.manual_seed(458)
        second = sample_present_masks(
            object_names=("a", "b", "c"),
            env_count=16,
            disappearance_probability=0.5,
            device="cpu",
        )
        for name in first:
            self.assertTrue(torch.equal(first[name], second[name]))

    def test_duplicate_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            sample_present_masks(
                object_names=("a", "a"),
                env_count=1,
                disappearance_probability=0.5,
                device="cpu",
            )

    def test_invalid_probability_is_rejected(self):
        for value in (-0.01, 1.01):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                sample_present_masks(
                    object_names=("a",),
                    env_count=1,
                    disappearance_probability=value,
                    device="cpu",
                )


class PresenceIntegrationContractTest(unittest.TestCase):
    def test_event_builder_depends_on_presence_module_directly(self):
        source = EVENT_CFG.read_text(encoding="utf-8")
        self.assertIn("from . import presence_events", source)
        self.assertIn(
            "func=presence_events.randomize_non_target_presence,", source
        )
        self.assertNotIn(
            "appearance_events.randomize_non_target_presence", source
        )

    def test_partial_mask_state_keeps_legacy_writer_compatible(self):
        class FakeEnv:
            num_envs = 3
            device = "cpu"

        env = FakeEnv()
        controller = object.__new__(SimulationVisibilityPresenceController)
        controller._env = env
        controller._remember_masks(
            torch.tensor([1]), {"packet": torch.tensor([False])}
        )

        self.assertEqual(env._showroom_non_target_presence["packet"].tolist(), [True, False, True])
        self.assertIs(
            env._showroom_non_target_presence,
            env._task458_non_target_presence,
        )


if __name__ == "__main__":
    unittest.main()
