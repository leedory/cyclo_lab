"""Tests for the lightweight UI-launched simulation status contract."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from cyclo_lab.runtime.simulation import SimulationSession


class TestSimulationSession(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.status_file = Path(self.temporary_directory.name) / "session.json"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def read_status(self) -> dict:
        return json.loads(self.status_file.read_text(encoding="utf-8"))

    def test_ready_and_reset_lifecycle(self):
        session = SimulationSession(
            task="Cyclo-Real-Showroom-Task000525-FFW-SG2-v0",
            bridge="ffw_sg2",
            status_file=self.status_file,
        )

        self.assertEqual(self.read_status()["state"], "starting")
        session.ready(control_hz=15.0, camera_hz=15.0)
        ready = self.read_status()
        self.assertEqual(ready["state"], "ready")
        self.assertEqual(ready["control_hz"], 15.0)
        self.assertEqual(ready["camera_hz"], 15.0)

        session.begin_reset("/simulation/reset")
        self.assertEqual(self.read_status()["state"], "resetting")
        session.finish_reset()
        session.heartbeat(4, observation_sequence=4, camera_sequence=3, force=True)
        reset = self.read_status()
        self.assertEqual(reset["state"], "ready")
        self.assertEqual(reset["reset_count"], 1)
        self.assertEqual(reset["observation_sequence"], 4)
        self.assertEqual(reset["camera_sequence"], 3)

    def test_error_is_not_overwritten_by_stop(self):
        session = SimulationSession(task="test-task", bridge="ffw_sg2", status_file=self.status_file)
        session.fail("boom")
        session.stopping()
        session.stopped()

        status = self.read_status()
        self.assertEqual(status["state"], "error")
        self.assertEqual(status["error"], "boom")


if __name__ == "__main__":
    unittest.main()
