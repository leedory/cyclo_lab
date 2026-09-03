"""Focused integration checks for the closed Task000525 physical HDF audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import h5py


REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO / "scripts/sim2real/imitation_learning/tasks/task_000525"
SCRIPT = SCRIPT_DIR / "audit_physical_pick_hdf5.py"
SMOKE = REPO / "datasets/tmp/task_000525_orange_abcd_pick_no_nav_smoke_4.hdf5"
REGRESSION_B = REPO / "datasets/tmp/task_000525_orange_B_minimal_lid_regression_4.hdf5"


def load_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("task525_physical_pick_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Task000525PhysicalPickAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_closed_abcd_pick_smoke_passes_exact_contract(self):
        with h5py.File(SMOKE, "r") as handle:
            report = self.module.audit_file(
                handle["data"], source_path=SMOKE, expected_total=4, expected_per_region=1
            )
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(report["observed_region_counts"], {"A": 1, "B": 1, "C": 1, "D": 1})
        self.assertEqual(report["source_contract"]["fps"], 15)
        self.assertEqual(
            report["camera_contract_h_w"],
            {"cam_head": [376, 672], "cam_wrist_left": [640, 480], "cam_wrist_right": [640, 480]},
        )
        for region in self.module.REGIONS:
            distribution = report["accepted_pose_distributions_by_region"][region]
            self.assertEqual(distribution["accepted_count"], 1)
            self.assertIn("target_root_dx_m", distribution["metrics"])
            self.assertIn("root_yaw_delta_rad", distribution["metrics"])

    def test_closed_regression_rejects_unsuccessful_missing_sentinel_attempt(self):
        with h5py.File(REGRESSION_B, "r") as handle:
            accepted = self.module.audit_demo("demo_0", handle["data/demo_0"])
            rejected = self.module.audit_demo("demo_1", handle["data/demo_1"])
        self.assertTrue(accepted["accepted"], accepted["reasons"])
        self.assertFalse(rejected["accepted"])
        self.assertIn("success attribute is not true", rejected["reasons"])
        self.assertTrue(any("sentinel" in reason for reason in rejected["reasons"]))

    def test_full_closed_regression_fails_closed(self):
        with h5py.File(REGRESSION_B, "r") as handle:
            report = self.module.audit_file(
                handle["data"], source_path=REGRESSION_B, expected_total=4, expected_per_region=1
            )
        self.assertFalse(report["passed"])
        self.assertGreater(report["rejected_demo_count"], 0)
        self.assertTrue(any("region counts" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
