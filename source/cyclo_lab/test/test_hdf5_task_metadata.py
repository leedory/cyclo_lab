"""Tests for generic HDF5 task identity parsing."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "data_converter"
    / "hdf5_task_metadata.py"
)


def load_module():
    spec = spec_from_file_location("hdf5_task_metadata", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Group:
    def __init__(self, **attrs):
        self.attrs = attrs


class Hdf5TaskMetadataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_reads_direct_task_name(self):
        group = _Group(task_env_name=b"Cyclo-Task000525-v0")
        self.assertEqual(
            self.module.read_task_env_name(group),
            "Cyclo-Task000525-v0",
        )
        self.assertTrue(self.module.is_task_hdf(group, "task_000525"))
        self.assertFalse(self.module.is_task_hdf(group, 458))

    def test_reads_legacy_env_args(self):
        group = _Group(
            env_args=json.dumps(
                {"env_name": "Cyclo_Real_Showroom_Task000458_Mimic_Seed-v0"}
            )
        )
        self.assertTrue(self.module.is_task_hdf(group, "000458"))

    def test_rejects_partial_task_id_match(self):
        group = _Group(task_env_name="Cyclo-Task4581-v0")
        self.assertFalse(self.module.is_task_hdf(group, 458))

    def test_handles_missing_or_malformed_metadata(self):
        self.assertIsNone(self.module.read_task_env_name(_Group()))
        self.assertFalse(self.module.is_task_hdf(_Group(env_args="{"), 458))
        with self.assertRaises(ValueError):
            self.module.is_task_hdf(_Group(task_env_name="anything"), "peanut")


if __name__ == "__main__":
    unittest.main()
