"""Focused tests for external Coffee runtime-rate provenance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / (
    "scripts/sim2real/imitation_learning/tasks/coffee_transport/"
    "runtime_rate_attestation.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "coffee_runtime_rate_attestation", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CoffeeRuntimeRateAttestationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def manifest(self, raw: Path, *, physics_hz: float = 120.0) -> dict:
        evidence = {
            "process_started_utc": "2026-09-07T17:46:36Z",
            "generation_log_path": "/evidence/generation.log",
            "generation_log_sha256": "a" * 64,
            "isaac_kit_log_name": "kit_20260908_024636.log",
            "environment_cfg": "Task000001TrajectoryGenerationEnvCfg",
            "initial_seed": 101,
            "logged_environment_steps_s": {
                "physics": 1.0 / 120.0,
                "rendering": 1.0 / 15.0,
                "environment": 1.0 / 15.0,
            },
        }
        rates = dict(self.module.EXPECTED_RATE_CONTRACT)
        rates["physics_hz"] = physics_hz
        return {
            "schema": self.module.ATTESTATION_SCHEMA,
            "attestation_id": self.module.ATTESTATION_ID,
            "review_status": "reviewed",
            "reviewed_utc": "2026-09-08T09:30:00Z",
            "rate_config_commit": self.module.RATE_CONFIG_COMMIT,
            "rate_contract": rates,
            "inputs": [{
                "task_id": "000001",
                "hdf5_basename": raw.name,
                "hdf5_size_bytes": raw.stat().st_size,
                "hdf5_sha256": self.module.sha256(raw),
                "actual_process_start_evidence": evidence,
            }],
        }

    def write_manifest(self, root: Path, payload: dict) -> Path:
        path = root / "attestation.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_exact_reviewed_input_is_attested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "physical.hdf5"
            raw.write_bytes(b"reviewed-v12")
            manifest = self.write_manifest(root, self.manifest(raw))

            result = self.module.load_runtime_rate_attestation(
                manifest, [raw], "000001"
            )

            self.assertEqual(result["rate_contract"]["physics_hz"], 120.0)
            self.assertEqual(result["rate_contract"]["decimation"], 8)
            self.assertEqual(result["inputs"][0]["hdf5_sha256"], self.module.sha256(raw))

    def test_hash_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "physical.hdf5"
            raw.write_bytes(b"reviewed-v12")
            manifest = self.write_manifest(root, self.manifest(raw))
            raw.write_bytes(b"changed-input")

            with self.assertRaises(self.module.RuntimeRateAttestationError):
                self.module.load_runtime_rate_attestation(manifest, [raw], "000001")

    def test_old_30hz_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "task_000001_orange_transport_physical_success50_v3.hdf5"
            raw.write_bytes(b"old-30hz-v3")
            manifest = self.write_manifest(
                root, self.manifest(raw, physics_hz=30.0)
            )

            with self.assertRaisesRegex(
                self.module.RuntimeRateAttestationError, "contract mismatch"
            ):
                self.module.load_runtime_rate_attestation(manifest, [raw], "000001")

    def test_curated_attrs_require_all_seven_rate_values(self):
        provenance = {
            "schema": self.module.ATTESTATION_SCHEMA,
            "attestation_id": self.module.ATTESTATION_ID,
            "review_status": "reviewed",
            "manifest_path": "/reviewed/attestation.json",
            "manifest_sha256": "b" * 64,
            "rate_config_commit": self.module.RATE_CONFIG_COMMIT,
            "rate_contract": dict(self.module.EXPECTED_RATE_CONTRACT),
            "inputs": [{
                "hdf5_sha256": "c" * 64,
                "actual_process_start_evidence": {
                    "process_started_utc": "2026-09-07T17:46:36Z",
                    "generation_log_path": "/evidence/generation.log",
                    "generation_log_sha256": "d" * 64,
                    "isaac_kit_log_name": "kit.log",
                    "environment_cfg": "Task000001TrajectoryGenerationEnvCfg",
                    "initial_seed": 101,
                    "logged_environment_steps_s": {
                        "physics": 1.0 / 120.0,
                        "rendering": 1.0 / 15.0,
                        "environment": 1.0 / 15.0,
                    },
                },
            }],
        }
        attrs = self.module.hdf5_rate_attributes(provenance)
        self.module.physical_runtime_rate_provenance_from_attrs(attrs)
        del attrs["swerve_update_interval_physics_steps"]
        with self.assertRaises(self.module.RuntimeRateAttestationError):
            self.module.physical_runtime_rate_provenance_from_attrs(attrs)


if __name__ == "__main__":
    unittest.main()
