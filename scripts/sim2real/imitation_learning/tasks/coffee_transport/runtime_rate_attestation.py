"""Fail-closed validation for reviewed Coffee runtime-rate attestations."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ATTESTATION_SCHEMA = "cyclo.coffee_transport.runtime_rate_attestation.v1"
ATTESTATION_ID = "coffee_transport_final450_v12_runtime_rates_20260908"
RATE_CONFIG_COMMIT = "f106306bb450f6fb38cb3abf1f35f528b9c1b994"
EXPECTED_RATE_CONTRACT = {
    "physics_hz": 120.0,
    "control_hz": 15.0,
    "camera_hz": 15.0,
    "decimation": 8,
    "render_interval": 8,
    "swerve_controller_hz": 30.0,
    "swerve_update_interval_physics_steps": 4,
}


class RuntimeRateAttestationError(RuntimeError):
    """Raised when an input is not covered by the reviewed rate evidence."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RuntimeRateAttestationError(f"{label} must be an explicit UTC timestamp")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise RuntimeRateAttestationError(f"invalid {label}: {value!r}") from error
    return value


def _validate_evidence(entry: Mapping[str, Any]) -> None:
    evidence = entry.get("actual_process_start_evidence")
    if not isinstance(evidence, Mapping):
        raise RuntimeRateAttestationError("input lacks actual_process_start_evidence")
    _require_utc_timestamp(evidence.get("process_started_utc"), "process_started_utc")
    required_text = (
        "generation_log_path",
        "generation_log_sha256",
        "isaac_kit_log_name",
        "environment_cfg",
        "initial_seed",
    )
    for key in required_text:
        value = evidence.get(key)
        if not isinstance(value, (str, int)) or str(value).strip() == "":
            raise RuntimeRateAttestationError(f"process evidence lacks {key}")
    log_hash = str(evidence["generation_log_sha256"])
    if len(log_hash) != 64 or any(char not in "0123456789abcdef" for char in log_hash):
        raise RuntimeRateAttestationError("generation_log_sha256 is not lowercase SHA256")
    logged = evidence.get("logged_environment_steps_s")
    expected = {
        "physics": 1.0 / EXPECTED_RATE_CONTRACT["physics_hz"],
        "rendering": 1.0 / EXPECTED_RATE_CONTRACT["camera_hz"],
        "environment": 1.0 / EXPECTED_RATE_CONTRACT["control_hz"],
    }
    if not isinstance(logged, Mapping) or any(
        abs(float(logged.get(key, -1.0)) - value) > 1.0e-12
        for key, value in expected.items()
    ):
        raise RuntimeRateAttestationError(
            f"logged environment step evidence does not match {expected}"
        )


def load_runtime_rate_attestation(
    manifest_path: Path,
    input_paths: Sequence[Path],
    task_id: str,
) -> dict[str, Any]:
    """Validate one reviewed manifest and exact raw HDF hashes for a task."""

    path = manifest_path.resolve()
    if not path.is_file():
        raise RuntimeRateAttestationError(f"missing runtime-rate attestation: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_top = {
        "schema": ATTESTATION_SCHEMA,
        "attestation_id": ATTESTATION_ID,
        "review_status": "reviewed",
        "rate_config_commit": RATE_CONFIG_COMMIT,
        "rate_contract": EXPECTED_RATE_CONTRACT,
    }
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in expected_top.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeRateAttestationError(
            f"runtime-rate attestation contract mismatch: {mismatches}"
        )
    _require_utc_timestamp(payload.get("reviewed_utc"), "reviewed_utc")
    entries = payload.get("inputs")
    if not isinstance(entries, list) or not entries:
        raise RuntimeRateAttestationError("runtime-rate attestation has no inputs")
    task_entries = [entry for entry in entries if str(entry.get("task_id")) == task_id]
    if len(task_entries) != len(input_paths):
        raise RuntimeRateAttestationError(
            f"task {task_id} attests {len(task_entries)} inputs, received {len(input_paths)}"
        )
    by_hash: dict[str, Mapping[str, Any]] = {}
    for entry in task_entries:
        digest = str(entry.get("hdf5_sha256", ""))
        if len(digest) != 64 or digest in by_hash:
            raise RuntimeRateAttestationError("invalid or duplicate attested HDF SHA256")
        if not isinstance(entry.get("hdf5_size_bytes"), int) or entry["hdf5_size_bytes"] <= 0:
            raise RuntimeRateAttestationError("invalid attested HDF size")
        _validate_evidence(entry)
        by_hash[digest] = entry

    matched = []
    for input_path in input_paths:
        resolved = input_path.resolve()
        actual_size = resolved.stat().st_size
        candidates = [
            entry for entry in task_entries
            if int(entry["hdf5_size_bytes"]) == actual_size
            and str(entry.get("hdf5_basename", "")) == resolved.name
        ]
        if len(candidates) != 1:
            raise RuntimeRateAttestationError(
                f"{resolved} is not uniquely attested by basename and size"
            )
        actual_hash = sha256(resolved)
        entry = candidates[0]
        if actual_hash != entry["hdf5_sha256"]:
            raise RuntimeRateAttestationError(
                f"raw HDF SHA256 mismatch for {resolved}: {actual_hash}"
            )
        matched.append(
            {
                "task_id": task_id,
                "source_file": str(resolved),
                "hdf5_basename": resolved.name,
                "hdf5_size_bytes": actual_size,
                "hdf5_sha256": actual_hash,
                "actual_process_start_evidence": dict(
                    entry["actual_process_start_evidence"]
                ),
            }
        )
    if len({item["hdf5_sha256"] for item in matched}) != len(matched):
        raise RuntimeRateAttestationError("the same raw HDF was supplied more than once")
    return {
        "schema": ATTESTATION_SCHEMA,
        "attestation_id": ATTESTATION_ID,
        "review_status": "reviewed",
        "reviewed_utc": payload["reviewed_utc"],
        "manifest_path": str(path),
        "manifest_sha256": sha256(path),
        "rate_config_commit": RATE_CONFIG_COMMIT,
        "rate_contract": dict(EXPECTED_RATE_CONTRACT),
        "inputs": matched,
    }


def hdf5_rate_attributes(attestation: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten reviewed provenance into stable HDF5 scalar attributes."""

    rate = attestation["rate_contract"]
    return {
        "runtime_rate_attestation_schema": attestation["schema"],
        "runtime_rate_attestation_id": attestation["attestation_id"],
        "runtime_rate_attestation_review_status": attestation["review_status"],
        "runtime_rate_attestation_manifest_path": attestation["manifest_path"],
        "runtime_rate_attestation_manifest_sha256": attestation["manifest_sha256"],
        "rate_config_commit": attestation["rate_config_commit"],
        **rate,
        "runtime_rate_attestation_inputs_json": json.dumps(
            attestation["inputs"], sort_keys=True, ensure_ascii=True
        ),
    }


def validate_physical_runtime_rate_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a structured staging/dataset copy of the reviewed contract."""

    expected = {
        "schema": ATTESTATION_SCHEMA,
        "attestation_id": ATTESTATION_ID,
        "review_status": "reviewed",
        "rate_config_commit": RATE_CONFIG_COMMIT,
        "rate_contract": EXPECTED_RATE_CONTRACT,
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise RuntimeRateAttestationError(
            f"physical runtime-rate provenance mismatch: {mismatches}"
        )
    manifest_hash = provenance.get("manifest_sha256")
    inputs = provenance.get("inputs")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise RuntimeRateAttestationError("physical rate provenance lacks manifest SHA256")
    if not isinstance(inputs, list) or not inputs:
        raise RuntimeRateAttestationError("physical rate provenance has no raw inputs")
    for entry in inputs:
        if not isinstance(entry, Mapping):
            raise RuntimeRateAttestationError("physical rate input record is not an object")
        digest = entry.get("hdf5_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeRateAttestationError("physical rate input lacks HDF SHA256")
        _validate_evidence(entry)
    return dict(provenance)


def physical_runtime_rate_provenance_from_attrs(
    attrs: Mapping[str, Any],
) -> dict[str, Any]:
    """Read and validate the flattened contract before policy staging."""

    def scalar(key: str) -> Any:
        value = attrs.get(key)
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return value

    expected = {
        "runtime_rate_attestation_schema": ATTESTATION_SCHEMA,
        "runtime_rate_attestation_id": ATTESTATION_ID,
        "runtime_rate_attestation_review_status": "reviewed",
        "rate_config_commit": RATE_CONFIG_COMMIT,
        **EXPECTED_RATE_CONTRACT,
    }
    mismatches = {key: (scalar(key), value) for key, value in expected.items() if scalar(key) != value}
    if mismatches:
        raise RuntimeRateAttestationError(f"curated HDF rate provenance mismatch: {mismatches}")
    manifest_hash = str(scalar("runtime_rate_attestation_manifest_sha256") or "")
    if len(manifest_hash) != 64:
        raise RuntimeRateAttestationError("curated HDF lacks attestation manifest SHA256")
    try:
        inputs = json.loads(str(scalar("runtime_rate_attestation_inputs_json")))
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeRateAttestationError("curated HDF has invalid attested input JSON") from error
    if not isinstance(inputs, list) or not inputs:
        raise RuntimeRateAttestationError("curated HDF has no attested raw inputs")
    return validate_physical_runtime_rate_provenance(
        {
            "schema": ATTESTATION_SCHEMA,
            "attestation_id": ATTESTATION_ID,
            "review_status": "reviewed",
            "manifest_path": str(
                scalar("runtime_rate_attestation_manifest_path") or ""
            ),
            "manifest_sha256": manifest_hash,
            "rate_config_commit": RATE_CONFIG_COMMIT,
            "rate_contract": dict(EXPECTED_RATE_CONTRACT),
            "inputs": inputs,
        }
    )
