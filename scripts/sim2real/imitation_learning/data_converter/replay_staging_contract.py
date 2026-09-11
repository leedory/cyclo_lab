"""Shared contracts for replay-staging merge and LeRobot conversion."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


STAGING_SCHEMA = "cyclo.isaac_action_replay_staging.v1"
COFFEE_ACTION_SEMANTICS = (
    "pre_step_named_joint_position_plus_optional_body_velocity"
)
RATE_ATTESTATION_SCHEMA = "cyclo.coffee_transport.runtime_rate_attestation.v1"
RATE_CONTRACT_KEYS = (
    "physics_hz",
    "control_hz",
    "camera_hz",
    "decimation",
    "render_interval",
    "swerve_controller_hz",
    "swerve_update_interval_physics_steps",
)


class ReplayStagingContractError(ValueError):
    """Raised when replay-staging provenance is incomplete or inconsistent."""


def validate_runtime_rate_provenance(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Validate reviewed Coffee rate provenance without pinning one campaign.

    Exact dataset IDs and source hashes belong to the campaign audit. This
    shared boundary verifies the stable schema and timing relationships only.
    """

    if manifest.get("action_semantics") != COFFEE_ACTION_SEMANTICS:
        return None
    rate = manifest.get("physical_runtime_rate_provenance")
    if not isinstance(rate, Mapping):
        raise ReplayStagingContractError(
            "missing physical runtime-rate provenance"
        )
    expected_metadata = {
        "schema": RATE_ATTESTATION_SCHEMA,
        "review_status": "reviewed",
    }
    mismatches = {
        key: (rate.get(key), expected)
        for key, expected in expected_metadata.items()
        if rate.get(key) != expected
    }
    if mismatches:
        raise ReplayStagingContractError(
            f"invalid physical runtime-rate metadata: {mismatches}"
        )
    if not isinstance(rate.get("attestation_id"), str) or not rate["attestation_id"]:
        raise ReplayStagingContractError("missing runtime-rate attestation_id")
    commit = rate.get("rate_config_commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ReplayStagingContractError(
            "rate_config_commit must be a full lowercase Git SHA"
        )
    inputs = rate.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ReplayStagingContractError("runtime-rate inputs must be non-empty")

    contract = rate.get("rate_contract")
    if not isinstance(contract, Mapping) or set(contract) != set(RATE_CONTRACT_KEYS):
        raise ReplayStagingContractError(
            f"rate_contract must contain exactly {RATE_CONTRACT_KEYS}"
        )
    try:
        physics_hz = float(contract["physics_hz"])
        control_hz = float(contract["control_hz"])
        camera_hz = float(contract["camera_hz"])
        swerve_hz = float(contract["swerve_controller_hz"])
        decimation = int(contract["decimation"])
        render_interval = int(contract["render_interval"])
        swerve_interval = int(contract["swerve_update_interval_physics_steps"])
    except (TypeError, ValueError) as error:
        raise ReplayStagingContractError(
            "rate_contract values must be numeric"
        ) from error
    if min(physics_hz, control_hz, camera_hz, swerve_hz) <= 0.0:
        raise ReplayStagingContractError("runtime rates must be positive")
    if min(decimation, render_interval, swerve_interval) <= 0:
        raise ReplayStagingContractError("runtime intervals must be positive integers")
    relationships = {
        "physics_hz / decimation": (physics_hz / decimation, control_hz),
        "physics_hz / render_interval": (physics_hz / render_interval, camera_hz),
        "physics_hz / swerve_interval": (physics_hz / swerve_interval, swerve_hz),
        "manifest fps": (float(manifest.get("fps", 0.0)), control_hz),
    }
    inconsistent = {
        label: values
        for label, values in relationships.items()
        if not math.isclose(*values, rel_tol=0.0, abs_tol=1.0e-6)
    }
    if inconsistent:
        raise ReplayStagingContractError(
            f"inconsistent runtime-rate relationships: {inconsistent}"
        )
    return rate
