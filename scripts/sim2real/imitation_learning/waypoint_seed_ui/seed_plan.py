"""Isaac-free plan validation for SG2 desired-joint-position waypoints."""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import runpy
import xml.etree.ElementTree as ET


LAB_ROOT = Path(os.environ.get("CYCLO_LAB_ROOT", Path(__file__).resolve().parents[4]))
JOINTS_FILE = LAB_ROOT / "source/cyclo_lab/cyclo_lab/robot_specs/ffw/sg2/joints.py"
SG2_URDF_FILE = Path(os.environ.get(
    "SG2_URDF_PATH",
    str(LAB_ROOT.parent / "cyclo_intelligence_s2r/shared/shared/robot_configs/urdf/ffw_sg2_follower.urdf"),
))
REGISTRATION_ROOTS = (
    LAB_ROOT / "source/cyclo_lab/cyclo_lab/manager_based/manipulation",
    LAB_ROOT / "scripts/sim2real/imitation_learning/tasks/local",
)
INITIAL_POSE_HOLD_S = 0.2
IK_SOLVER_LABELS = {
    "iterative_lm": "Iterative LM",
    "posture_lm": "Posture-biased LM",
    "constrained_qp": "Constrained QP",
    "local_dls": "Local DLS (legacy)",
}


def joint_contract() -> tuple[tuple[str, ...], dict[str, tuple[float, float]]]:
    """Read SG2 19D names and URDF position limits without importing Isaac."""
    definitions = runpy.run_path(str(JOINTS_FILE))
    names = tuple(definitions["FFW_SG2_ACTION_JOINT_NAMES"])
    contract_limits = dict(definitions["FFW_SG2_JOINT_POSITION_LIMITS"])
    if not SG2_URDF_FILE.is_file():
        # Isaac containers may mount cyclo_lab but not cyclo_intelligence_s2r.
        # This Cyclo Lab table is documented as copied from the same follower URDF.
        return names, contract_limits
    root = ET.parse(SG2_URDF_FILE).getroot()
    urdf_limits = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in names:
            continue
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"SG2 URDF has no position limits for {name}")
        lower, upper = float(limit.get("lower")), float(limit.get("upper"))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(f"SG2 URDF has invalid position limits for {name}")
        urdf_limits[name] = (lower, upper)
    if set(urdf_limits) != set(names):
        raise ValueError(f"SG2 URDF is missing 19D joints: {sorted(set(names) - set(urdf_limits))}")
    if urdf_limits != contract_limits:
        raise ValueError("SG2 URDF joint limits differ from Cyclo Lab's 19D action contract")
    return names, urdf_limits


def discover_environments() -> list[str]:
    """List literal SG2 Gym IDs declared in the existing source and local plugins."""
    names: set[str] = set()
    for root in REGISTRATION_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Attribute) and func.attr == "register"):
                    continue
                for keyword in node.keywords:
                    if keyword.arg != "id" or not isinstance(keyword.value, ast.Constant):
                        continue
                    name = keyword.value.value
                    if isinstance(name, str) and name.startswith("Cyclo-") and "FFW-SG2" in name:
                        names.add(name)
    return sorted(names)


def validate_plan(raw: object, environments: list[str] | None = None) -> dict:
    """Return a normalized plan starting at the selected env's measured reset pose."""
    if not isinstance(raw, dict):
        raise ValueError("Plan must be a JSON object.")
    environment = raw.get("environment")
    if not isinstance(environment, str) or not environment:
        raise ValueError("Select an environment.")
    if environments is not None and environment not in environments:
        raise ValueError(f"Environment is not in the discovered list: {environment}")
    if "Mimic-Generate" in environment or "Trajectory-Generation" in environment:
        raise ValueError("Select a nominal or Random task environment for waypoint recording.")
    rate = raw.get("control_hz", 15)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("Control rate must be a positive number.")
    if rate > 240:
        raise ValueError("Control rate must be at most 240 Hz.")
    joint_names, limits = joint_contract()

    object_position = raw.get("object_position_xyz_m")
    if object_position is not None:
        if not isinstance(object_position, list) or len(object_position) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or abs(value) > 10.0
            for value in object_position
        ):
            raise ValueError("Object position must contain three finite world XYZ values in metres.")
        object_position = [float(value) for value in object_position]

    def clean_q_des(values: object, label: str) -> dict[str, float]:
        if not isinstance(values, dict):
            raise ValueError(f"{label} q_des must be an object.")
        clean = {}
        for name, value in values.items():
            if name not in joint_names:
                raise ValueError(f"{label} has unknown joint: {name}")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{label} {name} needs a finite number.")
            lower, upper = limits[name]
            if not lower <= value <= upper:
                raise ValueError(f"{label} {name}={value} is outside [{lower}, {upper}].")
            clean[name] = float(value)
        return clean

    if "init_pose" in raw:
        raise ValueError("Old plan contains init_pose. Create a new plan; the env reset pose is now the start pose.")
    waypoints = raw.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("Add at least one waypoint.")
    if len(waypoints) > 100:
        raise ValueError("A plan can contain at most 100 waypoints.")
    normalized = []
    for index, waypoint in enumerate(waypoints, start=1):
        if not isinstance(waypoint, dict):
            raise ValueError(f"Waypoint {index} must be an object.")
        seconds = waypoint.get("duration_s")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"Waypoint {index} duration must be positive seconds.")
        memo = waypoint.get("memo", "")
        if not isinstance(memo, str):
            raise ValueError(f"Waypoint {index} memo must be text.")
        memo = memo.strip()
        if len(memo) > 120 or any(ord(char) < 32 for char in memo):
            raise ValueError(f"Waypoint {index} memo must be at most 120 characters on one line.")
        enabled = waypoint.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"Waypoint {index} enabled must be true or false.")
        interpolation = waypoint.get("interpolation", "linear")
        if interpolation not in ("linear", "smoothstep"):
            raise ValueError(f"Waypoint {index} interpolation must be linear or smoothstep.")
        steps = max(1, round(float(seconds) * float(rate))) if enabled else 0
        normalized.append({
            "duration_s": float(seconds),
            "steps": steps,
            "actual_duration_s": steps / float(rate),
            "memo": memo,
            "enabled": enabled,
            "interpolation": interpolation,
            "q_des": clean_q_des(waypoint.get("q_des", {}), f"Waypoint {index}"),
        })
    if not any(waypoint["enabled"] for waypoint in normalized):
        raise ValueError("Select at least one point for Preview Motion or Seed Export.")
    return {
        "schema_version": 3,
        "environment": environment,
        "control_hz": float(rate),
        "initial_pose_hold_s": INITIAL_POSE_HOLD_S,
        "object_position_xyz_m": object_position,
        "preview": raw.get("preview", "head") if raw.get("preview") in ("head", "video") else "head",
        "joint_order": list(joint_names),
        "waypoints": normalized,
    }
