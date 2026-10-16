"""Observation configuration shared by seed and MimicGen recording."""

from __future__ import annotations

import math


def disable_recorded_camera_observations(env_cfg) -> tuple[str, ...]:
    """Remove image terms from the recorded policy group before env creation.

    The scene camera sensors remain available for an operator preview or a
    later replay render. Action/state and MimicGen subtask terms are retained.
    """

    policy = env_cfg.observations.policy
    removed = []
    for name, term in vars(policy).items():
        if term is None:
            continue
        params = getattr(term, "params", None)
        if not isinstance(params, dict) or "sensor_cfg" not in params:
            continue
        sensor_name = getattr(params["sensor_cfg"], "name", None)
        scene_sensor = getattr(env_cfg.scene, sensor_name, None) if sensor_name else None
        is_camera_sensor = getattr(scene_sensor, "data_types", None) is not None
        is_image_function = getattr(getattr(term, "func", None), "__name__", None) == "image"
        if not (is_camera_sensor or is_image_function):
            continue
        setattr(policy, name, None)
        removed.append(name)
    return tuple(removed)


def configure_recording_rates(
    env_cfg, *, physics_hz: float | None = None, camera_hz: float | None = None
) -> tuple[float, float, float]:
    """Override capture simulation rates while keeping the task control rate.

    The physics and render rates must both divide into integer physics steps.
    This is intended for measured capture trials; a different physics rate can
    change contact dynamics and must be validated before collecting final data.
    """

    control = float(env_cfg.control_hz)
    physics = float(env_cfg.physics_hz if physics_hz is None else physics_hz)
    camera = float(env_cfg.camera_hz if camera_hz is None else camera_hz)
    if not all(math.isfinite(rate) and rate > 0.0 for rate in (control, physics, camera)):
        raise ValueError("Control, physics, and camera rates must be finite and positive")
    if physics < max(control, camera):
        raise ValueError("Physics rate must be at least the control and camera rates")
    control_ratio = physics / control
    render_ratio = physics / camera
    if not math.isclose(control_ratio, round(control_ratio), abs_tol=1e-9):
        raise ValueError(f"Physics/control rate ratio must be an integer, got {control_ratio}")
    if not math.isclose(render_ratio, round(render_ratio), abs_tol=1e-9):
        raise ValueError(f"Physics/camera rate ratio must be an integer, got {render_ratio}")

    env_cfg.physics_hz = physics
    env_cfg.camera_hz = camera
    env_cfg.decimation = round(control_ratio)
    env_cfg.sim.dt = 1.0 / physics
    env_cfg.sim.render_interval = round(render_ratio)
    return control, physics, camera
