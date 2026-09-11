"""Small helpers for running a controller slower than the physics loop."""

import math


class ControllerRateScheduler:
    """Run a controller at an integer divisor of the physics rate."""

    def __init__(self, physics_dt: float, controller_hz: float):
        if physics_dt <= 0.0:
            raise ValueError(f"physics_dt must be positive, got {physics_dt}.")
        if controller_hz <= 0.0:
            raise ValueError(f"controller_hz must be positive, got {controller_hz}.")

        self.physics_hz = 1.0 / float(physics_dt)
        self.controller_hz = float(controller_hz)
        rate_ratio = self.physics_hz / self.controller_hz
        rounded_ratio = round(rate_ratio)
        if rounded_ratio < 1 or not math.isclose(
            rate_ratio, rounded_ratio, rel_tol=0.0, abs_tol=1.0e-6
        ):
            raise ValueError(
                "physics rate must be an integer multiple of the controller rate: "
                f"physics={self.physics_hz:g} Hz, controller={self.controller_hz:g} Hz."
            )

        self.physics_ticks_per_update = int(rounded_ratio)
        self.controller_dt = 1.0 / self.controller_hz
        self._physics_tick = 0

    def update_due(self) -> bool:
        due = self._physics_tick == 0
        self._physics_tick = (self._physics_tick + 1) % self.physics_ticks_per_update
        return due

    def reset(self):
        self._physics_tick = 0
