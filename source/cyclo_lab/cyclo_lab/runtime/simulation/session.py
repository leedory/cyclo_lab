"""Small, file-backed status contract for one UI-launched Isaac session."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


DEFAULT_SESSION_STATUS_FILE = "/tmp/cyclo_lab_ui_session.json"


class SimulationSession:
    """Publish one simulator process' lifecycle without adding a daemon."""

    def __init__(
        self,
        *,
        task: str,
        bridge: str,
        status_file: str | Path = DEFAULT_SESSION_STATUS_FILE,
    ) -> None:
        self.status_file = Path(status_file)
        now = time.time()
        self._status: dict[str, Any] = {
            "state": "starting",
            "task": task,
            "bridge": bridge,
            "pid": os.getpid(),
            "reset_count": 0,
            "control_hz": None,
            "camera_hz": None,
            "step_count": 0,
            "observation_sequence": 0,
            "camera_sequence": 0,
            "started_at": now,
            "updated_at": now,
            "message": "Creating Isaac environment",
            "error": None,
        }
        self._last_heartbeat = 0.0
        self._write()

    @property
    def state(self) -> str:
        return str(self._status["state"])

    def ready(self, *, control_hz: float, camera_hz: float) -> None:
        self._status["control_hz"] = float(control_hz)
        self._status["camera_hz"] = float(camera_hz)
        self._transition("ready", "Environment and topic bridge are ready")

    def begin_reset(self, source: str) -> None:
        self._transition("resetting", f"Reset requested by {source}")

    def finish_reset(self) -> None:
        self._status["reset_count"] = int(self._status["reset_count"]) + 1
        self._transition("ready", "Reset complete; waiting for fresh observations")

    def heartbeat(
        self,
        step_count: int,
        *,
        observation_sequence: int,
        camera_sequence: int,
        force: bool = False,
        min_interval_s: float = 1.0,
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last_heartbeat < min_interval_s:
            return
        self._last_heartbeat = now
        self._status["step_count"] = int(step_count)
        self._status["observation_sequence"] = int(observation_sequence)
        self._status["camera_sequence"] = int(camera_sequence)
        self._status["updated_at"] = time.time()
        self._write()

    def stopping(self) -> None:
        if self.state != "error":
            self._transition("stopping", "Closing Isaac environment")

    def stopped(self) -> None:
        if self.state != "error":
            self._transition("stopped", "Isaac environment stopped")

    def fail(self, error: BaseException | str) -> None:
        self._status["error"] = str(error)
        self._transition("error", "Isaac environment failed")

    def _transition(self, state: str, message: str) -> None:
        self._status["state"] = state
        self._status["message"] = message
        self._status["updated_at"] = time.time()
        self._write()

    def _write(self) -> None:
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._status, indent=2, sort_keys=True) + "\n"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.status_file.parent,
                prefix=f".{self.status_file.name}.",
                delete=False,
            ) as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                temporary_path = Path(stream.name)
            os.replace(temporary_path, self.status_file)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
