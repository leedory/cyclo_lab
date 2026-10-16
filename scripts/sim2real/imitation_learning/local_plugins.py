"""Discover local sim-to-real environment and task plugins.

The plugin roots are intentionally ignored by Git.  This module is the tracked
contract between the reusable recorder/generation tools and locally authored
scene/task packages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any


_TASK_PATTERN = re.compile(
    r"^Cyclo-Real-(?P<environment>.+?)-Task(?P<task_id>\d{6})(?:-|$)"
)


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load local plugin: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LocalPluginBundle:
    """Pre-launch-safe handles for one environment and one task plugin."""

    task_name: str
    environment_key: str | None
    task_id: str | None
    environment_path: Path | None
    task_path: Path | None
    environment: ModuleType | None
    task: ModuleType | None

    @property
    def found(self) -> bool:
        return self.environment is not None or self.task is not None

    def add_cli_arguments(self, parser: Any) -> None:
        for module in (self.environment, self.task):
            callback = getattr(module, "add_cli_arguments", None)
            if callback is not None:
                callback(parser)

    def configure_launcher_args(self, args: Any) -> None:
        for module in (self.environment, self.task):
            callback = getattr(module, "configure_launcher_args", None)
            if callback is not None:
                callback(args)

    def register_environments(self) -> None:
        for label, module in (("environment", self.environment), ("task", self.task)):
            if module is None:
                continue
            callback = getattr(module, "register_environments", None)
            if callback is None:
                raise RuntimeError(
                    f"Local {label} plugin must define register_environments(): "
                    f"{getattr(module, '__file__', '<unknown>')}"
                )
            callback()

    def build_recording_factory(self, args: Any):
        if self.task is None:
            return None
        callback = getattr(self.task, "build_recording_factory", None)
        return None if callback is None else callback(args)

    def mimic_data_generator_class(self):
        """Return an optional task-specific MimicGen DataGenerator class."""

        if self.task is None:
            return None
        callback = getattr(self.task, "mimic_data_generator_class", None)
        return None if callback is None else callback()

    def provenance(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "local_environment_key": self.environment_key or "",
            "local_task_id": self.task_id or "",
        }
        if self.environment_path is not None:
            result["local_environment_plugin_sha256"] = _sha256(self.environment_path)
        if self.task_path is not None:
            result["local_task_plugin_sha256"] = _sha256(self.task_path)
        for prefix, module in (("environment", self.environment), ("task", self.task)):
            callback = getattr(module, "provenance", None) if module is not None else None
            if callback is not None:
                for key, value in callback().items():
                    result[f"local_{prefix}_{key}"] = value
        return result


def _plugin_root() -> Path:
    return Path(__file__).resolve().parent


def load_for_task(task_name: str | None) -> LocalPluginBundle:
    """Load matching local plugins without importing Isaac runtime modules."""

    normalized_task_name = task_name or ""
    match = _TASK_PATTERN.match(normalized_task_name)
    if match is None:
        return LocalPluginBundle(normalized_task_name, None, None, None, None, None, None)

    environment_key = _snake_case(match.group("environment"))
    task_id = match.group("task_id")
    root = _plugin_root()
    environment_path = root / "environments/local" / environment_key / "plugin.py"
    task_path = root / "tasks/local" / f"task_{task_id}" / "plugin.py"
    environment = (
        _load_module(environment_path, f"_cyclo_local_environment_{environment_key}")
        if environment_path.is_file()
        else None
    )
    task = (
        _load_module(task_path, f"_cyclo_local_task_{task_id}")
        if task_path.is_file()
        else None
    )
    if task is not None:
        declared_environment = getattr(task, "ENVIRONMENT", environment_key)
        if _snake_case(str(declared_environment)) != environment_key:
            raise ValueError(
                f"Local task {task_id} declares environment {declared_environment!r}, "
                f"but task name requests {environment_key!r}."
            )
    return LocalPluginBundle(
        normalized_task_name,
        environment_key,
        task_id,
        environment_path if environment is not None else None,
        task_path if task is not None else None,
        environment,
        task,
    )


def task_name_from_hdf5(path: str | Path) -> str | None:
    """Read an Isaac Lab environment name before launching Isaac Sim.

    New recorder files carry ``task_env_name`` directly. Older Isaac Lab files
    only carry it inside the JSON encoded ``env_args`` attribute.
    """

    import h5py

    with h5py.File(Path(path).expanduser(), "r") as dataset:
        data = dataset.get("data")
        if data is None:
            return None
        direct = data.attrs.get("task_env_name")
        if isinstance(direct, bytes):
            direct = direct.decode("utf-8")
        if isinstance(direct, str) and direct:
            return direct
        encoded = data.attrs.get("env_args")
        if isinstance(encoded, bytes):
            encoded = encoded.decode("utf-8")
        if not isinstance(encoded, str) or not encoded:
            return None
        payload = json.loads(encoded)
        value = payload.get("env_name")
        return value if isinstance(value, str) and value else None
