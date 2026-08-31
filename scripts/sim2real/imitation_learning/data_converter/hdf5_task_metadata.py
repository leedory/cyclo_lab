"""Read task identity from Isaac Lab HDF5 metadata."""

from __future__ import annotations

from collections.abc import Iterable
import json
import re
from typing import Any


def read_task_env_name(data_group: Any) -> str | None:
    """Return the registered environment name stored on an HDF5 data group."""
    direct_name = data_group.attrs.get("task_env_name")
    if direct_name is not None:
        if isinstance(direct_name, bytes):
            direct_name = direct_name.decode("utf-8")
        return str(direct_name)

    serialized = data_group.attrs.get("env_args")
    if serialized is None:
        return None
    if isinstance(serialized, bytes):
        serialized = serialized.decode("utf-8")
    try:
        env_args = json.loads(str(serialized))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(env_args, dict) or not env_args.get("env_name"):
        return None
    return str(env_args["env_name"])


def is_task_hdf(
    data_group: Any,
    task_id: str | int,
    *,
    env_aliases: Iterable[str] = (),
) -> bool:
    """Return whether HDF5 metadata identifies the requested numeric task."""
    match = re.fullmatch(r"(?:task[-_]?)?0*(\d+)", str(task_id).lower())
    if match is None:
        raise ValueError(f"task_id must be numeric or task-prefixed: {task_id!r}")

    env_name = read_task_env_name(data_group)
    if env_name is None:
        return False
    normalized_name = env_name.lower().replace("_", "-")
    normalized_aliases = {alias.lower().replace("_", "-") for alias in env_aliases}
    if normalized_name in normalized_aliases:
        return True

    numeric_id = str(int(match.group(1)))
    pattern = rf"(?:^|-)task-?0*{re.escape(numeric_id)}(?:-|$)"
    return re.search(pattern, normalized_name) is not None
