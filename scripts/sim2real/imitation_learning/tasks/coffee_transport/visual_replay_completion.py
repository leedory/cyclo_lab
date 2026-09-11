"""Fail-closed completion and resume checks for Coffee visual replay."""

from __future__ import annotations

import atexit
import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, TextIO


INCOMPLETE_EXIT_CODE = 70


class VisualReplayCompletionGuard:
    """Turn an otherwise clean interpreter exit into failure until finalized."""

    def __init__(
        self,
        *,
        exit_fn: Callable[[int], Any] = os._exit,
        stderr: TextIO = sys.stderr,
        register: bool = True,
    ) -> None:
        self._armed = False
        self._complete = False
        self._manifest_path: Path | None = None
        self._expected_outputs: int | None = None
        self._exit_fn = exit_fn
        self._stderr = stderr
        if register:
            atexit.register(self.fail_if_incomplete)

    def arm(self, manifest_path: Path, expected_outputs: int) -> None:
        self._armed = True
        self._complete = False
        self._manifest_path = manifest_path
        self._expected_outputs = expected_outputs

    def mark_complete(self) -> None:
        if not self._armed:
            raise RuntimeError("visual replay completion guard was not armed")
        self._complete = True

    def fail_if_incomplete(self) -> None:
        if not self._armed or self._complete:
            return
        print(
            "COFFEE_VISUAL_INCOMPLETE "
            f"expected_outputs={self._expected_outputs} manifest={self._manifest_path}",
            file=self._stderr,
            flush=True,
        )
        self._exit_fn(INCOMPLETE_EXIT_CODE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_indices(count: int) -> list[int]:
    return list(range(count))


def validate_resume_checkpoint(
    output: Path,
    manifest: Mapping[str, Any],
    *,
    batch_size: int,
    expected_outputs: int,
) -> int:
    """Validate every committed batch and artifact before resuming."""

    episodes = manifest.get("episodes", [])
    batches = manifest.get("batches", [])
    if not isinstance(episodes, list) or not isinstance(batches, list):
        raise ValueError("resume manifest episodes/batches must be lists")
    completed = len(episodes)
    if completed > expected_outputs:
        raise ValueError(f"resume has {completed} outputs, expected at most {expected_outputs}")
    if completed % batch_size:
        raise ValueError(f"resume boundary {completed} is not a {batch_size}-episode batch boundary")

    episode_indices = [int(record["episode_index"]) for record in episodes]
    if episode_indices != _expected_indices(completed):
        raise ValueError("resume episode indices are not contiguous from zero")

    batch_indices: list[int] = []
    for batch in batches:
        indices = [int(value) for value in batch.get("output_indices", [])]
        if len(indices) != batch_size:
            raise ValueError("resume manifest contains a partial batch")
        batch_indices.extend(indices)
    if batch_indices != _expected_indices(completed):
        raise ValueError("resume batch indices do not match committed episodes")

    for record in episodes:
        array_path = output / str(record["arrays"])
        if not array_path.is_file():
            raise ValueError(f"resume array is missing: {array_path}")
        if _sha256(array_path) != str(record["array_sha256"]):
            raise ValueError(f"resume array checksum mismatch: {array_path}")
        videos = record.get("videos")
        if not isinstance(videos, Mapping) or not videos:
            raise ValueError("resume episode has no videos")
        for relative in videos.values():
            video_path = output / str(relative)
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                raise ValueError(f"resume video is missing or empty: {video_path}")

    for key in ("episode_count", "total_frames"):
        if key in manifest and completed != expected_outputs:
            raise ValueError(f"partial resume unexpectedly contains final field: {key}")
    return completed


def finalize_manifest(
    manifest: dict[str, Any], *, expected_outputs: int, elapsed_s: float
) -> None:
    """Set final fields only after the complete output index contract passes."""

    episodes = manifest.get("episodes", [])
    indices = [int(record["episode_index"]) for record in episodes]
    if indices != _expected_indices(expected_outputs):
        raise ValueError(
            f"visual replay incomplete: got {len(indices)} ordered outputs, expected {expected_outputs}"
        )
    manifest["episode_count"] = expected_outputs
    manifest["total_frames"] = sum(int(item["length"]) for item in episodes)
    manifest["elapsed_s"] = float(elapsed_s)
