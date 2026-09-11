"""Renderer frame-0 sanity metrics, ported to run on in-memory frames.

Today this gate lives in
``tasks/coffee_transport/audit_and_select_physical_trajectories.py::_frame0_camera_metrics``
and reads ``obs/cam_head`` out of the generation HDF5. After Work Package A the
generation HDF5 has no pixels, so the gate has to move to where the pixels now
are: the visual-replay staging run.

The arithmetic below is a **line-for-line port** of the original so the two
implementations agree exactly; ``tests/test_frame0_metrics_parity.py`` proves
that against real episodes. The only difference is the input type: the original
takes an ``h5py.Dataset``, this takes any sequence of frames, so the replay loop
can pass frames it already holds in memory (negligible extra cost).

Thresholds are copied from the original module. Keep them in sync; if the
audit's constants change, change them here too.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np


# Copied verbatim from audit_and_select_physical_trajectories.py
FRAME0_JUMP_ABS_LIMIT = 45.0
FRAME0_JUMP_RATIO_LIMIT = 4.0
FRAME0_MIN_RANGE = 8
FRAME0_MIN_STD = 2.0

FRAME0_GATE_ID = "coffee_renderer_frame0_sanity_v1"
LEADING_FRAMES = 8


def frame_hash(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def frame0_metrics(leading_frames: Sequence[np.ndarray], last_frame: np.ndarray) -> dict[str, Any]:
    """Metrics for one episode of one camera.

    ``leading_frames`` must be the first ``min(len(episode), 8)`` frames in
    order; ``last_frame`` the episode's final frame. Both as HxWx3 uint8.
    """
    if not len(leading_frames):
        raise ValueError("frame0 metrics need at least one leading frame")
    frames = [np.asarray(frame) for frame in leading_frames[:LEADING_FRAMES]]
    first = frames[0]
    deltas = [
        float(np.mean(np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32))))
        for i in range(len(frames) - 1)
    ]
    baseline = float(np.median(deltas[1:])) if len(deltas) > 1 else 0.0
    ratio = deltas[0] / max(baseline, 1.0e-6) if deltas else 0.0
    return {
        "frame0_hash": frame_hash(first),
        "last_frame_hash": frame_hash(np.asarray(last_frame)),
        "frame0_mean": float(np.mean(first)),
        "frame0_std": float(np.std(first)),
        "frame0_range": int(np.max(first)) - int(np.min(first)),
        "frame0_to_frame1_mad": deltas[0] if deltas else 0.0,
        "following_delta_median": baseline,
        "frame0_jump_ratio": ratio,
    }


class Frame0Collector:
    """Accumulates the frames the gate needs while the replay loop runs.

    Holds at most ``LEADING_FRAMES + 1`` frames per (episode, camera), so memory
    stays flat regardless of episode length.
    """

    def __init__(self) -> None:
        self._leading: dict[tuple[int, str], list[np.ndarray]] = {}
        self._last: dict[tuple[int, str], np.ndarray] = {}

    def observe(self, local_index: int, camera: str, frame: np.ndarray) -> None:
        key = (local_index, camera)
        leading = self._leading.setdefault(key, [])
        if len(leading) < LEADING_FRAMES:
            leading.append(np.array(frame, copy=True))
        self._last[key] = np.array(frame, copy=True)

    def metrics(self, local_index: int, camera: str) -> dict[str, Any]:
        key = (local_index, camera)
        if key not in self._leading:
            raise KeyError(f"no frames observed for {key}")
        return frame0_metrics(self._leading[key], self._last[key])

    def cameras(self, local_index: int) -> list[str]:
        return sorted(camera for index, camera in self._leading if index == local_index)


def check_frame0_metrics(
    camera: str,
    metrics: dict[str, Any],
    *,
    previous_last_hash: str | None = None,
) -> list[str]:
    """Return rejection reasons, empty when the episode passes.

    Mirrors the accept/reject logic of ``_check_cameras``. The cross-episode
    reuse and jump checks were head-only in the original; here they apply to
    whichever camera is passed, and the caller decides which cameras are gated
    (see the README: gate ``cam_head`` to preserve the original behaviour, and
    report the wrist numbers without gating until a wrist baseline exists).
    """
    reasons: list[str] = []
    if metrics["frame0_range"] < FRAME0_MIN_RANGE or metrics["frame0_std"] < FRAME0_MIN_STD:
        reasons.append(f"{camera} frame 0 is blank or nearly constant")
    if previous_last_hash and metrics["frame0_hash"] == previous_last_hash:
        reasons.append(f"{camera} frame 0 exactly reuses the previous episode's last frame")
    if (
        metrics["frame0_to_frame1_mad"] > FRAME0_JUMP_ABS_LIMIT
        and metrics["frame0_jump_ratio"] > FRAME0_JUMP_RATIO_LIMIT
    ):
        reasons.append(
            f"{camera} frame 0 to 1 jump is an isolated reset outlier "
            f"(MAD={metrics['frame0_to_frame1_mad']:.3f}, ratio={metrics['frame0_jump_ratio']:.2f})"
        )
    return reasons
