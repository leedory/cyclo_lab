"""Stream-copy-compatible H.264 frame writers backed by ffmpeg subprocesses.

Replaces ``cv2.VideoWriter`` in the Isaac visual-replay loop. Two reasons:

1. **Encoding leaves the Python process.** Each writer is its own ``ffmpeg``
   process, so encoding runs truly in parallel across cores with no GIL, and
   the render loop only performs a buffered ``stdin.write()``.
2. **The output qualifies for the LeRobot v3.0 stream-copy fast path.**
   ``cyclo_data/converter/to_lerobot_v30.py::_try_concatenate_videos_copy``
   concatenates per-camera episode videos with ``-c:v copy`` (no re-encode)
   when ``_videos_support_copy_concat`` accepts them. ``mp4v`` never qualifies,
   so today every staging video is encoded twice. The argument list below is
   built to satisfy every one of that gate's conditions.

The gate's conditions, and where each is satisfied (see ENCODER_ARGS):

    codec_name == "h264"                     -> -c:v libx264
    no audio stream                          -> -an
    |stream_fps - config.fps| <= 0.01        -> -r <fps>
    has_b_frames == 0                        -> -bf 0          <-- easy to miss
    frame count probe > 0                    -> non-empty output
    identical _video_stream_copy_signature   -> every field below pinned
    across all episodes of one camera            explicitly, never inferred

``_video_stream_copy_signature`` compares:
    (codec_name, profile, level, width, height, pix_fmt, sample_aspect_ratio,
     field_order, color_range, color_space, color_transfer, color_primaries,
     chroma_location, time_base, avg_frame_rate)

Aggregates are built **per camera**, so head (672x376) and wrist (480x640)
having different sizes is fine; only episodes of the *same* camera must match.

No third-party imports beyond ``imageio_ffmpeg``, which is present in the
``cyclo_lab`` image and bundles ffmpeg 7.0.2 with libx264. NVENC is NOT
available there and is not needed: encoding is off the critical path.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Sequence


class FrameWriterError(RuntimeError):
    """Raised when an encoder cannot be started, fed, or closed cleanly."""


def ffmpeg_executable() -> str:
    """Return an ffmpeg binary, preferring one already on PATH."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as error:  # pragma: no cover - environment guard
        raise FrameWriterError(
            "no ffmpeg on PATH and imageio_ffmpeg is not importable"
        ) from error
    return imageio_ffmpeg.get_ffmpeg_exe()


# Pinned so that every episode of a camera produces a byte-identical stream
# signature. Do not let ffmpeg infer any of these.
ENCODER_ARGS: tuple[str, ...] = (
    "-an",
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "18",
    "-pix_fmt", "yuv420p",
    "-bf", "0",
    "-profile:v", "high",
    "-level", "4.0",
    "-color_range", "tv",
    "-colorspace", "bt709",
    "-color_primaries", "bt709",
    "-color_trc", "bt709",
    "-video_track_timescale", "90000",
    "-movflags", "+faststart",
)

ENCODER_CONTRACT_ID = "coffee_staging_h264_streamcopy_v1"


def encoder_command(ffmpeg: str, path: Path, width: int, height: int, fps: int) -> list[str]:
    """Build the full ffmpeg argv for one raw-rgb24 -> h264 mp4 encoder."""
    if width % 2 or height % 2:
        # yuv420p needs even dimensions. 672x376 and 480x640 both satisfy this;
        # fail loudly rather than let ffmpeg silently pad.
        raise FrameWriterError(f"yuv420p requires even dimensions, got {width}x{height}")
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(int(fps)),
        "-i", "-",
        *ENCODER_ARGS,
        "-r", str(int(fps)),
        str(path),
    ]


class FfmpegFrameWriter:
    """One encoder subprocess for one (episode, camera) video."""

    def __init__(self, path: Path, width: int, height: int, fps: int, ffmpeg: str | None = None):
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.frames_written = 0
        self._expected_nbytes = self.width * self.height * 3
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._command = encoder_command(
            ffmpeg or ffmpeg_executable(), self.path, self.width, self.height, self.fps
        )
        try:
            self._process = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as error:
            raise FrameWriterError(f"could not start encoder for {self.path}: {error}") from error
        self._closed = False

    def write(self, frame) -> None:
        """Write one HxWx3 uint8 RGB frame.

        ``frame`` must already be contiguous RGB (not RGBA, not BGR). The
        caller slices and makes contiguous on the GPU; see B3 in the plan.
        """
        if self._closed:
            raise FrameWriterError(f"write after close: {self.path}")
        data = memoryview(frame).cast("B") if not isinstance(frame, (bytes, bytearray)) else frame
        if len(data) != self._expected_nbytes:
            raise FrameWriterError(
                f"{self.path}: frame is {len(data)} bytes, expected "
                f"{self._expected_nbytes} for {self.width}x{self.height} rgb24"
            )
        try:
            self._process.stdin.write(data)
        except BrokenPipeError as error:
            raise FrameWriterError(f"{self.path}: encoder died mid-stream: {self._stderr()}") from error
        self.frames_written += 1

    def _stderr(self) -> str:
        try:
            return (self._process.stderr.read() or b"").decode("utf-8", "replace").strip()
        except Exception:  # pragma: no cover - best effort diagnostics
            return "<unavailable>"

    def close(self) -> None:
        """Flush, wait, and require a clean exit. Raises on any failure."""
        if self._closed:
            return
        self._closed = True
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        code = self._process.wait(timeout=300)
        stderr = self._stderr()
        if code != 0:
            raise FrameWriterError(f"{self.path}: encoder exited {code}: {stderr}")
        if self.frames_written == 0:
            raise FrameWriterError(f"{self.path}: no frames were written")
        if not self.path.is_file() or self.path.stat().st_size == 0:
            raise FrameWriterError(f"{self.path}: encoder produced no output file")

    def abort(self) -> None:
        """Kill the encoder and remove any partial file. Never raises."""
        if self._closed:
            return
        self._closed = True
        try:
            self._process.stdin.close()
        except Exception:
            pass
        try:
            self._process.kill()
            self._process.wait(timeout=30)
        except Exception:
            pass
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass


class FrameWriterGroup:
    """All writers for one replay batch, keyed ``(env_index, camera_name)``.

    Mirrors Task000525's ``open_video_writers`` key shape so the replay loop
    changes as little as possible. Use as a context manager: a normal exit
    closes every writer strictly, and an exception aborts every writer and
    deletes the partial files, so a half-written video can never reach the
    manifest.
    """

    def __init__(
        self,
        output: Path,
        output_indices: Sequence[int],
        camera_map: Mapping[str, str],
        camera_shapes_hw: Mapping[str, tuple[int, int]],
        fps: int,
    ):
        self.fps = int(fps)
        self.paths: dict[tuple[int, str], Path] = {}
        self.writers: dict[tuple[int, str], FfmpegFrameWriter] = {}
        ffmpeg = ffmpeg_executable()
        try:
            for local_index, output_index in enumerate(output_indices):
                for source_camera, output_camera in camera_map.items():
                    height, width = camera_shapes_hw[source_camera]
                    path = (
                        Path(output) / "videos"
                        / f"episode_{int(output_index):06d}" / f"{output_camera}.mp4"
                    )
                    key = (local_index, source_camera)
                    self.writers[key] = FfmpegFrameWriter(
                        path, width=width, height=height, fps=self.fps, ffmpeg=ffmpeg
                    )
                    self.paths[key] = path
        except Exception:
            self.abort()
            raise

    def write(self, local_index: int, source_camera: str, frame) -> None:
        self.writers[(local_index, source_camera)].write(frame)

    def frames_written(self, local_index: int, source_camera: str) -> int:
        return self.writers[(local_index, source_camera)].frames_written

    def close(self) -> None:
        errors: list[str] = []
        for key, writer in self.writers.items():
            try:
                writer.close()
            except Exception as error:
                errors.append(f"{key}: {error}")
        if errors:
            raise FrameWriterError("; ".join(errors))

    def abort(self) -> None:
        for writer in self.writers.values():
            writer.abort()

    def __enter__(self) -> "FrameWriterGroup":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False


def expected_video_paths(
    output: Path, output_indices: Iterable[int], camera_map: Mapping[str, str]
) -> dict[tuple[int, str], Path]:
    """Paths a group would produce, for manifest construction without opening."""
    result: dict[tuple[int, str], Path] = {}
    for local_index, output_index in enumerate(output_indices):
        for source_camera, output_camera in camera_map.items():
            result[(local_index, source_camera)] = (
                Path(output) / "videos"
                / f"episode_{int(output_index):06d}" / f"{output_camera}.mp4"
            )
    return result
