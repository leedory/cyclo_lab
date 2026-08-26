#!/usr/bin/env python3
"""Render per-episode, three-camera previews from an Isaac Lab HDF5 dataset.

The renderer reads recorded RGB observations directly. It does not launch Isaac
Sim or modify the source dataset. By default, previews are written next to the
dataset in ``<dataset_stem>_camera_previews``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import cv2
import h5py
import imageio_ffmpeg
import numpy as np


DEFAULT_CAMERAS = ("cam_head", "cam_wrist_left", "cam_wrist_right")
DEFAULT_ROTATIONS = {"cam_wrist_left": 1, "cam_wrist_right": 1}
DEFAULT_LABELS = {
    "cam_head": "Head",
    "cam_wrist_left": "Wrist Left",
    "cam_wrist_right": "Wrist Right",
}


@dataclass(frozen=True)
class EpisodePreview:
    """Metadata for one rendered episode preview."""

    episode: str
    video: str
    frames: int
    width: int
    height: int


def _natural_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def _parse_assignments(values: Sequence[str], *, value_name: str) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for value in values:
        try:
            name, raw_number = value.split("=", 1)
            number = int(raw_number)
        except ValueError as exc:
            raise ValueError(f"Invalid {value_name} {value!r}; expected CAMERA=INTEGER.") from exc
        if not name:
            raise ValueError(f"Invalid {value_name} {value!r}; camera name is empty.")
        assignments[name] = number
    return assignments


def _as_rgb(frame: np.ndarray, camera_name: str) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(
            f"Camera {camera_name!r} frame must have shape (height, width, >=3); got {frame.shape}."
        )
    frame = frame[..., :3]
    if frame.dtype == np.uint8:
        return np.ascontiguousarray(frame)
    if np.issubdtype(frame.dtype, np.floating):
        finite = frame[np.isfinite(frame)]
        if finite.size and float(finite.max()) <= 1.0:
            frame = frame * 255.0
    return np.ascontiguousarray(np.nan_to_num(frame, nan=0.0).clip(0, 255).astype(np.uint8))


def _prepare_panel(frame: np.ndarray, camera_name: str, panel_width: int, quarter_turns: int) -> np.ndarray:
    frame = _as_rgb(frame, camera_name)
    if quarter_turns % 4:
        frame = np.rot90(frame, k=quarter_turns % 4)
    scale = panel_width / frame.shape[1]
    panel_height = max(1, int(round(frame.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (panel_width, panel_height), interpolation=interpolation)


def _compose_frame(
    frames: Sequence[np.ndarray],
    camera_names: Sequence[str],
    *,
    labels: Mapping[str, str],
    panel_width: int,
    rotations: Mapping[str, int],
    content_height: int,
    label_height: int,
    gap: int,
    frame_index: int,
    frame_count: int,
) -> np.ndarray:
    panels = [
        _prepare_panel(frame, name, panel_width, rotations.get(name, 0))
        for name, frame in zip(camera_names, frames)
    ]
    output_height = label_height + content_height
    output_width = panel_width * len(panels) + gap * (len(panels) - 1)
    canvas = np.full((output_height, output_width, 3), 18, dtype=np.uint8)

    x = 0
    for name, panel in zip(camera_names, panels):
        y = label_height + (content_height - panel.shape[0]) // 2
        canvas[y : y + panel.shape[0], x : x + panel_width] = panel
        cv2.putText(
            canvas,
            labels.get(name, name),
            (x + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        x += panel_width + gap

    counter = f"{frame_index + 1}/{frame_count}"
    (text_width, _), _ = cv2.getTextSize(counter, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(
        canvas,
        counter,
        (output_width - text_width - 10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _episode_camera_datasets(
    episode: h5py.Group, camera_names: Sequence[str]
) -> tuple[list[h5py.Dataset], int]:
    if "obs" not in episode or not isinstance(episode["obs"], h5py.Group):
        raise ValueError(f"Episode {episode.name!r} does not contain an 'obs' group.")
    observations = episode["obs"]
    missing = [name for name in camera_names if name not in observations]
    if missing:
        raise ValueError(f"Episode {episode.name!r} is missing camera observations: {missing}.")

    datasets = [observations[name] for name in camera_names]
    invalid = [dataset.name for dataset in datasets if dataset.ndim != 4 or dataset.shape[-1] < 3]
    if invalid:
        raise ValueError(f"Camera datasets must have shape (frames, height, width, channels): {invalid}.")
    frame_counts = {int(dataset.shape[0]) for dataset in datasets}
    if len(frame_counts) != 1:
        details = {name: int(dataset.shape[0]) for name, dataset in zip(camera_names, datasets)}
        raise ValueError(f"Episode {episode.name!r} camera lengths do not match: {details}.")
    frame_count = frame_counts.pop()
    if frame_count == 0:
        raise ValueError(f"Episode {episode.name!r} has no camera frames.")
    return datasets, frame_count


def _render_episode(
    episode_name: str,
    episode: h5py.Group,
    output_path: Path,
    *,
    camera_names: Sequence[str],
    labels: Mapping[str, str],
    rotations: Mapping[str, int],
    fps: float,
    panel_width: int,
    overwrite: bool,
) -> EpisodePreview:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Preview already exists: {output_path}. Pass --overwrite to replace it.")

    datasets, frame_count = _episode_camera_datasets(episode, camera_names)
    first_panels = [
        _prepare_panel(dataset[0], name, panel_width, rotations.get(name, 0))
        for name, dataset in zip(camera_names, datasets)
    ]
    content_height = max(panel.shape[0] for panel in first_panels)
    label_height = 42
    gap = 8
    output_width = panel_width * len(camera_names) + gap * (len(camera_names) - 1)
    output_height = label_height + content_height
    # Common MP4 encoders require even dimensions.
    output_width += output_width % 2
    output_height += output_height % 2

    temporary_path = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    if temporary_path.exists():
        temporary_path.unlink()
    writer = imageio_ffmpeg.write_frames(
        str(temporary_path),
        (output_width, output_height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        quality=7,
        macro_block_size=2,
        ffmpeg_log_level="error",
        output_params=["-movflags", "+faststart"],
    )

    try:
        writer.send(None)
        for frame_index in range(frame_count):
            canvas = _compose_frame(
                [dataset[frame_index] for dataset in datasets],
                camera_names,
                labels=labels,
                panel_width=panel_width,
                rotations=rotations,
                content_height=content_height,
                label_height=label_height,
                gap=gap,
                frame_index=frame_index,
                frame_count=frame_count,
            )
            if canvas.shape[1] != output_width or canvas.shape[0] != output_height:
                canvas = cv2.copyMakeBorder(
                    canvas,
                    0,
                    output_height - canvas.shape[0],
                    0,
                    output_width - canvas.shape[1],
                    cv2.BORDER_CONSTANT,
                    value=(18, 18, 18),
                )
            writer.send(canvas)
    except Exception:
        writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    writer.close()
    temporary_path.replace(output_path)
    return EpisodePreview(episode_name, output_path.name, frame_count, output_width, output_height)


def _write_index(
    output_dir: Path,
    dataset_file: Path,
    previews: Sequence[EpisodePreview],
    *,
    camera_names: Sequence[str],
    rotations: Mapping[str, int],
    fps: float,
) -> Path:
    manifest = {
        "schema": "cyclo_lab_hdf5_camera_previews/v1",
        "source_dataset": str(dataset_file),
        "fps": fps,
        "cameras": list(camera_names),
        "quarter_turns_ccw": {name: int(rotations.get(name, 0)) % 4 for name in camera_names},
        "episodes": [preview.__dict__ for preview in previews],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    cards = "\n".join(
        f"""<article>
  <h2>{html.escape(preview.episode)}</h2>
  <video controls preload="metadata" src="{html.escape(preview.video)}"></video>
  <p>{preview.frames} frames · {fps:g} FPS · {preview.width}×{preview.height}</p>
</article>"""
        for preview in previews
    )
    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(dataset_file.stem)} camera previews</title>
  <style>
    :root {{ color-scheme: dark; font-family: sans-serif; }}
    body {{ margin: 0 auto; max-width: 1500px; padding: 24px; background: #101214; color: #f2f2f2; }}
    h1 {{ margin-bottom: 6px; }}
    .source {{ color: #aeb6bf; overflow-wrap: anywhere; }}
    article {{ margin: 28px 0 42px; }}
    video {{ width: 100%; background: #000; border-radius: 8px; }}
    p {{ color: #aeb6bf; }}
  </style>
</head>
<body>
  <h1>{html.escape(dataset_file.stem)}</h1>
  <div class="source">{html.escape(str(dataset_file))}</div>
  {cards}
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(index, encoding="utf-8")
    return index_path


def render_dataset(
    dataset_file: Path,
    *,
    output_dir: Path | None = None,
    camera_names: Sequence[str] = DEFAULT_CAMERAS,
    rotations: Mapping[str, int] = DEFAULT_ROTATIONS,
    labels: Mapping[str, str] = DEFAULT_LABELS,
    fps: float = 15.0,
    panel_width: int = 480,
    overwrite: bool = False,
) -> Path:
    """Render every episode and return the generated HTML index path."""
    dataset_file = Path(dataset_file).expanduser().resolve()
    if not dataset_file.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {dataset_file}")
    if not camera_names:
        raise ValueError("At least one camera name is required.")
    if fps <= 0.0:
        raise ValueError(f"FPS must be positive, got {fps}.")
    if panel_width <= 0:
        raise ValueError(f"Panel width must be positive, got {panel_width}.")

    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else dataset_file.with_name(f"{dataset_file.stem}_camera_previews")
    )

    try:
        hdf5_file = h5py.File(dataset_file, "r")
    except OSError as exc:
        raise ValueError(
            f"Could not open {dataset_file} as a complete HDF5 dataset. "
            "The file may still be recording or may not contain any saved episodes."
        ) from exc

    with hdf5_file:
        if "data" not in hdf5_file or not isinstance(hdf5_file["data"], h5py.Group):
            raise ValueError(f"Dataset {dataset_file} does not contain a 'data' group.")
        episode_names = sorted(hdf5_file["data"].keys(), key=_natural_key)
        if not episode_names:
            raise ValueError(f"Dataset {dataset_file} does not contain any episodes under 'data'.")

        output_dir.mkdir(parents=True, exist_ok=True)
        previews = []
        for episode_name in episode_names:
            preview = _render_episode(
                episode_name,
                hdf5_file["data"][episode_name],
                output_dir / f"{episode_name}_three_cameras.mp4",
                camera_names=camera_names,
                labels=labels,
                rotations=rotations,
                fps=fps,
                panel_width=panel_width,
                overwrite=overwrite,
            )
            previews.append(preview)
            print(f"[camera preview] {episode_name}: {preview.frames} frames -> {preview.video}", flush=True)

    return _write_index(
        output_dir,
        dataset_file,
        previews,
        camera_names=camera_names,
        rotations=rotations,
        fps=fps,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", "--input_file", type=Path, required=True, help="Source HDF5 dataset.")
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        help="Output directory. Defaults to <input_stem>_camera_previews beside the HDF5 file.",
    )
    parser.add_argument("--fps", type=float, default=15.0, help="Preview playback frame rate (default: 15).")
    parser.add_argument("--panel-width", type=int, default=480, help="Width of each camera panel (default: 480).")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=list(DEFAULT_CAMERAS),
        metavar="CAMERA",
        help="Observation camera keys in left-to-right order.",
    )
    parser.add_argument(
        "--rotation",
        action="append",
        default=[],
        metavar="CAMERA=QUARTER_TURNS",
        help="Counter-clockwise 90-degree rotations. May be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing episode preview videos.")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        rotations = dict(DEFAULT_ROTATIONS)
        rotations.update(_parse_assignments(args.rotation, value_name="rotation"))
        index_path = render_dataset(
            args.input_file,
            output_dir=args.output_dir,
            camera_names=args.cameras,
            rotations=rotations,
            fps=args.fps,
            panel_width=args.panel_width,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"[camera preview] ERROR: {exc}")
        return 1
    print(f"[camera preview] index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
