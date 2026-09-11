#!/usr/bin/env python3
"""Render one LeRobot v3 episode with named state/action policy labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


def _dependencies():
    try:
        import cv2
        import numpy as np
        import pyarrow.dataset as pyarrow_dataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "viewer requires opencv-python, numpy, and pyarrow"
        ) from error
    return cv2, np, pyarrow_dataset


def _read_info(root: Path) -> dict[str, Any]:
    path = root / "meta/info.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing LeRobot metadata: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("codebase_version") != "v3.0":
        raise ValueError(
            f"{path}: expected LeRobot codebase_version v3.0, "
            f"got {value.get('codebase_version')!r}"
        )
    return value


def _video_keys(info: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        name
        for name, feature in info.get("features", {}).items()
        if feature.get("dtype") == "video"
    )


def _feature_names(info: dict[str, Any], feature: str) -> tuple[str, ...]:
    try:
        names = info["features"][feature]["names"]
    except KeyError as error:
        raise ValueError(f"dataset has no {feature!r} feature") from error
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f"{feature!r} must have an explicit string name list")
    return tuple(names)


def select_signals(
    requested: Sequence[str],
    state_names: Sequence[str],
    action_names: Sequence[str],
) -> tuple[str, ...]:
    available = tuple(dict.fromkeys((*state_names, *action_names)))
    if not requested:
        return available
    unknown = [name for name in requested if name not in available]
    if unknown:
        raise ValueError(
            f"unknown signals {unknown}; available: {', '.join(available)}"
        )
    return tuple(dict.fromkeys(requested))


def _read_episode_table(root: Path, episode: int):
    _cv2, np, pyarrow_dataset = _dependencies()
    dataset = pyarrow_dataset.dataset(str(root / "data"), format="parquet")
    required = ("episode_index", "frame_index", "timestamp", "observation.state", "action")
    missing = [name for name in required if name not in dataset.schema.names]
    if missing:
        raise ValueError(f"{root}: missing data columns {missing}")
    table = dataset.to_table(
        columns=list(required),
        filter=pyarrow_dataset.field("episode_index") == episode,
    )
    if table.num_rows == 0:
        raise ValueError(f"episode {episode} does not exist in {root}")
    values = table.to_pydict()
    order = np.argsort(np.asarray(values["frame_index"], dtype=np.int64))
    return {
        "frame_index": np.asarray(values["frame_index"], dtype=np.int64)[order],
        "timestamp": np.asarray(values["timestamp"], dtype=np.float64)[order],
        "state": np.asarray(values["observation.state"], dtype=np.float32)[order],
        "action": np.asarray(values["action"], dtype=np.float32)[order],
    }


def _read_episode_metadata(root: Path, episode: int) -> dict[str, Any]:
    _cv2, _np, pyarrow_dataset = _dependencies()
    dataset = pyarrow_dataset.dataset(
        str(root / "meta/episodes"), format="parquet"
    )
    table = dataset.to_table(
        filter=pyarrow_dataset.field("episode_index") == episode,
    )
    if table.num_rows != 1:
        raise ValueError(
            f"episode {episode} metadata rows: expected 1, got {table.num_rows}"
        )
    return {name: table[name][0].as_py() for name in table.column_names}


def _resolve_video(
    root: Path,
    info: dict[str, Any],
    metadata: dict[str, Any],
    camera: str,
) -> tuple[Path, int, float]:
    prefix = f"videos/{camera}"
    file_key = f"{prefix}/file_index"
    chunk_key = f"{prefix}/chunk_index"
    from_key = f"{prefix}/from_timestamp"
    for key in (file_key, chunk_key, from_key):
        if key not in metadata:
            raise ValueError(f"episode metadata has no {key!r}")
    video_path = info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    )
    relative = video_path.format(
        video_key=camera,
        chunk_index=int(metadata[chunk_key]),
        file_index=int(metadata[file_key]),
    )
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing episode video: {path}")
    fps = float(info["features"][camera]["info"]["video.fps"])
    start_video_frame = round(float(metadata[from_key]) * fps)
    return path, start_video_frame, fps


def _put_text(cv2, image, text, xy, *, color=(235, 235, 235), scale=0.42):
    cv2.putText(
        image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
        (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
        color, 1, cv2.LINE_AA,
    )


def _write_csv(
    path: Path,
    rows: dict[str, Any],
    state_names: Sequence[str],
    action_names: Sequence[str],
) -> None:
    headings = ["frame_index", "timestamp_s"]
    headings.extend(f"state.{name}" for name in state_names)
    headings.extend(f"action.{name}" for name in action_names)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headings)
        for index in range(len(rows["frame_index"])):
            writer.writerow(
                [int(rows["frame_index"][index]), float(rows["timestamp"][index])]
                + [float(value) for value in rows["state"][index]]
                + [float(value) for value in rows["action"][index]]
            )


def _signal_summary(rows, state_names, action_names) -> dict[str, Any]:
    _cv2, np, _pyarrow_dataset = _dependencies()
    result: dict[str, Any] = {}
    for key, values, names in (
        ("state", rows["state"], state_names),
        ("action", rows["action"], action_names),
    ):
        result[key] = {
            name: {
                "first": float(values[0, index]),
                "min": float(np.min(values[:, index])),
                "max": float(np.max(values[:, index])),
                "last": float(values[-1, index]),
            }
            for index, name in enumerate(names)
        }
    if "lift_joint" in action_names:
        lift = rows["action"][:, action_names.index("lift_joint")]
        endpoint_error = np.minimum(np.abs(lift), np.abs(lift + 0.28))
        invalid = np.flatnonzero(endpoint_error > 1.0e-6)
        result["lift_endpoint_contract"] = {
            "allowed_m": [0.0, -0.28],
            "invalid_count": int(len(invalid)),
            "first_invalid_episode_frame": (
                int(rows["frame_index"][invalid[0]]) if len(invalid) else None
            ),
        }
    return result


def render_episode(args: argparse.Namespace) -> dict[str, Any]:
    cv2, np, _pyarrow_dataset = _dependencies()
    video_root = Path(args.dataset).expanduser().resolve()
    signal_root = (
        Path(args.signals_dataset).expanduser().resolve()
        if args.signals_dataset
        else video_root
    )
    video_info = _read_info(video_root)
    signal_info = _read_info(signal_root)
    cameras = _video_keys(video_info)
    camera = args.camera or (cameras[0] if cameras else None)
    if camera not in cameras:
        raise ValueError(f"camera must be one of: {', '.join(cameras)}")

    state_names = _feature_names(signal_info, "observation.state")
    action_names = _feature_names(signal_info, "action")
    signals = select_signals(args.signal, state_names, action_names)
    rows = _read_episode_table(signal_root, args.episode)
    video_metadata = _read_episode_metadata(video_root, args.episode)
    video_path, video_start, video_fps = _resolve_video(
        video_root, video_info, video_metadata, camera
    )
    expected_length = int(video_metadata["length"])
    if len(rows["frame_index"]) != expected_length:
        raise ValueError(
            f"signal rows {len(rows['frame_index'])} != video episode length "
            f"{expected_length}"
        )

    start = args.start_frame
    end = expected_length - 1 if args.end_frame is None else args.end_frame
    if start < 0 or end < start or end >= expected_length:
        raise ValueError(
            f"frame range [{start}, {end}] is outside episode length {expected_length}"
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    manifest_path = output.with_suffix(".json")
    sliced_rows = {
        name: values[start : end + 1] for name, values in rows.items()
    }
    _write_csv(csv_path, sliced_rows, state_names, action_names)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, video_start + start)
    writer = None
    try:
        state_index = {name: index for index, name in enumerate(state_names)}
        action_index = {name: index for index, name in enumerate(action_names)}
        for local in range(end - start + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"cannot decode {video_path} at frame {video_start + start + local}"
                )
            row = start + local
            panel_width = 720
            row_height = 18
            canvas_height = max(frame.shape[0], 72 + row_height * len(signals))
            if frame.shape[0] < canvas_height:
                frame = cv2.copyMakeBorder(
                    frame, 0, canvas_height - frame.shape[0], 0, 0,
                    cv2.BORDER_CONSTANT, value=(18, 20, 24),
                )
            panel = np.full((canvas_height, panel_width, 3), (30, 33, 39), np.uint8)
            canvas = np.hstack((frame, panel))
            x = frame.shape[1] + 14
            _put_text(
                cv2, canvas,
                f"episode {args.episode}  frame {row}/{expected_length - 1}  {video_fps:g} Hz",
                (14, 24), color=(255, 220, 80), scale=0.50,
            )
            _put_text(
                cv2, canvas, "signal                         state       action       action-state",
                (x, 24), color=(255, 255, 255), scale=0.40,
            )
            for signal_row, name in enumerate(signals):
                state_value = (
                    float(rows["state"][row, state_index[name]])
                    if name in state_index else None
                )
                action_value = (
                    float(rows["action"][row, action_index[name]])
                    if name in action_index else None
                )
                delta = (
                    action_value - state_value
                    if state_value is not None and action_value is not None else None
                )
                state_text = "       -" if state_value is None else f"{state_value:+9.5f}"
                action_text = "       -" if action_value is None else f"{action_value:+9.5f}"
                delta_text = "       -" if delta is None else f"{delta:+9.5f}"
                _put_text(
                    cv2, canvas,
                    f"{name:<28} {state_text}   {action_text}   {delta_text}",
                    (x, 54 + signal_row * row_height), scale=0.39,
                )
            if writer is None:
                writer = cv2.VideoWriter(
                    str(output), cv2.VideoWriter_fourcc(*"mp4v"), video_fps,
                    (canvas.shape[1], canvas.shape[0]),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"cannot create {output}")
            writer.write(canvas)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    manifest = {
        "dataset": str(video_root),
        "signals_dataset": str(signal_root),
        "episode_index": args.episode,
        "camera": camera,
        "source_video": str(video_path),
        "source_video_start_frame": video_start,
        "episode_length": expected_length,
        "rendered_frame_range": [start, end],
        "output_video": str(output),
        "output_csv": str(csv_path),
        "signals": list(signals),
        "summary": _signal_summary(sliced_rows, state_names, action_names),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="LeRobot v3 dataset root")
    parser.add_argument("--episode", required=True, type=int)
    parser.add_argument("--output", required=True, help="output MP4 path")
    parser.add_argument("--camera", help="video feature key; defaults to first video")
    parser.add_argument(
        "--signals-dataset",
        help="optional same-length dataset supplying state/action overlays",
    )
    parser.add_argument(
        "--signal", action="append", default=[],
        help="named signal to show; repeat this option, or omit it to show all",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(render_episode(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
