#!/usr/bin/env python3
"""Plot reviewed spawn ranges and successful coffee-transport initial poses."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


TASK_IDS = ("000001", "000002")
CAN_NAMES = ("coffee_can_right", "coffee_can_left")

# These values mirror coffee_transport_common.py. Keeping this plotting utility
# Isaac-free makes it usable in dataset audit containers as well as Isaac Sim.
ROBOT_NOMINAL_XY_M = (-1.47138, 0.80011813)
ROBOT_NOMINAL_YAW_RAD = math.pi
ROBOT_HOME_XY_HALF_RANGE_M = 0.02
ROBOT_HOME_YAW_HALF_RANGE_RAD = math.radians(1.0)
LIFT_RANGE_M = (-0.28, 0.0)
ROUTE_RANGE_M = (0.0, 0.815072039386852)

# Active depth is the reviewed robot-side half of the former 12 cm depth.
ACTIVE_RANGES_M = {
    "coffee_can_right": (-2.20137, -2.14137, 0.93394, 1.05394),
    "coffee_can_left": (-2.20137, -2.14137, 0.52042, 0.64042),
}


@dataclass(frozen=True)
class EpisodePose:
    episode_id: str
    task_id: str
    generation_seed: int
    values: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot successful initial poses from coffee-transport HDF5 datasets."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Input HDF5. May be specified more than once.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for PNG, CSV, and JSON audit artifacts.",
    )
    parser.add_argument(
        "--task-id",
        choices=TASK_IDS,
        help="Expected task id. Required only when input metadata is incomplete.",
    )
    return parser.parse_args()


def scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def wrap_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def required_float(attrs: h5py.AttributeManager, key: str, source: str) -> float:
    if key not in attrs:
        raise KeyError(f"{source}: missing required episode attribute {key!r}")
    return float(attrs[key])


def load_successes(paths: list[Path], expected_task_id: str | None) -> list[EpisodePose]:
    episodes: list[EpisodePose] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            if "data" not in handle:
                raise KeyError(f"{path}: missing /data group")
            raw_file_task_id = str(
                scalar(handle["data"].attrs.get("task_id", ""))
            ).strip()
            file_task_id = raw_file_task_id.zfill(6) if raw_file_task_id else ""
            for demo_id in sorted(handle["data"].keys()):
                group = handle["data"][demo_id]
                attrs = group.attrs
                if not bool(scalar(attrs.get("success", False))):
                    continue
                raw_task_id = str(
                    scalar(attrs.get("task_id", file_task_id or expected_task_id or ""))
                )
                task_id = raw_task_id.zfill(6)
                if task_id not in TASK_IDS:
                    raise ValueError(f"{path}:{demo_id}: unsupported task_id={raw_task_id!r}")
                if expected_task_id and task_id != expected_task_id:
                    raise ValueError(
                        f"{path}:{demo_id}: expected task {expected_task_id}, got {task_id}"
                    )
                keys = [
                    "robot_spawn_x_m",
                    "robot_spawn_y_m",
                    "robot_spawn_yaw_rad",
                    "robot_home_x_m",
                    "robot_home_y_m",
                    "robot_home_yaw_rad",
                    "lift_spawn_m",
                    "base_route_spawn_offset_m",
                ]
                for can_name in CAN_NAMES:
                    keys.extend(
                        (
                            f"{can_name}_x_m",
                            f"{can_name}_y_m",
                            f"{can_name}_yaw_rad",
                        )
                    )
                values = {
                    key: required_float(attrs, key, f"{path}:{demo_id}") for key in keys
                }
                episodes.append(
                    EpisodePose(
                        episode_id=f"{path.name}:{demo_id}",
                        task_id=task_id,
                        generation_seed=int(scalar(attrs.get("generation_seed", -1))),
                        values=values,
                    )
                )
    if not episodes:
        raise ValueError("no successful episodes found in the supplied HDF5 files")
    task_ids = {episode.task_id for episode in episodes}
    if len(task_ids) != 1:
        raise ValueError(f"plot one task at a time; found task ids {sorted(task_ids)}")
    return episodes


def original_range(active: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_min, x_max, y_min, y_max = active
    active_depth = x_max - x_min
    return x_min - active_depth, x_max, y_min, y_max


def add_yaw_arrows(
    axis: plt.Axes,
    xs: np.ndarray,
    ys: np.ndarray,
    yaws: np.ndarray,
    *,
    color: str,
    length: float,
    alpha: float = 0.9,
) -> None:
    axis.quiver(
        xs,
        ys,
        np.cos(yaws),
        np.sin(yaws),
        color=color,
        angles="xy",
        scale_units="xy",
        scale=1.0 / length,
        width=0.004,
        headwidth=4.0,
        alpha=alpha,
        zorder=5,
    )


def value_array(episodes: list[EpisodePose], key: str) -> np.ndarray:
    return np.asarray([episode.values[key] for episode in episodes], dtype=np.float64)


def plot_top_down(axis: plt.Axes, episodes: list[EpisodePose]) -> None:
    task_id = episodes[0].task_id
    colors = {"coffee_can_right": "#d55e00", "coffee_can_left": "#0072b2"}
    labels = {"coffee_can_right": "right can", "coffee_can_left": "left can"}

    for index, can_name in enumerate(CAN_NAMES):
        active = ACTIVE_RANGES_M[can_name]
        full = original_range(active)
        axis.add_patch(
            Rectangle(
                (full[0], full[2]),
                full[1] - full[0],
                full[3] - full[2],
                fill=False,
                edgecolor="#777777",
                linestyle="--",
                linewidth=1.6,
                label="original 12 x 12 cm" if index == 0 else None,
                zorder=1,
            )
        )
        axis.add_patch(
            Rectangle(
                (active[0], active[2]),
                active[1] - active[0],
                active[3] - active[2],
                facecolor=colors[can_name],
                edgecolor=colors[can_name],
                alpha=0.12,
                linewidth=2.2,
                label=f"{labels[can_name]} active 6 x 12 cm",
                zorder=2,
            )
        )
        xs = value_array(episodes, f"{can_name}_x_m")
        ys = value_array(episodes, f"{can_name}_y_m")
        yaws = value_array(episodes, f"{can_name}_yaw_rad")
        axis.scatter(
            xs,
            ys,
            s=42,
            color=colors[can_name],
            edgecolor="white",
            linewidth=0.7,
            label=f"{labels[can_name]} success init",
            zorder=4,
        )
        add_yaw_arrows(axis, xs, ys, yaws, color=colors[can_name], length=0.025)

    nominal_x, nominal_y = ROBOT_NOMINAL_XY_M
    axis.add_patch(
        Rectangle(
            (nominal_x - ROBOT_HOME_XY_HALF_RANGE_M, nominal_y - ROBOT_HOME_XY_HALF_RANGE_M),
            2.0 * ROBOT_HOME_XY_HALF_RANGE_M,
            2.0 * ROBOT_HOME_XY_HALF_RANGE_M,
            facecolor="#009e73",
            edgecolor="#006b4f",
            alpha=0.13,
            linewidth=2.0,
            label="episode HOME allowed +/-2 cm",
            zorder=2,
        )
    )
    home_x = value_array(episodes, "robot_home_x_m")
    home_y = value_array(episodes, "robot_home_y_m")
    home_yaw = value_array(episodes, "robot_home_yaw_rad")
    axis.scatter(home_x, home_y, marker="s", s=55, color="#009e73", label="episode HOME", zorder=5)
    add_yaw_arrows(axis, home_x, home_y, home_yaw, color="#006b4f", length=0.06)

    spawn_x = value_array(episodes, "robot_spawn_x_m")
    spawn_y = value_array(episodes, "robot_spawn_y_m")
    spawn_yaw = value_array(episodes, "robot_spawn_yaw_rad")
    if task_id == "000002":
        axis.scatter(
            spawn_x,
            spawn_y,
            marker="x",
            s=70,
            linewidth=2.0,
            color="#cc79a7",
            label="route spawn before HOME normalize",
            zorder=5,
        )
        add_yaw_arrows(axis, spawn_x, spawn_y, spawn_yaw, color="#9a4e7c", length=0.06)
        for x0, y0, x1, y1 in zip(spawn_x, spawn_y, home_x, home_y):
            axis.plot((x0, x1), (y0, y1), color="#cc79a7", alpha=0.34, linewidth=1.0)
    else:
        axis.scatter(
            spawn_x,
            spawn_y,
            marker="x",
            s=58,
            color="#009e73",
            label="robot spawn (= episode HOME)",
            zorder=5,
        )

    axis.scatter(
        [nominal_x],
        [nominal_y],
        marker="*",
        s=120,
        color="#111111",
        label="authored nominal robot pose",
        zorder=6,
    )
    axis.set_xlabel("world X [m] (robot side is toward larger X)")
    axis.set_ylabel("world Y [m]")
    axis.set_title(f"Task {task_id}: successful initial poses (top-down)")
    axis.grid(True, color="#dddddd", linewidth=0.6)
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=8)


def plot_robot_home(axis: plt.Axes, episodes: list[EpisodePose]) -> None:
    dx_cm = 100.0 * (value_array(episodes, "robot_home_x_m") - ROBOT_NOMINAL_XY_M[0])
    dy_cm = 100.0 * (value_array(episodes, "robot_home_y_m") - ROBOT_NOMINAL_XY_M[1])
    axis.add_patch(Rectangle((-2.0, -2.0), 4.0, 4.0, color="#009e73", alpha=0.12))
    axis.scatter(dx_cm, dy_cm, s=48, color="#009e73", edgecolor="white", linewidth=0.7)
    for index, (x, y) in enumerate(zip(dx_cm, dy_cm)):
        axis.annotate(str(index), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
    axis.axhline(0.0, color="#777777", linewidth=0.7)
    axis.axvline(0.0, color="#777777", linewidth=0.7)
    axis.set_xlim(-2.3, 2.3)
    axis.set_ylim(-2.3, 2.3)
    axis.set_aspect("equal")
    axis.grid(True, color="#dddddd", linewidth=0.6)
    axis.set_xlabel("HOME delta X [cm]")
    axis.set_ylabel("HOME delta Y [cm]")
    axis.set_title("Episode HOME inside reviewed +/-2 cm box")


def plot_yaw(axis: plt.Axes, episodes: list[EpisodePose]) -> None:
    yaw_deg = np.asarray(
        [
            math.degrees(wrap_angle(value - ROBOT_NOMINAL_YAW_RAD))
            for value in value_array(episodes, "robot_home_yaw_rad")
        ]
    )
    indices = np.arange(len(episodes))
    axis.axhspan(-1.0, 1.0, color="#56b4e9", alpha=0.12, label="reviewed +/-1 deg")
    axis.scatter(indices, yaw_deg, color="#0072b2", s=40)
    axis.axhline(0.0, color="#777777", linewidth=0.7)
    axis.set_ylim(-1.15, 1.15)
    axis.set_xlabel("success episode index")
    axis.set_ylabel("HOME delta yaw [deg]")
    axis.set_title("Robot HOME rotation")
    axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    axis.legend(fontsize=8)


def plot_task_axis(axis: plt.Axes, episodes: list[EpisodePose]) -> None:
    task_id = episodes[0].task_id
    indices = np.arange(len(episodes))
    if task_id == "000001":
        values = 100.0 * value_array(episodes, "lift_spawn_m")
        axis.axhspan(100.0 * LIFT_RANGE_M[0], 100.0 * LIFT_RANGE_M[1], color="#e69f00", alpha=0.12)
        axis.scatter(indices, values, s=40, color="#e69f00")
        axis.set_ylabel("initial lift [cm]")
        axis.set_ylim(-29.0, 1.0)
        axis.set_title("Task 000001 lift init in [-28, 0] cm")
    else:
        values = 100.0 * value_array(episodes, "base_route_spawn_offset_m")
        axis.axhspan(100.0 * ROUTE_RANGE_M[0], 100.0 * ROUTE_RANGE_M[1], color="#cc79a7", alpha=0.12)
        axis.scatter(indices, values, s=40, color="#cc79a7")
        axis.set_ylabel("route spawn offset +Y [cm]")
        axis.set_ylim(-2.0, 100.0 * ROUTE_RANGE_M[1] + 2.0)
        axis.set_title("Task 000002 route spawn relative to episode HOME")
    axis.set_xlabel("success episode index")
    axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)


def save_plot(episodes: list[EpisodePose], output_path: Path) -> None:
    figure = plt.figure(figsize=(15.5, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(2.0, 1.0, 1.0))
    top_down = figure.add_subplot(grid[:, 0])
    home = figure.add_subplot(grid[0, 1])
    yaw = figure.add_subplot(grid[0, 2])
    task_axis = figure.add_subplot(grid[1, 1:])
    plot_top_down(top_down, episodes)
    plot_robot_home(home, episodes)
    plot_yaw(yaw, episodes)
    plot_task_axis(task_axis, episodes)
    figure.suptitle(
        f"Coffee Transport {episodes[0].task_id} | {len(episodes)} successful trajectories",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=180, facecolor="white")
    plt.close(figure)


def row_for_episode(episode: EpisodePose) -> dict[str, Any]:
    values = dict(episode.values)
    values.update(
        {
            "episode_id": episode.episode_id,
            "task_id": episode.task_id,
            "generation_seed": episode.generation_seed,
            "robot_home_delta_x_cm": 100.0
            * (values["robot_home_x_m"] - ROBOT_NOMINAL_XY_M[0]),
            "robot_home_delta_y_cm": 100.0
            * (values["robot_home_y_m"] - ROBOT_NOMINAL_XY_M[1]),
            "robot_home_delta_yaw_deg": math.degrees(
                wrap_angle(values["robot_home_yaw_rad"] - ROBOT_NOMINAL_YAW_RAD)
            ),
        }
    )
    return values


def within(value: float, lower: float, upper: float, tolerance: float = 1.0e-6) -> bool:
    return lower - tolerance <= value <= upper + tolerance


def build_summary(episodes: list[EpisodePose], inputs: list[Path]) -> dict[str, Any]:
    rows = [row_for_episode(episode) for episode in episodes]
    violations: list[str] = []
    for row in rows:
        label = row["episode_id"]
        for can_name in CAN_NAMES:
            x_min, x_max, y_min, y_max = ACTIVE_RANGES_M[can_name]
            if not within(row[f"{can_name}_x_m"], x_min, x_max):
                violations.append(f"{label}: {can_name} X outside active region")
            if not within(row[f"{can_name}_y_m"], y_min, y_max):
                violations.append(f"{label}: {can_name} Y outside active region")
        if not within(row["robot_home_delta_x_cm"], -2.0, 2.0):
            violations.append(f"{label}: robot HOME delta X outside +/-2 cm")
        if not within(row["robot_home_delta_y_cm"], -2.0, 2.0):
            violations.append(f"{label}: robot HOME delta Y outside +/-2 cm")
        if not within(row["robot_home_delta_yaw_deg"], -1.0, 1.0):
            violations.append(f"{label}: robot HOME delta yaw outside +/-1 deg")
        if episodes[0].task_id == "000001" and not within(row["lift_spawn_m"], *LIFT_RANGE_M):
            violations.append(f"{label}: lift init outside [-0.28, 0] m")
        if episodes[0].task_id == "000002" and not within(
            row["base_route_spawn_offset_m"], *ROUTE_RANGE_M
        ):
            violations.append(f"{label}: route spawn offset outside reviewed route")

    def stats(key: str) -> dict[str, float]:
        values = np.asarray([float(row[key]) for row in rows])
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    def yaw_stats_deg(key: str, reference_rad: float = 0.0) -> dict[str, float]:
        values = np.asarray(
            [math.degrees(wrap_angle(float(row[key]) - reference_rad)) for row in rows]
        )
        return {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    return {
        "task_id": episodes[0].task_id,
        "success_count": len(episodes),
        "inputs": [str(path.resolve()) for path in inputs],
        "reviewed_ranges": {
            "can_original_full_12x12_m": {
                name: list(original_range(active)) for name, active in ACTIVE_RANGES_M.items()
            },
            "can_active_robot_side_6x12_m": {
                name: list(active) for name, active in ACTIVE_RANGES_M.items()
            },
            "robot_home_xy_delta_m": [-0.02, 0.02],
            "robot_home_yaw_delta_deg": [-1.0, 1.0],
            "task000001_lift_init_m": list(LIFT_RANGE_M),
            "task000002_route_spawn_offset_m": list(ROUTE_RANGE_M),
        },
        "observed": {
            "robot_spawn_x_m": stats("robot_spawn_x_m"),
            "robot_spawn_y_m": stats("robot_spawn_y_m"),
            "robot_spawn_delta_yaw_deg": yaw_stats_deg(
                "robot_spawn_yaw_rad", ROBOT_NOMINAL_YAW_RAD
            ),
            "robot_home_x_m": stats("robot_home_x_m"),
            "robot_home_y_m": stats("robot_home_y_m"),
            "robot_home_delta_x_cm": stats("robot_home_delta_x_cm"),
            "robot_home_delta_y_cm": stats("robot_home_delta_y_cm"),
            "robot_home_delta_yaw_deg": stats("robot_home_delta_yaw_deg"),
            "lift_spawn_m": stats("lift_spawn_m"),
            "base_route_spawn_offset_m": stats("base_route_spawn_offset_m"),
            **{
                f"{name}_{axis}_m": stats(f"{name}_{axis}_m")
                for name in CAN_NAMES
                for axis in ("x", "y")
            },
            **{
                f"{name}_yaw_deg": yaw_stats_deg(f"{name}_yaw_rad")
                for name in CAN_NAMES
            },
        },
        "range_check_passed": not violations,
        "range_violations": violations,
        "notes": [
            "Can yaw is the recorded rigid-body yaw; appearance-only label yaw is not stored in these episode attributes.",
            "Task000002 robot_spawn is the randomized route start; robot_home is the episode-local HOME used after normalization.",
        ],
    }


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    preferred = [
        "episode_id",
        "task_id",
        "generation_seed",
        "coffee_can_right_x_m",
        "coffee_can_right_y_m",
        "coffee_can_right_yaw_rad",
        "coffee_can_left_x_m",
        "coffee_can_left_y_m",
        "coffee_can_left_yaw_rad",
        "robot_spawn_x_m",
        "robot_spawn_y_m",
        "robot_spawn_yaw_rad",
        "robot_home_x_m",
        "robot_home_y_m",
        "robot_home_yaw_rad",
        "robot_home_delta_x_cm",
        "robot_home_delta_y_cm",
        "robot_home_delta_yaw_deg",
        "lift_spawn_m",
        "base_route_spawn_offset_m",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=preferred)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    inputs = [path.resolve() for path in args.input]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    episodes = load_successes(inputs, args.task_id)
    task_id = episodes[0].task_id
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [row_for_episode(episode) for episode in episodes]
    summary = build_summary(episodes, inputs)
    plot_path = args.output_dir / f"task_{task_id}_success_initial_poses.png"
    csv_path = args.output_dir / f"task_{task_id}_success_initial_poses.csv"
    json_path = args.output_dir / f"task_{task_id}_success_initial_poses_summary.json"
    save_plot(episodes, plot_path)
    write_csv(rows, csv_path)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "task_id": task_id,
                "success_count": len(episodes),
                "range_check_passed": summary["range_check_passed"],
                "plot": str(plot_path),
                "csv": str(csv_path),
                "summary": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
