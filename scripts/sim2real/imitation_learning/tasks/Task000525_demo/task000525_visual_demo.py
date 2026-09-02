#!/usr/bin/env python3
"""Run a looping 16-environment Task000525 visual-augmentation demo.

The demo reuses exact recorded scene states from the successful Task000525
trajectory HDF5.  It does not generate a dataset and it does not rely on
contact-sensitive action replay, so the robot and coffee cans follow the
already validated trajectories exactly.  Every environment loops complete
all-subtask episodes independently and receives fresh per-environment visual
randomization whenever its own episode ends.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
import traceback
from typing import Any

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SOURCE_HDF5 = (
    REPO_ROOT
    / "datasets"
    / "task_000525_trajectory_ccw_rootstable_success_50_v2.hdf5"
)
DEFAULT_TASK = "Cyclo-Real-Showroom-Task000525-FFW-SG2-v0"
DEMO_ENV_SPACING = 8.0
DEFAULT_ZOOM_ENV_IDS = (0, 5, 10)
HIDDEN_WALL_SUFFIXES = (
    "/RobotisShowroom/robotis_showroom/ShowroomShell/LeftWall_01",
    "/RobotisShowroom/robotis_showroom/ShowroomShell/BackWall_01",
    "/RobotisShowroom/robotis_showroom/ShowroomShell/WallBackground",
)



parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-file", type=Path, default=DEFAULT_SOURCE_HDF5)
parser.add_argument("--task", default=DEFAULT_TASK)
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=20260902)
parser.add_argument(
    "--speed",
    type=int,
    default=1,
    help="Wall-clock source playback multiplier (default: 1).",
)
parser.add_argument(
    "--playback-fps",
    type=float,
    default=15.0,
    help=(
        "Recorded source FPS and maximum GUI refresh rate. Playback remains wall-clock "
        "correct when rendering is slower (default: 15)."
    ),
)
parser.add_argument(
    "--progress-interval",
    type=float,
    default=2.0,
    help="Seconds between terminal playback reports; 0 disables them (default: 2).",
)
parser.add_argument(
    "--sequential",
    action="store_true",
    help="Cycle through source episodes in order instead of sampling a unique random batch.",
)
parser.add_argument(
    "--no-throttle",
    action="store_true",
    help="Render as fast as possible instead of limiting playback to --playback-fps.",
)
parser.add_argument(
    "--zoom-envs",
    type=int,
    nargs=3,
    default=DEFAULT_ZOOM_ENV_IDS,
    metavar=("ENV_A", "ENV_B", "ENV_C"),
    help="Three environment IDs shown in the right-side close-up panel.",
)
parser.add_argument(
    "--no-zoom-panel", action="store_true", help="Disable the three GUI close-up views."
)
parser.add_argument(
    "--no-hud",
    action="store_true",
    help="Do not create the small Task000525 status window in GUI mode.",
)
parser.add_argument(
    "--max-render-frames",
    type=int,
    default=0,
    help="Stop after this many rendered frames; 0 repeats until the window closes.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# This presentation intentionally uses only Isaac Sim's third-person viewport.
# The robot-mounted policy camera sensors are removed from the environment below.
app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app


import gymnasium as gym
import h5py
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg

import cyclo_lab  # noqa: F401  Ensures that the Task000525 Gym ID is registered.

from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization import (
    appearance_events,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.appearance_events import (
    randomize_coffee_can_visual_yaw,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.profiles import (
    TASK000525_VISUAL_REPLAY_AUGMENTATION,
    validate_task000525_randomization_cfg,
)


class DemoError(RuntimeError):
    """Raised when the demo cannot preserve its recorded-state contract."""


@dataclass(frozen=True)
class EpisodeReplay:
    """Compact CPU copy of one exact scene-state trajectory."""

    name: str
    frame_count: int
    initial: dict[str, np.ndarray]
    states: dict[str, np.ndarray]


@dataclass(frozen=True)
class GpuEpisodeReplay:
    """One complete all-subtask episode resident on the simulation GPU."""

    name: str
    frame_count: int
    state_tensors: dict[str, torch.Tensor]


@dataclass
class EnvPlayback:
    """Independent episode cursor and loop count for one environment."""

    episode_index: int
    frame_index: int = 0
    completed_episodes: int = 0


class DemoHud:
    """Small floating status window for presentation use."""

    def __init__(self) -> None:
        import omni.ui as ui

        self._window = ui.Window("Task000525 Visual Augmentation", width=650, height=190)
        with self._window.frame:
            with ui.VStack(spacing=4):
                ui.Label(
                    f"TASK 000525  |  {args.num_envs}-ENV FULL-EPISODE REPLAY  |  {args.speed}x"
                )
                self._status = ui.Label("")
                self._episodes_a = ui.Label("")
                self._episodes_b = ui.Label("")
                ui.Label("RESET AUG: LOCAL LIGHT  WALL  COFFEE-LABEL YAW")

    def update(
        self,
        render_frame: int,
        viewport_fps: float,
        playbacks: list[EnvPlayback],
        replays: tuple[GpuEpisodeReplay, ...],
    ) -> None:
        source_fps = args.playback_fps * args.speed
        self._status.text = (
            f"render {render_frame + 1} | viewport {viewport_fps:.1f} FPS | "
            f"source {source_fps:g} FPS ({args.speed}x wall-clock)"
        )
        pairs = [
            (
                f"e{env_id}:{replays[state.episode_index].name.removeprefix('demo_')} "
                f"{state.frame_index}/{replays[state.episode_index].frame_count} "
                f"loop{state.completed_episodes}"
            )
            for env_id, state in enumerate(playbacks)
        ]
        split = (len(pairs) + 1) // 2
        self._episodes_a.text = "  ".join(pairs[:split])
        self._episodes_b.text = "  ".join(pairs[split:])


class DemoZoomPanel:
    """Right-side panel with three robot-following third-person viewports."""

    _EYE_OFFSET = np.asarray((2.6, 2.9, 2.2), dtype=np.float64)
    _TARGET_OFFSET = np.asarray((0.0, 0.0, 0.85), dtype=np.float64)

    def __init__(self, env: Any, env_ids: tuple[int, int, int]) -> None:
        import omni.ui as ui
        from omni.kit.widget.viewport import ViewportWidget
        from pxr import Gf, Sdf, UsdGeom

        self._env_ids = env_ids
        self._gf = Gf
        self._widgets: list[Any] = []
        self._camera_xforms: list[Any] = []

        stage = env.scene.stage
        camera_root = "/World/Task000525DemoZoomCameras"
        UsdGeom.Xform.Define(stage, camera_root)
        camera_paths: list[str] = []
        for env_id in env_ids:
            camera_path = f"{camera_root}/Env{env_id:02d}"
            camera = UsdGeom.Camera.Define(stage, camera_path)
            camera.GetFocalLengthAttr().Set(18.0)
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 100.0))
            camera.GetPrim().CreateAttribute(
                "omni:kit:centerOfInterest",
                Sdf.ValueTypeNames.Vector3d,
                True,
                Sdf.VariabilityUniform,
            ).Set(Gf.Vec3d(0.0, 0.0, -5.0))
            self._camera_xforms.append(
                UsdGeom.Xformable(camera.GetPrim()).AddTransformOp()
            )
            camera_paths.append(camera_path)

        self._window = ui.Window(
            "Task000525 Close-ups",
            width=480,
            height=900,
            padding_x=2,
            padding_y=2,
        )
        with self._window.frame:
            with ui.VStack(spacing=3):
                for env_id, camera_path in zip(env_ids, camera_paths):
                    ui.Label(
                        f"ENV {env_id:02d}  ROBOT CLOSE-UP",
                        height=22,
                        alignment=ui.Alignment.CENTER,
                    )
                    widget = ViewportWidget(
                        camera_path=camera_path,
                        resolution=(480, 270),
                        width=ui.Fraction(1),
                        height=ui.Fraction(1),
                    )
                    widget.expand_viewport = True
                    self._widgets.append(widget)

        main_viewport = ui.Workspace.get_window("Viewport")
        if main_viewport is None:
            raise DemoError("Isaac Sim main Viewport window is unavailable")
        self._window.dock_in(main_viewport, ui.DockPosition.RIGHT, 0.30)
        self.update(env)
        print(
            "TASK000525_DEMO_ZOOM_VIEWS="
            + json.dumps({"env_ids": env_ids, "layout": "right_vertical"}),
            flush=True,
        )

    def update(self, env: Any) -> None:
        root_positions = (
            env.scene["robot"]
            .data.root_pos_w[list(self._env_ids)]
            .detach()
            .cpu()
            .numpy()
        )
        for root_position, transform_op in zip(root_positions, self._camera_xforms):
            eye = root_position.astype(np.float64) + self._EYE_OFFSET
            target = root_position.astype(np.float64) + self._TARGET_OFFSET
            eye_gf = self._gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
            target_gf = self._gf.Vec3d(
                float(target[0]), float(target[1]), float(target[2])
            )
            camera_transform = (
                self._gf.Matrix4d(1.0)
                .SetLookAt(eye_gf, target_gf, self._gf.Vec3d(0.0, 0.0, 1.0))
                .GetInverse()
            )
            transform_op.Set(camera_transform)

    def close(self) -> None:
        for widget in self._widgets:
            widget.destroy()
        self._widgets.clear()
        self._window.destroy()


def natural_episode_key(name: str) -> tuple[str, int]:
    prefix, separator, suffix = name.rpartition("_")
    if separator and suffix.isdigit():
        return prefix, int(suffix)
    return name, -1


def collect_dataset_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    """Return all dataset paths below an HDF5 group."""

    paths: list[str] = []
    for name, value in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            paths.append(path)
        elif isinstance(value, h5py.Group):
            paths.extend(collect_dataset_paths(value, path))
    return paths


def load_replays(path: Path) -> list[EpisodeReplay]:
    """Load only compact scene states, never the large recorded camera arrays."""

    if not path.is_file():
        raise DemoError(
            f"source trajectory HDF5 is missing: {path}\n"
            "The demo needs the source named by the LeRobot dataset provenance, not its videos."
        )

    result: list[EpisodeReplay] = []
    with h5py.File(path, "r") as source:
        if "data" not in source:
            raise DemoError(f"source HDF5 has no /data group: {path}")
        data = source["data"]
        names = sorted(data.keys(), key=natural_episode_key)
        if not names:
            raise DemoError(f"source HDF5 contains no episodes: {path}")

        reference_paths = collect_dataset_paths(data[names[0]]["initial_state"])
        for name in names:
            group = data[name]
            if "actions" not in group or "initial_state" not in group or "states" not in group:
                raise DemoError(f"{group.name} lacks actions, initial_state, or states")
            frame_count = int(group["actions"].shape[0]) - 1
            if frame_count < 2:
                raise DemoError(f"{group.name} has only {frame_count} causal replay frames")
            paths = collect_dataset_paths(group["initial_state"])
            if paths != reference_paths:
                raise DemoError(f"{group.name} scene-state paths differ from the first episode")

            initial: dict[str, np.ndarray] = {}
            states: dict[str, np.ndarray] = {}
            for state_path in reference_paths:
                initial_value = np.asarray(group[f"initial_state/{state_path}"], dtype=np.float32)
                state_value = np.asarray(group[f"states/{state_path}"], dtype=np.float32)
                if initial_value.shape[0] != 1:
                    raise DemoError(
                        f"{group.name}/initial_state/{state_path} must have one environment row"
                    )
                if state_value.shape[0] < frame_count - 1:
                    raise DemoError(
                        f"{group.name}/states/{state_path} has {state_value.shape[0]} rows, "
                        f"expected at least {frame_count - 1}"
                    )
                initial[state_path] = initial_value[0]
                states[state_path] = state_value
            result.append(
                EpisodeReplay(
                    name=name,
                    frame_count=frame_count,
                    initial=initial,
                    states=states,
                )
            )
    return result


def episode_state_sequence(
    episode: EpisodeReplay,
    state_path: str,
) -> np.ndarray:
    """Return every causal pre-step state for one complete all-subtask episode."""

    return np.concatenate(
        (
            episode.initial[state_path][None],
            episode.states[state_path][: episode.frame_count - 1],
        ),
        axis=0,
    )


def upload_replays(
    replays: list[EpisodeReplay],
    device: str,
) -> tuple[GpuEpisodeReplay, ...]:
    """Upload compact state-only episodes once; recorded camera arrays stay on disk."""

    result: list[GpuEpisodeReplay] = []
    for episode in replays:
        tensors = {
            state_path: torch.as_tensor(
                episode_state_sequence(episode, state_path),
                device=device,
            )
            for state_path in episode.initial
        }
        result.append(
            GpuEpisodeReplay(
                name=episode.name,
                frame_count=episode.frame_count,
                state_tensors=tensors,
            )
        )
    return tuple(result)


def initial_playbacks(
    replay_count: int,
    rng: np.random.Generator,
) -> list[EnvPlayback]:
    if args.num_envs > replay_count:
        raise DemoError(
            f"--num-envs {args.num_envs} exceeds {replay_count} available unique trajectories"
        )
    if args.sequential:
        indices = np.arange(args.num_envs, dtype=np.int64)
    else:
        indices = rng.choice(replay_count, size=args.num_envs, replace=False)
    return [EnvPlayback(episode_index=int(index)) for index in indices]


def choose_next_episode(
    current_index: int,
    replay_count: int,
    rng: np.random.Generator,
) -> int:
    if replay_count <= 1:
        return 0
    if args.sequential:
        return (current_index + 1) % replay_count
    sampled = int(rng.integers(0, replay_count - 1))
    return sampled + int(sampled >= current_index)

def advance_playbacks(
    replays: tuple[GpuEpisodeReplay, ...],
    playbacks: list[EnvPlayback],
    source_steps: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Advance every env by wall-clock source steps and report episode crossings."""

    completed_envs: list[int] = []
    reset_records: list[dict[str, Any]] = []
    for env_id, state in enumerate(playbacks):
        remaining = source_steps
        env_completed = False
        while remaining > 0:
            episode = replays[state.episode_index]
            frames_to_end = episode.frame_count - state.frame_index
            if remaining < frames_to_end:
                state.frame_index += remaining
                break
            remaining -= frames_to_end
            previous_name = episode.name
            state.episode_index = choose_next_episode(
                state.episode_index,
                len(replays),
                rng,
            )
            state.frame_index = 0
            state.completed_episodes += 1
            env_completed = True
            reset_records.append(
                {
                    "env": env_id,
                    "completed": previous_name,
                    "next": replays[state.episode_index].name,
                    "loop": state.completed_episodes,
                }
            )
            break
        if env_completed:
            completed_envs.append(env_id)
    return completed_envs, reset_records


def nested_assign(root: dict[str, Any], path: str, value: torch.Tensor) -> None:
    cursor = root
    parts = path.split("/")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def playback_state(
    replays: tuple[GpuEpisodeReplay, ...],
    playbacks: list[EnvPlayback],
    env_indices: list[int] | None = None,
) -> dict[str, Any]:
    if env_indices is None:
        env_indices = list(range(len(playbacks)))
    state: dict[str, Any] = {}
    for state_path in replays[0].state_tensors:
        rows = [
            replays[playbacks[env_id].episode_index]
            .state_tensors[state_path][playbacks[env_id].frame_index]
            for env_id in env_indices
        ]
        nested_assign(state, state_path, torch.stack(rows, dim=0))
    return state


def disable_configured_events(env_cfg: Any) -> None:
    if env_cfg.events is not None:
        for name in tuple(vars(env_cfg.events)):
            setattr(env_cfg.events, name, None)


def disable_camera_observations(env_cfg: Any) -> None:
    """Remove policy-camera observations from this third-person-only demo."""

    policy = getattr(env_cfg.observations, "policy", None)
    if policy is None:
        return
    for name in ("cam_head", "cam_wrist_left", "cam_wrist_right"):
        if hasattr(policy, name):
            setattr(policy, name, None)


def apply_visual_profile(
    env: Any,
    env_ids: torch.Tensor,
    *,
    resample_global_dome: bool,
) -> None:
    """Apply full or partial Task000525 visual-only randomization."""

    profile = TASK000525_VISUAL_REPLAY_AUGMENTATION
    validate_task000525_randomization_cfg(profile)
    if resample_global_dome:
        if hasattr(env, "_task458_dome_sample"):
            delattr(env, "_task458_dome_sample")
    elif not hasattr(env, "_task458_dome_sample"):
        raise DemoError("global dome must be initialized before partial visual resets")

    lighting = profile.lighting
    appearance_events.randomize_dome_and_weak_keys(
        env,
        env_ids,
        lighting.dome_intensity_range,
        lighting.dome_rgb_range,
        lighting.weak_key_intensity_range,
    )
    wall = profile.wall
    appearance_events.randomize_wall_solid_rgb(
        env,
        env_ids,
        wall.mode,
        wall.rgb_range,
        wall.near_white_range,
    )
    # Camera augmentation is intentionally omitted: this presentation removes
    # all policy-camera sensors and renders only the third-person viewport.
    coffee_yaw = profile.coffee_visual_yaw
    randomize_coffee_can_visual_yaw(
        env,
        env_ids,
        coffee_yaw.object_names,
        coffee_yaw.yaw_range_rad,
    )


def hide_demo_walls(env: Any) -> None:
    """Hide only the opposite _01 walls and authored background in every env."""

    from pxr import UsdGeom

    hidden_paths: list[str] = []
    for env_path in env.scene.env_prim_paths:
        for suffix in HIDDEN_WALL_SUFFIXES:
            prim_path = env_path + suffix
            prim = env.scene.stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise DemoError(f"demo wall prim is missing: {prim_path}")
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                raise DemoError(f"demo wall prim is not imageable: {prim_path}")
            imageable.MakeInvisible()
            hidden_paths.append(prim_path)

    print(
        "TASK000525_DEMO_HIDDEN_WALLS="
        + json.dumps(
            {
                "count": len(hidden_paths),
                "per_env": [suffix.rsplit("/", 1)[-1] for suffix in HIDDEN_WALL_SUFFIXES],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def set_overview_camera(env: Any) -> None:
    origins = env.scene.env_origins.detach().cpu().numpy()
    minimum = origins.min(axis=0)
    maximum = origins.max(axis=0)
    center = 0.5 * (minimum + maximum)
    footprint = max(float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1]))
    footprint += float(env.cfg.scene.env_spacing)
    eye = [
        float(center[0] + 0.85 * footprint),
        float(center[1] + 1.00 * footprint),
        float(max(16.0, 0.90 * footprint)),
    ]
    target = [float(center[0] - 0.8), float(center[1]), float(center[2] + 0.7)]
    env.sim.set_camera_view(eye=eye, target=target)
    print(f"TASK000525_DEMO_VIEW={json.dumps({'eye': eye, 'target': target})}", flush=True)


def validate_args() -> None:
    if args.num_envs < 1:
        raise DemoError("--num-envs must be positive")
    if args.playback_fps <= 0.0:
        raise DemoError("--playback-fps must be positive")
    if args.progress_interval < 0.0:
        raise DemoError("--progress-interval must be zero or positive")
    if args.speed < 1:
        raise DemoError("--speed must be a positive integer")
    if args.max_render_frames < 0:
        raise DemoError("--max-render-frames must be zero or positive")
    if not args.headless and not args.no_zoom_panel:
        zoom_env_ids = tuple(args.zoom_envs)
        if len(set(zoom_env_ids)) != 3:
            raise DemoError("--zoom-envs must contain three unique environment IDs")
        if min(zoom_env_ids) < 0 or max(zoom_env_ids) >= args.num_envs:
            raise DemoError(
                f"--zoom-envs {zoom_env_ids} must be inside [0, {args.num_envs - 1}]"
            )


def main() -> None:
    validate_args()
    source_path = args.input_file.resolve()
    print(f"TASK000525_DEMO_LOADING={source_path}", flush=True)
    cpu_replays = load_replays(source_path)
    print(
        f"TASK000525_DEMO_READY source_episodes={len(cpu_replays)} "
        f"num_envs={args.num_envs} speed={args.speed}x",
        flush=True,
    )

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.scene.env_spacing = DEMO_ENV_SPACING
    env_cfg.seed = args.seed
    env_cfg.recorders = None
    env_cfg.terminations = None
    env_cfg.rerender_on_reset = True
    disable_configured_events(env_cfg)
    disable_camera_observations(env_cfg)
    for camera_name in ("cam_head", "cam_wrist_left", "cam_wrist_right"):
        setattr(env_cfg.scene, camera_name, None)
    env = gym.make(args.task, cfg=env_cfg).unwrapped

    zoom_panel: DemoZoomPanel | None = None
    try:
        env.reset()
        hide_demo_walls(env)
        unexpected_cameras = [
            name
            for name in ("cam_head", "cam_wrist_left", "cam_wrist_right")
            if name in env.scene.sensors
        ]
        if unexpected_cameras:
            raise DemoError(
                f"third-person demo still has policy cameras: {unexpected_cameras}"
            )
        print("TASK000525_DEMO_POLICY_CAMERAS=disabled", flush=True)

        set_overview_camera(env)
        all_env_ids = torch.arange(args.num_envs, device=env.device, dtype=torch.long)
        rng = np.random.default_rng(args.seed)
        replays = upload_replays(cpu_replays, env.device)
        playbacks = initial_playbacks(len(replays), rng)
        hud = DemoHud() if not args.headless and not args.no_hud else None
        frame_period = 1.0 / args.playback_fps
        render_frame = 0
        reset_event = 0
        deadline = time.monotonic()
        playback_clock = deadline
        source_frame_accumulator = 0.0
        total_source_steps = 0
        viewport_fps = 0.0
        rate_started = deadline
        rate_render_frames = 0
        progress_started = deadline
        progress_source_steps = 0
        hud_updated = 0.0

        torch.manual_seed(args.seed)
        np.random.seed(args.seed % (2**32 - 1))
        env.scene.reset_to(
            playback_state(replays, playbacks),
            all_env_ids,
            is_relative=True,
        )
        env.sim.forward()
        apply_visual_profile(
            env,
            all_env_ids,
            resample_global_dome=True,
        )
        env.sim.forward()
        # Warm the viewport before starting the clock so shader compilation
        # cannot make a 5x run skip most of its first episode.
        env.sim.render()
        print(
            "TASK000525_DEMO_START="
            + json.dumps(
                {
                    "speed": args.speed,
                    "episodes": [
                        replays[state.episode_index].name for state in playbacks
                    ],
                    "visual_contract": (
                        "third-person only; LeftWall_01, BackWall_01, and WallBackground "
                        "hidden; per-env weak light, remaining wall, and coffee-label yaw "
                        "resample; policy cameras disabled; DomeLight remains shared"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        deadline = time.monotonic()
        playback_clock = deadline
        rate_started = deadline
        progress_started = deadline

        while simulation_app.is_running() and not simulation_app.is_exiting():
            if args.max_render_frames and render_frame >= args.max_render_frames:
                break

            env.scene.reset_to(
                playback_state(replays, playbacks),
                all_env_ids,
                is_relative=True,
            )
            env.sim.forward()
            env.sim.render()

            now = time.monotonic()
            elapsed = max(0.0, now - playback_clock)
            playback_clock = now
            source_frame_accumulator += elapsed * args.playback_fps * args.speed
            source_steps = int(source_frame_accumulator)
            source_frame_accumulator -= source_steps
            total_source_steps += source_steps

            rate_render_frames += 1
            rate_elapsed = now - rate_started
            if rate_elapsed >= 0.5:
                viewport_fps = rate_render_frames / rate_elapsed
                rate_started = now
                rate_render_frames = 0

            if hud is not None and now - hud_updated >= 0.25:
                hud.update(render_frame, viewport_fps, playbacks, replays)
                hud_updated = now

            completed_envs, reset_records = advance_playbacks(
                replays,
                playbacks,
                source_steps,
                rng,
            )

            if completed_envs:
                reset_event += 1
                reset_seed = args.seed + reset_event * 1_000_003
                torch.manual_seed(reset_seed)
                np.random.seed(reset_seed % (2**32 - 1))
                completed_ids = torch.as_tensor(
                    completed_envs,
                    device=env.device,
                    dtype=torch.long,
                )
                env.scene.reset_to(
                    playback_state(replays, playbacks, completed_envs),
                    completed_ids,
                    is_relative=True,
                )
                resample_dome = len(completed_envs) == args.num_envs
                apply_visual_profile(
                    env,
                    completed_ids,
                    resample_global_dome=resample_dome,
                )
                env.sim.forward()
                print(
                    "TASK000525_DEMO_ENV_RESET="
                    + json.dumps(
                        {
                            "seed": reset_seed,
                            "resampled_global_dome": resample_dome,
                            "environments": reset_records,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            progress_elapsed = now - progress_started
            if args.progress_interval and progress_elapsed >= args.progress_interval:
                source_delta = total_source_steps - progress_source_steps
                state = playbacks[0]
                print(
                    "TASK000525_DEMO_PROGRESS="
                    + json.dumps(
                        {
                            "env0_episode": replays[state.episode_index].name,
                            "env0_frame": state.frame_index,
                            "env0_loop": state.completed_episodes,
                            "source_fps_actual": round(source_delta / progress_elapsed, 2),
                            "source_fps_target": args.playback_fps * args.speed,
                            "speed": args.speed,
                            "viewport_fps": round(viewport_fps, 2),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                progress_started = now
                progress_source_steps = total_source_steps

            if not args.no_throttle:
                deadline += frame_period
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
                else:
                    deadline = time.monotonic()
            render_frame += 1
    finally:
        if zoom_panel is not None:
            zoom_panel.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("TASK000525_DEMO_STOPPED=keyboard_interrupt", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
