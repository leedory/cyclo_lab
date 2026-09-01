# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a cyclo_lab teleoperation with cyclo_lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
import contextlib
from collections import defaultdict

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="cyclo_lab teleoperation for cyclo_lab environments.")
parser.add_argument("--robot_type", type=str, default="keyboard", choices=['OMY', 'FFW_SG2'], help="Type of robot to use for teleoperation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment.")

# recorder_parameter
parser.add_argument(
    "--step_hz",
    type=float,
    default=None,
    help="Environment stepping rate in Hz. Uses the task recording rate, or 60 Hz when unspecified.",
)
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos.")
parser.add_argument("--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite.")
parser.add_argument("--flush_steps", type=int, default=30, help="Streaming HDF5 flush interval in environment steps.")
parser.add_argument(
    "--camera_view",
    default="none",
    choices=("none", "operator"),
    help="Optional local operator camera dashboard while recording.",
)
parser.add_argument(
    "--publish_camera_topics",
    action="store_true",
    help="Publish compressed camera topics while recording. HDF5 camera recording does not require this.",
)
extra_state_topic_group = parser.add_mutually_exclusive_group()
extra_state_topic_group.add_argument(
    "--publish_extra_state_topics",
    dest="publish_extra_state_topics",
    action="store_true",
    help="Publish odom/tf in addition to /joint_states while recording. Enabled by default.",
)
extra_state_topic_group.add_argument(
    "--no_publish_extra_state_topics",
    dest="publish_extra_state_topics",
    action="store_false",
    help="Disable odom/tf publishing while recording. /joint_states is always published.",
)
parser.set_defaults(publish_extra_state_topics=True)
parser.add_argument("--profile", action="store_true", help="Print timing statistics for the recording loop.")
parser.add_argument("--profile_interval", type=int, default=120, help="Loop iterations between profile reports.")
parser.add_argument(
    "--render_episode_cameras",
    action="store_true",
    help="After recording, render one labeled three-camera MP4 per saved episode beside the HDF5 dataset.",
)
parser.add_argument(
    "--profile_cuda_sync",
    action="store_true",
    help="Synchronize CUDA around profiled sections for more accurate GPU timing. This adds overhead.",
)
parser.add_argument(
    "--task525_phase_markers",
    action="store_true",
    help=(
        "Enable Task525 continuous-demo markers. In Dijkstra mode G=grasp complete "
        "and automatically returns the carrying arm home before navigation; F/H are optional annotations. "
        "Manual mode additionally uses M=base motion starts and P=place starts."
    ),
)
parser.add_argument(
    "--task525_base_mode",
    choices=("manual", "dijkstra"),
    default="manual",
    help=(
        "Task525 base segment source. 'manual' preserves foot-pad/keyboard control; "
        "'dijkstra' plans and executes the simulated base path after G."
    ),
)
parser.add_argument(
    "--task525_dijkstra_linear_max",
    type=float,
    default=0.10,
    help="Task525 online Dijkstra maximum linear body speed in m/s.",
)
parser.add_argument(
    "--task525_dijkstra_angular_max",
    type=float,
    default=0.25,
    help="Task525 online Dijkstra maximum yaw speed in rad/s.",
)
parser.add_argument(
    "--task525_dijkstra_timeout_s",
    type=float,
    default=90.0,
    help="Task525 online Dijkstra fail-closed timeout in seconds.",
)
parser.add_argument(
    "--keyboard_mobile",
    action="store_true",
    help="Enable W/S, A/D, Q/E, Space base control while preserving external /cmd_vel when idle.",
)
parser.add_argument(
    "--keyboard_linear_speed",
    type=float,
    default=0.10,
    help="Recording keyboard base translation speed in m/s.",
)
parser.add_argument(
    "--keyboard_angular_speed",
    type=float,
    default=0.25,
    help="Recording keyboard base yaw speed in rad/s.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if args_cli.camera_view == "operator":
    args_cli.enable_cameras = True

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import time
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import DatasetExportMode, RecorderTerm, TerminationTermCfg

import cyclo_lab
import os

from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_JOINT_POSITION_LIMITS,
    FFW_SG2_LIFT_POSITION_UPPER,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.home_pose import (
    TASK000525_SAVE_POSE_3_JOINT_POSITIONS,
)

from recorder_manager.recorder_manager import StreamingRecorderManager


TASK525_RIGHT_GRIPPER_ACTION_INDEX = FFW_SG2_ACTION_JOINT_NAMES.index("gripper_r_joint1")
TASK525_HEAD_PITCH_ACTION_INDEX = FFW_SG2_ACTION_JOINT_NAMES.index("head_joint1")
TASK525_HEAD_YAW_ACTION_INDEX = FFW_SG2_ACTION_JOINT_NAMES.index("head_joint2")
TASK525_LIFT_ACTION_INDEX = FFW_SG2_ACTION_JOINT_NAMES.index("lift_joint")
TASK525_HEAD_PITCH_DOWN_MAX_RAD = FFW_SG2_JOINT_POSITION_LIMITS["head_joint1"][1]

class RateLimiter:
    """Keep a fixed loop schedule while allowing short overruns to catch up."""

    def __init__(self, hz: float):
        if hz <= 0.0:
            raise ValueError(f"Rate must be positive, got {hz} Hz.")
        self.hz = hz
        self.sleep_duration = 1.0 / hz
        self.next_wakeup_time = time.perf_counter() + self.sleep_duration

    def sleep(self):
        """Sleep until the next deadline without discarding a one-cycle overrun."""
        sleep_time = self.next_wakeup_time - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)

        now = time.perf_counter()
        self.next_wakeup_time += self.sleep_duration
        if now - self.next_wakeup_time > self.sleep_duration:
            self.next_wakeup_time = now + self.sleep_duration


class LoopProfiler:
    """Small timing profiler for the demonstration recorder."""

    def __init__(self, enabled: bool, interval: int, cuda_sync: bool = False):
        self.enabled = enabled
        self.interval = max(1, int(interval))
        self.cuda_sync = bool(cuda_sync)
        self.loop_count = 0
        self._window_start = time.perf_counter()
        self._stats = defaultdict(lambda: {"count": 0, "total": 0.0, "max": 0.0})
        self._installed_hooks = []

    def _sync(self):
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def time(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            stat = self._stats[name]
            stat["count"] += 1
            stat["total"] += elapsed
            stat["max"] = max(stat["max"], elapsed)

    def tick(self):
        if not self.enabled:
            return
        self.loop_count += 1
        if self.loop_count % self.interval != 0:
            return

        now = time.perf_counter()
        window_elapsed = max(now - self._window_start, 1e-9)
        measured_hz = self.interval / window_elapsed
        loop_total = self._stats.get("loop_total", {}).get("total", 0.0)

        print(f"\n[PROFILE] last {self.interval} loops: wall_hz={measured_hz:.2f}")
        rows = sorted(
            self._stats.items(),
            key=lambda item: item[1]["total"],
            reverse=True,
        )
        for name, stat in rows:
            if stat["count"] == 0:
                continue
            percent = (stat["total"] / loop_total * 100.0) if loop_total > 0.0 and name != "loop_total" else 0.0
            print(
                f"[PROFILE] {name:36s} "
                f"mean={stat['total'] / stat['count'] * 1000.0:8.2f}ms "
                f"max={stat['max'] * 1000.0:8.2f}ms "
                f"total={stat['total']:7.3f}s "
                f"n={stat['count']:5d} "
                f"{percent:5.1f}%"
            )
        self._stats.clear()
        self._window_start = now

    def register_hook(self, obj, method_name: str, original) -> None:
        """Track an instance-method hook so object lifetimes are restored before shutdown."""
        self._installed_hooks.append((obj, method_name, original))

    def remove_hooks(self) -> None:
        """Restore methods wrapped for profiling and release bound-method references."""
        for obj, method_name, original in reversed(self._installed_hooks):
            try:
                setattr(obj, method_name, original)
            except Exception as exc:
                print(f"[PROFILE] Failed to remove hook for {method_name}: {exc}")
        self._installed_hooks.clear()


def install_profile_hook(obj, method_name: str, label: str, profiler: LoopProfiler):
    """Wrap an instance method with a profiler timer."""
    if not profiler.enabled or not hasattr(obj, method_name):
        return
    original = getattr(obj, method_name)
    if getattr(original, "_cyclo_profile_wrapped", False):
        return

    def wrapped(*args, **kwargs):
        with profiler.time(label):
            return original(*args, **kwargs)

    wrapped._cyclo_profile_wrapped = True
    try:
        setattr(obj, method_name, wrapped)
        profiler.register_hook(obj, method_name, original)
    except Exception as exc:
        print(f"[PROFILE] Failed to install hook for {label}: {exc}")


def install_env_profile_hooks(env, profiler: LoopProfiler):
    """Install coarse Isaac Lab manager timing hooks."""
    install_profile_hook(env.observation_manager, "compute", "isaaclab_observation_compute", profiler)
    install_profile_hook(env.termination_manager, "compute", "isaaclab_termination_compute", profiler)
    install_profile_hook(env.reward_manager, "compute", "isaaclab_reward_compute", profiler)
    install_profile_hook(env.command_manager, "compute", "isaaclab_command_compute", profiler)
    install_profile_hook(env.action_manager, "process_action", "isaaclab_action_process", profiler)
    install_profile_hook(env.action_manager, "apply_action", "isaaclab_action_apply", profiler)
    install_profile_hook(env.scene, "write_data_to_sim", "isaaclab_scene_write", profiler)
    install_profile_hook(env.scene, "update", "isaaclab_scene_update", profiler)
    install_profile_hook(env.sim, "step", "isaaclab_sim_step", profiler)
    install_profile_hook(env.sim, "render", "isaaclab_sim_render", profiler)


def install_recorder_term_profile_hooks(recorder_manager, profiler: LoopProfiler):
    """Install timing hooks for each active recorder term and phase."""
    for term_name, term in getattr(recorder_manager, "_terms", {}).items():
        for phase in ("pre_step", "post_step", "post_physics_decimation_step"):
            method_name = f"record_{phase}"
            if getattr(type(term), method_name) is getattr(RecorderTerm, method_name):
                continue
            install_profile_hook(
                term,
                method_name,
                f"recorder_term_{term_name}_{phase}",
                profiler,
            )


def release_camera_sensors_before_close(env) -> None:
    """Detach camera annotators before Isaac Sim starts clearing simulation callbacks."""
    for sensor in getattr(env.scene, "sensors", {}).values():
        registry = getattr(sensor, "_rep_registry", None)
        if registry is None:
            continue
        try:
            sensor.__del__()
        except Exception as exc:
            print(f"[WARN] Failed to detach camera sensor during shutdown: {exc}")
        finally:
            registry.clear()


def enable_operator_camera_view(env_cfg) -> None:
    """Enable the task's operator dashboard cameras before environment creation."""
    enable_operator_cameras = getattr(env_cfg, "enable_operator_preview_cameras", None)
    if enable_operator_cameras is not None:
        enable_operator_cameras()
    if not getattr(env_cfg, "operator_camera_rows", None):
        raise ValueError(f"Task {args_cli.task} does not support --camera_view operator.")


def make_operator_view(env_cfg, env):
    """Create the same multi-camera dashboard used by sim2real bringup."""
    if args_cli.camera_view != "operator":
        return None

    from cyclo_lab.runtime.viewers import CameraDashboard

    rows = getattr(env_cfg, "operator_camera_rows")
    camera_names = {camera_name for row in rows for camera_name, _label in row}
    missing_cameras = sorted(camera_names.difference(env.scene.sensors))
    if missing_cameras:
        raise ValueError(f"Operator dashboard sensors are unavailable: {missing_cameras}")

    dashboard = CameraDashboard(
        env,
        rows=rows,
        panel_rotations=dict(getattr(env_cfg, "operator_camera_rotations", ())),
        window_title=getattr(env_cfg, "operator_camera_title", "Camera Dashboard"),
        window_size=getattr(env_cfg, "operator_camera_window_size", 1800),
        window_position=(20, 20),
    )
    print(f"[Camera Preview] operator dashboard: {dashboard.width}x{dashboard.height}")
    return dashboard



def _recording_metadata(env_cfg, task_name: str, target_step_hz: float) -> dict:
    """Return explicit HDF5 contract metadata for seed-demo recording."""
    metadata = {
        "schema_version": "cyclo_lab_hdf5_v1",
        "control_hz": float(target_step_hz),
        "camera_hz": float(getattr(env_cfg, "camera_hz", target_step_hz)),
        "observation_semantics": "pre_step",
        "obs_last_action_semantics": "previous_step_action",
        "scene_state_semantics": "post_step",
        "task_env_name": task_name,
        "task_id": str(getattr(env_cfg, "task_id", "")),
        "task_instruction": str(getattr(env_cfg, "task_instruction", "")),
        "target_object_name": str(getattr(env_cfg, "target_object", "")),
        "target_side": str(getattr(env_cfg, "target_side", "")),
        "success_criterion_id": "manual_operator_acceptance",
        "dataset_origin": "isaaclab_hdf5_seed",
    }

    if args_cli.robot_type == "FFW_SG2":
        from cyclo_lab.robot_specs.ffw.sg2 import hdf5_contract_metadata

        metadata.update(hdf5_contract_metadata(env_cfg.actions))

    return metadata


def _set_dataset_metadata(recorder_manager, metadata: dict) -> None:
    handler = getattr(recorder_manager, "_dataset_file_handler", None)
    if handler is None:
        return
    set_metadata = getattr(handler, "set_dataset_metadata", None)
    if set_metadata is None:
        return
    set_metadata(metadata)


def _set_episode_metadata(recorder_manager, metadata: dict) -> None:
    for ep in getattr(recorder_manager, "_episodes", {}).values():
        if ep is not None and not ep.is_empty():
            ep.metadata = dict(metadata)


def _record_step_metadata(
    recorder_manager,
    step_index: int,
    target_step_hz: float,
    device: str,
    *,
    task525_phase: int | None = None,
) -> None:
    timestamp_s = step_index / target_step_hz
    recorder_manager.add_to_episodes(
        "obs/timestamp_s",
        torch.tensor([[timestamp_s]], dtype=torch.float64, device=device),
    )
    recorder_manager.add_to_episodes(
        "obs/step_index",
        torch.tensor([[step_index]], dtype=torch.int64, device=device),
    )
    if task525_phase is not None:
        recorder_manager.add_to_episodes(
            "obs/task525_demo_phase",
            torch.tensor([[task525_phase]], dtype=torch.int64, device=device),
        )



def _sg2_wheel_speed_norm(env) -> float:
    """Return the largest SG2 drive-wheel speed norm across environments."""

    from cyclo_lab.robot_specs.ffw.sg2 import SG2_SWERVE_WHEEL_JOINTS

    robot = env.scene["robot"]
    joint_ids, joint_names = robot.find_joints(
        list(SG2_SWERVE_WHEEL_JOINTS), preserve_order=True
    )
    if len(joint_ids) != len(SG2_SWERVE_WHEEL_JOINTS):
        raise RuntimeError(f"Failed to resolve SG2 wheel joints: {joint_names}")
    return float(
        torch.linalg.vector_norm(robot.data.joint_vel[:, joint_ids], dim=1)
        .amax()
        .item()
    )

def main():
    """Running cyclo_lab teleoperation with cyclo_lab manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.init_action_cfg("record")
    env_cfg.seed = args_cli.seed
    task_name = args_cli.task
    if args_cli.camera_view == "operator":
        enable_operator_camera_view(env_cfg)

    # modify configuration
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    if not hasattr(env_cfg.terminations, "success"):
        setattr(env_cfg.terminations, "success", None)
    env_cfg.terminations.success = TerminationTermCfg(func=lambda env: torch.zeros(1, dtype=torch.bool, device=env.device))
    # Do not save while stepping; only save explicitly on success (key 'N')
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE

    # create environment
    env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    task525_markers_enabled = bool(args_cli.task525_phase_markers)
    if task525_markers_enabled:
        if args_cli.robot_type != "FFW_SG2" or "Task000525" not in task_name:
            raise ValueError("--task525_phase_markers is only valid for the FFW_SG2 Task000525 environment.")
    if args_cli.task525_base_mode == "dijkstra" and not task525_markers_enabled:
        raise ValueError("--task525_base_mode=dijkstra requires --task525_phase_markers.")
    if args_cli.task525_base_mode == "dijkstra":
        # The upstream Dijkstra wrapper is an optional Isaac Sim extension and
        # is not loaded by the minimal headless experience by default.
        import omni.kit.app

        extension_manager = omni.kit.app.get_app().get_extension_manager()
        if not extension_manager.is_extension_enabled("isaacsim.replicator.mobility_gen"):
            extension_manager.set_extension_enabled_immediate(
                "isaacsim.replicator.mobility_gen", True
            )

    del env.recorder_manager
    # Ensure dataset file handler is created, but keep stepping in no-save mode
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
    env.recorder_manager.flush_steps = args_cli.flush_steps
    env.recorder_manager.compression = 'lzf'
    env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    env.recorder_manager.recording_enabled = False
    profiler = LoopProfiler(
        enabled=args_cli.profile,
        interval=args_cli.profile_interval,
        cuda_sync=args_cli.profile_cuda_sync,
    )
    env.recorder_manager.profiler = profiler
    install_env_profile_hooks(env, profiler)
    install_recorder_term_profile_hooks(env.recorder_manager, profiler)

    # create controller
    if args_cli.robot_type == "OMY":
        from cyclo_lab.runtime.sdk.omy import OMYSdk
        teleop_interface = OMYSdk(
            env,
            mode='record',
            camera_publish_hz=None if args_cli.publish_camera_topics else 0.0,
        )
    elif args_cli.robot_type == "FFW_SG2":
        from cyclo_lab.runtime.sdk.ffw_sg2 import FFWSG2Sdk
        teleop_interface = FFWSG2Sdk(
            env,
            mode='record',
            camera_publish_hz=None if args_cli.publish_camera_topics else 0.0,
            publish_odometry_tf=args_cli.publish_extra_state_topics,
            keyboard_mobile=args_cli.keyboard_mobile,
            keyboard_linear_speed=args_cli.keyboard_linear_speed,
            keyboard_angular_speed=args_cli.keyboard_angular_speed,
            # Task525 collection intentionally uses the physical A3 in
            # right-only mode. Keep the original left/right topic mapping,
            # but do not grant the unused left leader ownership at B.
            # This holds the already-settled reset pose without an offset or
            # per-joint action rewrite.
            active_trajectory_groups=(
                ("right_arm", "head", "lift")
                if task525_markers_enabled
                else None
            ),
        )
    else:
        raise ValueError(
            f"Invalid device interface '{args_cli.robot_type}'. Supported: 'OMY', 'FFW_SG2'."
        )

    start_record_state = False
    recorded_step_index = 0
    # The three legacy/SDG boundaries retain their upstream names. ``grasp``
    # and ``release`` make the five Mimic source subtasks explicit without
    # changing the locomanipulation-SDG ``lift < navigate < place`` contract.
    task525_markers = {
        "grasp_step": None,
        "lift_step": None,
        "navigate_step": None,
        "place_step": None,
        "release_step": None,
    }
    task525_auto_navigation = None
    should_start_task525_auto_navigation = False
    task525_auto_navigation_failure_reported = False
    task525_home_arm_action = None
    task525_place_activation_generation = None
    task525_reset_joint_hold_target = None
    task525_reset_lift_hold_target = None

    def task525_save_pose_target() -> torch.Tensor:
        """Return the A3-aligned Task525 init/home pose in public 19D order."""

        values = [
            TASK000525_SAVE_POSE_3_JOINT_POSITIONS[name]
            for name in FFW_SG2_ACTION_JOINT_NAMES
        ]
        return torch.tensor(values, dtype=torch.float32, device=env.device).unsqueeze(0).expand(
            env.num_envs, -1
        ).clone()

    def task525_reset_lift_target() -> torch.Tensor:
        """Return the known upper/reset lift target, independent of A3 cache."""

        return torch.full(
            (env.num_envs,),
            FFW_SG2_LIFT_POSITION_UPPER,
            dtype=torch.float32,
            device=env.device,
        )

    def clear_episode_cache(context: str):
        try:
            env.recorder_manager._clear_episode_cache()
        except Exception as e:
            print(f"Warning: Failed to clear {context} episode cache: {e}")
            print(f"Exception details: {type(e).__name__}: {str(e)}")

    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # add teleoperation key for task success
    should_reset_task_success = False

    def reset_task_success():
        nonlocal should_reset_task_success
        if not start_record_state:
            print("[Control] Save ignored because recording has not started.")
            return
        if task525_markers_enabled:
            if (
                args_cli.task525_base_mode == "dijkstra"
                and (task525_auto_navigation is None or not task525_auto_navigation.completed)
            ):
                status = "unavailable" if task525_auto_navigation is None else task525_auto_navigation.status
                reason = "" if task525_auto_navigation is None else f" ({task525_auto_navigation.failure_reason})"
                print(
                    "[Task525] Save refused: automatic Dijkstra navigation has not completed "
                    f"(status={status}){reason}. Press R to reset after a failure."
                )
                return
            required_marker_names = (
                ("grasp_step", "lift_step", "navigate_step", "place_step")
                if args_cli.task525_base_mode == "dijkstra"
                else tuple(task525_markers)
            )
            missing = [name for name in required_marker_names if task525_markers[name] is None]
            if missing:
                print(
                    "[Task525] Save refused: missing phase markers "
                    + (
                        f"{missing}. Use G after a stable grasp, wait for automatic carrying-home/navigation, "
                        "toggle the right tact after arrival, perform place/release, then press N."
                        if args_cli.task525_base_mode == "dijkstra"
                        else f"{missing}. Use F, G, M, stop the manually controlled base, P, place/release, H, "
                        "return home, then press N."
                    )
                )
                return
        should_reset_task_success = True

    teleop_interface.add_callback("R", reset_recording_instance)
    teleop_interface.add_callback("N", reset_task_success)

    def mark_grasp_step():
        if not task525_markers_enabled or not start_record_state:
            print("[Task525] F ignored: start a Task525 marked recording with B first.")
            return
        if task525_markers["grasp_step"] is not None:
            print("[Task525] F ignored: stable grasp has already been marked.")
            return
        if recorded_step_index < 2:
            print("[Task525] F ignored: record the approach and gripper close before marking a stable grasp.")
            return
        task525_markers["grasp_step"] = recorded_step_index
        task525_markers["lift_step"] = recorded_step_index
        print(
            f"[Task525] grasp_step={recorded_step_index}, lift_step={recorded_step_index}. "
            "Remove the can from the cabinet and return the right arm to the carry/home pose."
        )

    def mark_lift_step():
        nonlocal should_start_task525_auto_navigation
        if not task525_markers_enabled or not start_record_state:
            print("[Task525] G ignored: start a Task525 marked recording with B first.")
            return
        if args_cli.task525_base_mode == "dijkstra":
            if task525_auto_navigation is not None and task525_auto_navigation.active:
                print("[Task525] G ignored: automatic carrying-home/Dijkstra navigation is already running.")
                return
            if (
                task525_auto_navigation is not None
                and task525_auto_navigation.status == task525_auto_navigation.FAILED
            ):
                print("[Task525] G ignored: automatic navigation failed. Press R to reset before retrying.")
                return
            if task525_auto_navigation is not None and task525_auto_navigation.completed:
                print("[Task525] G ignored: automatic navigation has already completed; perform place.")
                return
            if (
                task525_auto_navigation is not None
                and task525_auto_navigation.awaiting_place_activation
            ):
                print("[Task525] G ignored: base has arrived; toggle the right A3 tact to enable place.")
                return
            # F remains an optional, more precise grasp annotation. G is the
            # only required operator signal in automatic mode: it records the
            # grasp boundary and initiates the safe return-home transition.
            task525_markers["grasp_step"] = (
                recorded_step_index
                if task525_markers["grasp_step"] is None
                else task525_markers["grasp_step"]
            )
            task525_markers["lift_step"] = recorded_step_index
            should_start_task525_auto_navigation = True
            print(
                "[Task525] G accepted: preserving the closed right gripper, returning the arm to its "
                "A3-aligned Task525 init/home, then starting Dijkstra navigation."
            )
            return
        if task525_markers["grasp_step"] is None:
            print("[Task525] G ignored: press F after the stable grasp first.")
            return
        if task525_markers["navigate_step"] is not None:
            print("[Task525] G ignored: navigation has already been marked.")
            return
        print("[Task525] G accepted: carry/home is ready; press M to begin manual navigation.")

    def mark_task525_navigation():
        if not task525_markers_enabled or not start_record_state:
            print("[Task525] M ignored: start a Task525 marked recording with B first.")
            return
        if args_cli.task525_base_mode == "dijkstra":
            print("[Task525] M is not needed in Dijkstra mode. Press G after pick/carry to start auto navigation.")
            return
        if task525_markers["lift_step"] is None:
            print("[Task525] M ignored: press F after grasp and G after carry/home first.")
            return
        task525_markers["navigate_step"] = recorded_step_index
        task525_markers["place_step"] = None
        print(
            f"[Task525] navigate_step={recorded_step_index}. "
            "Move the base with the foot pad or keyboard; A3 arm mappings remain unchanged."
        )

    def mark_task525_place():
        if not task525_markers_enabled or not start_record_state:
            print("[Task525] P ignored: start a Task525 marked recording with B first.")
            return
        if task525_markers["navigate_step"] is None:
            print("[Task525] P ignored: mark navigation start with M first.")
            return
        if args_cli.task525_base_mode == "dijkstra":
            if task525_auto_navigation is None or not task525_auto_navigation.completed:
                print("[Task525] P ignored: wait for the automatic Dijkstra arrival message first.")
                return
            task525_markers["place_step"] = recorded_step_index
            print(f"[Task525] place_step={recorded_step_index}. Begin right-arm place.")
            return
        wheel_speed_norm = _sg2_wheel_speed_norm(env)
        if wheel_speed_norm > 0.1:
            print(
                "[Task525] P ignored because the base is still moving "
                f"({wheel_speed_norm:.3f} rad/s > 0.100 rad/s). Stop and wait, then press P again."
            )
            return
        task525_markers["place_step"] = recorded_step_index
        print(f"[Task525] place_step={recorded_step_index}. Keep the base stopped and perform place.")

    def mark_task525_release():
        if not task525_markers_enabled or not start_record_state:
            print("[Task525] H ignored: start a Task525 marked recording with B first.")
            return
        if task525_markers["place_step"] is None:
            print("[Task525] H ignored: wait for the place phase before marking release.")
            return
        if task525_markers["release_step"] is not None:
            print("[Task525] H ignored: release has already been marked.")
            return
        task525_markers["release_step"] = recorded_step_index
        print(
            f"[Task525] release_step={recorded_step_index}. "
            "Return the right arm to its initial pose, then press N to save."
        )

    def current_task525_phase() -> int:
        if task525_markers["release_step"] is not None:
            return 4
        if task525_markers["place_step"] is not None:
            return 3
        if task525_markers["navigate_step"] is not None:
            return 2
        if task525_markers["lift_step"] is not None:
            return 1
        return 0

    teleop_interface.add_callback("F", mark_grasp_step)
    teleop_interface.add_callback("G", mark_lift_step)
    teleop_interface.add_callback("M", mark_task525_navigation)
    teleop_interface.add_callback("P", mark_task525_place)
    teleop_interface.add_callback("H", mark_task525_release)

    target_step_hz = args_cli.step_hz
    if target_step_hz is None:
        target_step_hz = getattr(env_cfg, "recording_control_hz", 60.0)
    if args_cli.task525_base_mode == "dijkstra":
        from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.online_dijkstra import (
            Task525OnlineDijkstraCfg,
            Task525OnlineDijkstraNavigator,
        )

        task525_auto_navigation = Task525OnlineDijkstraNavigator(
            env,
            Task525OnlineDijkstraCfg(
                linear_max=args_cli.task525_dijkstra_linear_max,
                angular_max=args_cli.task525_dijkstra_angular_max,
                max_navigation_seconds=args_cli.task525_dijkstra_timeout_s,
            ),
            control_hz=target_step_hz,
        )
    rate_limiter = RateLimiter(target_step_hz)
    recording_metadata = _recording_metadata(env_cfg, task_name, target_step_hz)
    if task525_markers_enabled:
        auto_dijkstra = args_cli.task525_base_mode == "dijkstra"
        recording_metadata.update({
            "task525_seed_mode": (
                "continuous_pick_online_dijkstra_base_place"
                if auto_dijkstra
                else "continuous_pick_manual_base_place"
            ),
            "task525_phase_semantics": (
                "0=grasp,1=clear_cabinet_and_return_carry_home,2=navigation,"
                "3=place_and_release,4=return_home_without_object"
            ),
            "task525_base_command_source": (
                "online_dijkstra_fixed_yaw_holonomic"
                if auto_dijkstra
                else "external_cmd_vel_foot_pad_or_keyboard"
            ),
            "task525_online_dijkstra": auto_dijkstra,
            "task525_a3_mapping": "original_dual_arm_left_to_left_right_to_right",
            "task525_collection_arm_usage": "right_only_for_this_recording",
        })
    _set_dataset_metadata(env.recorder_manager, recording_metadata)

    # reset environment
    env.reset()
    teleop_interface.reset()
    if task525_markers_enabled and args_cli.task525_base_mode == "dijkstra":
        # A new recorder process must start at the reset height even if the
        # A3 lift leader still publishes the previous episode's lowered pose.
        task525_reset_lift_hold_target = task525_reset_lift_target()
        task525_reset_joint_hold_target = task525_save_pose_target()
    operator_view = make_operator_view(env_cfg, env)

    current_recorded_demo_count = 0

    should_start_recording_instance = False

    def start_recording_instance():
        nonlocal start_record_state, recorded_step_index
        nonlocal should_start_task525_auto_navigation, task525_auto_navigation_failure_reported
        nonlocal task525_home_arm_action, task525_place_activation_generation
        nonlocal task525_reset_joint_hold_target, task525_reset_lift_hold_target
        if start_record_state:
            return
        if args_cli.robot_type == "FFW_SG2":
            wheel_speed_norm = _sg2_wheel_speed_norm(env)
            if wheel_speed_norm > 0.1:
                print(
                    "[Control] Recording was not started because the SG2 wheels "
                    f"are still moving ({wheel_speed_norm:.3f} rad/s > 0.100 rad/s). "
                    "Wait for the robot to settle, then press B again."
                )
                return
        clear_episode_cache("pre-recording")
        env.recorder_manager.recording_enabled = True
        env.recorder_manager.record_post_reset(torch.arange(env.num_envs, device=env.device))
        recorded_step_index = 0
        for marker_name in task525_markers:
            task525_markers[marker_name] = None
        should_start_task525_auto_navigation = False
        task525_auto_navigation_failure_reported = False
        task525_place_activation_generation = None
        if task525_auto_navigation is not None:
            task525_auto_navigation.reset()
            # G always returns both arms to the A3-aligned Task525 init/home.
            # It later substitutes only the measured carrying gripper and
            # task-specific maximum-down head target.
            task525_home_arm_action = task525_save_pose_target()
        task525_reset_joint_hold_target = None
        if task525_markers_enabled and args_cli.task525_base_mode == "dijkstra":
            # A3 and Task525 use the same absolute init/home joint values.
            # Do not add a task-local offset: subsequent A3 save-pose commands
            # must reach the exact specified joint target.
            teleop_interface.begin_control_activation()
            print(
                "[Task525] B activation: using absolute A3 joint commands for right-arm/head/lift; "
                "the unused left arm remains at its settled reset pose. No relative offset is used."
            )
        if not (task525_markers_enabled and args_cli.task525_base_mode == "dijkstra"):
            # Manual-base collection continues to expose the original lift
            # topic. Automatic Task525 owns lift height until G arrives.
            task525_reset_lift_hold_target = None
        start_record_state = True
        print("Start Recording!!!")
        if task525_markers_enabled and args_cli.task525_base_mode == "dijkstra":
            print(
                "[Task525] Recording is armed. B only starts recording; after a stable grasp, "
                "lift the can clear of the cabinet and press G to start automatic navigation."
            )

    def request_start_recording_instance():
        nonlocal should_start_recording_instance
        should_start_recording_instance = True

    teleop_interface.add_callback("B", request_start_recording_instance)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with profiler.time("loop_total"):
            with torch.inference_mode():
                with profiler.time("publish_observations"):
                    teleop_interface.publish_observations()
                with profiler.time("get_action"):
                    actions = teleop_interface.get_action()
                if should_start_recording_instance:
                    should_start_recording_instance = False
                    with profiler.time("start_recording"):
                        start_recording_instance()
                    # ``actions`` was sampled before B established the
                    # Refresh after activation so the first recording frame
                    # uses the newly anchored absolute-command blend.
                    actions = teleop_interface.get_action()
                if should_reset_task_success:
                    with profiler.time("save_success_export"):
                        should_reset_task_success = False
                        print("[Control] Saving operator-accepted demo.")

                        env.termination_manager.set_term_cfg(
                            "success",
                            TerminationTermCfg(
                                func=lambda env: torch.ones(
                                    env.num_envs, dtype=torch.bool, device=env.device
                                )
                            ),
                        )
                        env.termination_manager.compute()
                        try:
                            episode_metadata = dict(recording_metadata)
                            episode_metadata["operator_accepted"] = True
                            if task525_markers_enabled:
                                episode_metadata.update(task525_markers)
                            for ep in getattr(env.recorder_manager, "_episodes", {}).values():
                                if ep is not None and not ep.is_empty():
                                    ep.success = True
                            _set_episode_metadata(env.recorder_manager, episode_metadata)
                        except Exception as exc:
                            print(f"Warning: Failed to mark episodes as successful: {exc}")

                        env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                        env.recorder_manager.export_episodes(from_step=False)
                        env.recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
                        should_reset_recording_instance = True
                        if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                            current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                            print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                if should_reset_recording_instance:
                    with profiler.time("reset_recording"):
                        # Clear any buffered episode so failed episodes (key 'R') aren't saved
                        env.recorder_manager.recording_enabled = False
                        clear_episode_cache("recording")

                        env.reset()
                        # R/N must not replay the old lift target after a
                        # G-driven lower. Hold the freshly reset lift until
                        # the next B deliberately starts a new collection.
                        if task525_markers_enabled:
                            task525_reset_lift_hold_target = task525_reset_lift_target()
                            task525_reset_joint_hold_target = task525_save_pose_target()
                            teleop_interface.clear_command_cache()
                        for marker_name in task525_markers:
                            task525_markers[marker_name] = None
                        should_start_task525_auto_navigation = False
                        task525_auto_navigation_failure_reported = False
                        task525_home_arm_action = None
                        task525_place_activation_generation = None
                        if task525_auto_navigation is not None:
                            task525_auto_navigation.reset()
                        should_reset_recording_instance = False
                        if start_record_state:
                            print("Stop Recording!!!")
                        start_record_state = False
                        if task525_markers_enabled:
                            print(
                                "[Task525] Reset complete: lift is held at the reset height until G; "
                                "only a later G navigation can run the automatic 30 cm lower."
                            )
                        env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)))
                        # print out the current demo count if it has changed
                        print(f"Resetting recording instance. Current recorded demo count: {current_recorded_demo_count}")
                        if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                            current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                            print(f"Recorded {current_recorded_demo_count} successful demonstrations.")
                        if args_cli.num_demos > 0 and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos:
                            print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                            break

                elif actions is None:
                    with profiler.time("env_render_no_action"):
                        env.render()
                # apply actions
                else:
                    if isinstance(actions, dict):
                        # Handle dictionary actions (like reset)
                        if "reset" in actions:
                            # This is a reset action, don't step the environment
                            with profiler.time("env_render_reset_action"):
                                env.render()
                            if operator_view is not None:
                                with profiler.time("operator_camera_view"):
                                    operator_view.update()
                            continue
                    else:
                        # Handle tensor actions
                        if actions.ndim == 1:
                            actions = actions.unsqueeze(0)
                        if (
                            task525_markers_enabled
                            and args_cli.task525_base_mode == "dijkstra"
                            and not start_record_state
                            and task525_reset_joint_hold_target is not None
                        ):
                            actions[:, :19] = task525_reset_joint_hold_target.to(
                                device=actions.device, dtype=actions.dtype
                            )
                        if (
                            task525_markers_enabled
                            and args_cli.task525_base_mode == "dijkstra"
                            and task525_reset_lift_hold_target is not None
                        ):
                            actions[:, TASK525_LIFT_ACTION_INDEX] = task525_reset_lift_hold_target.to(
                                device=actions.device, dtype=actions.dtype
                            )
                        if (
                            task525_markers_enabled
                            and args_cli.task525_base_mode == "dijkstra"
                            and should_start_task525_auto_navigation
                        ):
                            should_start_task525_auto_navigation = False
                            if task525_auto_navigation is None:
                                raise RuntimeError("Task525 Dijkstra navigator was not initialized.")
                            if task525_home_arm_action is None:
                                raise RuntimeError(
                                    "Task525 automatic home target is unavailable; press B to begin a new episode."
                                )
                            home_arm_action = task525_home_arm_action.to(
                                device=actions.device, dtype=actions.dtype
                            ).clone()
                            # Re-enter canonical save-pose 3 without
                            # snapping the can-holding gripper to a cached A3
                            # target. The actual simulated master joint is
                            # the only authoritative grip value at G.
                            measured_action = teleop_interface.get_measured_joint_hold_action()
                            home_arm_action[:, TASK525_RIGHT_GRIPPER_ACTION_INDEX] = measured_action[
                                :, TASK525_RIGHT_GRIPPER_ACTION_INDEX
                            ]
                            # The carry/base segment has no operator head
                            # control: always look maximally downward and
                            # center left/right while returning home.
                            home_arm_action[:, TASK525_HEAD_PITCH_ACTION_INDEX] = (
                                TASK525_HEAD_PITCH_DOWN_MAX_RAD
                            )
                            home_arm_action[:, TASK525_HEAD_YAW_ACTION_INDEX] = 0.0
                            if task525_auto_navigation.start(actions, home_arm_action=home_arm_action):
                                task525_markers["place_step"] = None
                                print(
                                    "[Task525 AutoNav] Dijkstra path ready: "
                                    f"{len(task525_auto_navigation.path.points)} waypoints, "
                                    f"{float(task525_auto_navigation.path.get_path_length()):.3f} m. "
                                    "Returning the arm home first; base commands stay zero until it settles."
                                )
                            else:
                                task525_auto_navigation_failure_reported = True
                                print(
                                    "[Task525 AutoNav] Failed before motion. "
                                    f"{task525_auto_navigation.failure_reason} Press R to reset."
                                )

                        if task525_markers_enabled and args_cli.task525_base_mode == "dijkstra":
                            if task525_auto_navigation is None:
                                raise RuntimeError("Task525 Dijkstra navigator was not initialized.")
                            status_before = task525_auto_navigation.status
                            actions = task525_auto_navigation.apply(actions)
                            if (
                                status_before == task525_auto_navigation.RETURNING_HOME
                                and task525_auto_navigation.status == task525_auto_navigation.NAVIGATING
                            ):
                                task525_markers["navigate_step"] = recorded_step_index
                                print(
                                    "[Task525 AutoNav] Carrying-home pose settled. Starting Dijkstra base motion; "
                                    "manual base commands remain ignored."
                                )
                            if (
                                status_before != task525_auto_navigation.WAITING_FOR_PLACE_ACTIVATION
                                and task525_auto_navigation.awaiting_place_activation
                            ):
                                if task525_auto_navigation.frozen_arm_action is None:
                                    raise RuntimeError("Task525 lift target is unavailable at place handoff.")
                                # Keep the post-G lower through tact/place;
                                # outside this one path the reset-height lock
                                # remains in effect.
                                task525_reset_lift_hold_target = (
                                    task525_auto_navigation.frozen_arm_action[
                                        :, TASK525_LIFT_ACTION_INDEX
                                    ].detach().clone()
                                )
                                task525_place_activation_generation = teleop_interface.right_arm_tact_generation()
                                print(
                                    "[Task525 AutoNav] Arrived, wheel motion settled, and lift lowered. "
                                    "Base and arms are held. Press the right A3 tact once to enable right-arm place."
                                )
                            if task525_auto_navigation.awaiting_place_activation:
                                if task525_place_activation_generation is None:
                                    raise RuntimeError("Task525 place handoff is missing its right-arm command snapshot.")
                                if (
                                    teleop_interface.right_arm_tact_generation()
                                    > task525_place_activation_generation
                                ):
                                    if task525_auto_navigation.enable_place_control():
                                        teleop_interface.begin_control_activation()
                                        task525_markers["place_step"] = recorded_step_index
                                        print(
                                        "[Task525 AutoNav] Right A3 tact received. "
                                            "Right-arm place is enabled; base remains held at zero."
                                        )
                            if (
                                task525_auto_navigation.status == task525_auto_navigation.FAILED
                                and not task525_auto_navigation_failure_reported
                            ):
                                task525_auto_navigation_failure_reported = True
                                print(
                                    "[Task525 AutoNav] Stopped without reaching the goal. "
                                    f"{task525_auto_navigation.failure_reason} Press R to reset."
                                )
                        if start_record_state and env.recorder_manager.recording_enabled:
                            _record_step_metadata(
                                env.recorder_manager,
                                recorded_step_index,
                                target_step_hz,
                                env.device,
                                task525_phase=(
                                    current_task525_phase()
                                    if task525_markers_enabled
                                    else None
                                ),
                            )
                            recorded_step_index += 1
                        with profiler.time("env_step"):
                            env.step(actions)
                if operator_view is not None:
                    with profiler.time("operator_camera_view"):
                        operator_view.update()
                if rate_limiter:
                    with profiler.time("rate_sleep"):
                        rate_limiter.sleep()
        profiler.tick()

    # close the simulator
    if operator_view is not None:
        operator_view.close()
    teleop_interface.shutdown()
    profiler.remove_hooks()
    release_camera_sensors_before_close(env)
    env.close()
    simulation_app.close()

    if args_cli.render_episode_cameras:
        from pathlib import Path
        import sys

        renderer_dir = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(renderer_dir))
        from render_hdf5_cameras import render_dataset

        try:
            index_path = render_dataset(
                Path(args_cli.dataset_file),
                camera_names=getattr(
                    env_cfg,
                    "policy_camera_names",
                    ("cam_head", "cam_wrist_left", "cam_wrist_right"),
                ),
                rotations=dict(getattr(env_cfg, "operator_camera_rotations", ())),
                fps=float(getattr(env_cfg, "camera_hz", target_step_hz)),
                overwrite=True,
            )
            print(f"[Camera Preview] Open {index_path} to review all episodes.")
        except Exception as exc:
            print(f"[Camera Preview] Failed to render saved episodes: {exc}")


if __name__ == "__main__":
    # run the main function
    main()
