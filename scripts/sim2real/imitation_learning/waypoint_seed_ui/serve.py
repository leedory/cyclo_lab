"""Local browser UI for drafting plans and controlling its isolated Isaac runner."""

from __future__ import annotations

import argparse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urlsplit
from uuid import uuid4

from seed_plan import (
    IK_SOLVER_LABELS,
    LAB_ROOT,
    SG2_URDF_FILE,
    discover_environments,
    joint_contract,
    validate_plan,
)


HERE = Path(__file__).resolve().parent
PLANS = LAB_ROOT / ".runtime/joint_waypoint_seed_ui/plans"
STATE_FILE = LAB_ROOT / ".runtime/joint_waypoint_seed_ui/live_state.json"
CAMERA_DIR = STATE_FILE.parent / "cameras"
DATASET_ROOT = LAB_ROOT / "datasets/waypoint_seed"
CAMERA_LABELS = (
    "overhead_left", "overhead_center", "overhead_right",
    "wrist_left", "head", "wrist_right",
)
# The Isaac container writes datasets as root. Keep browser-to-runner control
# messages in a directory created by the host UI server so both sides can
# atomically replace files without changing dataset ownership.
CONTROL_DIR = LAB_ROOT / ".runtime/joint_waypoint_seed_ui"
IK_REQUEST_FILE = CONTROL_DIR / "ik_request.json"
IK_RESULT_FILE = CONTROL_DIR / "ik_result.json"
PREVIEW_REQUEST_FILE = CONTROL_DIR / "preview_request.json"
PREVIEW_RESULT_FILE = CONTROL_DIR / "preview_result.json"
RESET_REQUEST_FILE = CONTROL_DIR / "reset_request.json"
RESET_RESULT_FILE = CONTROL_DIR / "reset_result.json"
OBJECT_REQUEST_FILE = CONTROL_DIR / "object_request.json"
OBJECT_RESULT_FILE = CONTROL_DIR / "object_result.json"
STOP_REQUEST_FILE = CONTROL_DIR / "stop_request.json"
STOP_RESULT_FILE = CONTROL_DIR / "stop_result.json"
CONNECT_PLAN_FILE = CONTROL_DIR / "connect_plan.json"
STAGED_SG2_URDF_FILE = CONTROL_DIR / "ffw_sg2_follower.urdf"
ACCEPT_REQUEST_FILE = CONTROL_DIR / "accept_request.json"
EXPORT_STATUS_FILE = CONTROL_DIR / "export_status.json"
RUNNER_CONTAINER = os.environ.get("SEED_RUNNER_CONTAINER", "cyclo_lab")
RUNNER_LAB_ROOT = os.environ.get("SEED_RUNNER_LAB_ROOT", "/workspace/cyclo_lab")
RUNNER_DIR = os.environ.get(
    "SEED_RUNNER_DIR", f"{RUNNER_LAB_ROOT}/scripts/sim2real/imitation_learning/waypoint_seed_ui"
)
RUNNER_PYTHON = os.environ.get(
    "SEED_RUNNER_PYTHON", f"{RUNNER_LAB_ROOT}/third_party/IsaacLab/_isaac_sim/python.sh"
)
CONNECT_LOCK = threading.RLock()
EXPORT_LOCK = threading.Lock()
RENDER_LOCK = threading.Lock()
LAUNCH_STATE = {"environment": None, "started_at_ms": 0}
RENDER_JOBS: dict[str, dict] = {}


def live_state() -> dict:
    state = {"status": "idle"}
    if STATE_FILE.is_file():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            state = {"status": "unavailable"}
    now_ms = round(time.time() * 1000)
    if state.get("status") in ("running", "awaiting_acceptance") and now_ms - state.get("updated_at_ms", 0) > 3000:
        state["status"] = "stale"
    with CONNECT_LOCK:
        launched_at = int(LAUNCH_STATE["started_at_ms"] or 0)
        launched_env = LAUNCH_STATE["environment"]
        if state.get("status") in ("running", "awaiting_acceptance") and state.get("environment") == launched_env:
            LAUNCH_STATE.update(environment=None, started_at_ms=0)
        elif launched_env and now_ms - launched_at < 180_000:
            return {
                "status": "starting", "environment": launched_env,
                "updated_at_ms": launched_at, "phase": "launching",
            }
        elif launched_env:
            LAUNCH_STATE.update(environment=None, started_at_ms=0)
    return state


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stage_sg2_urdf() -> None:
    """Expose the validated host URDF inside the already-mounted Cyclo Lab tree."""
    joint_contract()
    if not SG2_URDF_FILE.is_file():
        raise ValueError(
            "The advanced TCP IK solvers need the SG2 URDF on the host. "
            "Set SG2_URDF_PATH before starting the UI server."
        )
    payload = SG2_URDF_FILE.read_bytes()
    if STAGED_SG2_URDF_FILE.is_file() and STAGED_SG2_URDF_FILE.read_bytes() == payload:
        return
    STAGED_SG2_URDF_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STAGED_SG2_URDF_FILE.with_suffix(".urdf.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, STAGED_SG2_URDF_FILE)


def validate_control_rate(control_hz: float, physics_hz: float = 120.0) -> tuple[float, int]:
    ratio = physics_hz / float(control_hz)
    decimation = round(ratio)
    if control_hz > physics_hz or decimation < 1 or abs(ratio - decimation) > 1.0e-9:
        supported = "1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120"
        raise ValueError(
            f"{control_hz:g} Hz cannot be applied with {physics_hz:g} Hz physics. "
            f"Supported integer rates: {supported}"
        )
    return physics_hz, int(decimation)


def validate_folder_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Enter a Seed Export folder name.")
    value = value.strip()
    if not value or len(value) > 80 or not re.fullmatch(r"[\w-]+", value, flags=re.UNICODE):
        raise ValueError("Use up to 80 letters, numbers, underscores, or hyphens for the folder name.")
    return value


def validate_plan_filename(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("Enter a Save Waypoints filename.")
    name = value.strip()
    if name.endswith(".json"):
        name = name[:-5]
    if not name or len(name) > 80 or not re.fullmatch(r"[\w-]+", name, flags=re.UNICODE):
        raise ValueError("Use up to 80 letters, numbers, underscores, or hyphens for the filename.")
    return f"{name}.json"


def seed_path(relative: str, *, suffix: str | None = None) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("A seed path is required.")
    path = (DATASET_ROOT / relative).resolve()
    root = DATASET_ROOT.resolve()
    if path == root or root not in path.parents or (suffix and path.suffix != suffix):
        raise ValueError("The seed path is invalid.")
    return path


def export_status(payload: dict) -> None:
    payload = {**payload, "updated_at_ms": round(time.time() * 1000)}
    write_json_atomic(EXPORT_STATUS_FILE, payload)


def start_export_job(job_id: str, plan: dict, folder: str) -> None:
    try:
        export_status({"id": job_id, "status": "stopping_preview", "folder": folder})
        state = live_state()
        if state.get("status") == "awaiting_acceptance" or state.get("recording_mode"):
            raise ValueError("A Seed Export is already recording.")
        if state.get("status") in ("running", "starting"):
            if state.get("environment") != plan["environment"]:
                raise ValueError(f"Another environment is running: {state.get('environment')}")
            request = {
                "id": uuid4().hex, "run_id": state.get("run_id"),
                "environment": state["environment"], "requested_at_ms": round(time.time() * 1000),
            }
            write_json_atomic(STOP_REQUEST_FILE, request)
            deadline = time.monotonic() + 90.0
            while time.monotonic() < deadline:
                current = live_state()
                if current.get("status") not in ("running", "starting", "awaiting_acceptance"):
                    break
                time.sleep(0.5)
            else:
                raise ValueError("The preview runner did not stop within 90 seconds. Check Disconnect status.")

        export_status({"id": job_id, "status": "launching_recorder", "folder": folder})
        STOP_REQUEST_FILE.unlink(missing_ok=True)
        STOP_RESULT_FILE.unlink(missing_ok=True)
        ACCEPT_REQUEST_FILE.unlink(missing_ok=True)
        plan_file = CONTROL_DIR / f"export_plan_{job_id}.json"
        write_json_atomic(plan_file, plan)
        # A folder is a seed collection. Always add a uniquely named HDF5 so
        # the same Seed Export folder can be reused without overwriting files.
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        dataset_name = f"seed_{stamp}.hdf5"
        dataset_stem = Path(dataset_name).stem
        container_folder = f"{RUNNER_LAB_ROOT}/datasets/waypoint_seed/{folder}"
        container_dataset = f"{container_folder}/{dataset_name}"
        container_plan = f"{RUNNER_LAB_ROOT}/.runtime/joint_waypoint_seed_ui/{plan_file.name}"
        manifest_file = CONTROL_DIR / f"export_manifest_{job_id}.json"
        write_json_atomic(manifest_file, {
            "folder": folder, "dataset_file": dataset_name,
            "environment": plan["environment"], "control_hz": plan["control_hz"],
            "initial_pose_hold_s": plan["initial_pose_hold_s"],
            "object_position_xyz_m": plan.get("object_position_xyz_m"),
        })
        container_manifest = f"{RUNNER_LAB_ROOT}/.runtime/joint_waypoint_seed_ui/{manifest_file.name}"
        subprocess.run(
            ["docker", "exec", RUNNER_CONTAINER, "mkdir", "-p", container_folder],
            check=True, capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["docker", "exec", RUNNER_CONTAINER, "cp", container_plan, f"{container_folder}/{dataset_stem}_plan.json"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        subprocess.run(
            ["docker", "exec", RUNNER_CONTAINER, "cp", container_manifest, f"{container_folder}/{dataset_stem}.json"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        subprocess.run([
            "docker", "exec", "-d", "-e", f"CYCLO_LAB_ROOT={RUNNER_LAB_ROOT}",
            "-w", RUNNER_DIR, RUNNER_CONTAINER, RUNNER_PYTHON, "run_seed.py",
            "--plan", container_plan, "--execute", "--headless", "--record-hdf5", "--auto-accept",
            "--dataset-file", container_dataset,
        ], check=True, capture_output=True, text=True, timeout=30)
        launched_at_ms = round(time.time() * 1000)
        with CONNECT_LOCK:
            LAUNCH_STATE.update(environment=plan["environment"], started_at_ms=launched_at_ms)
        export_status({"id": job_id, "status": "loading_environment", "folder": folder})
        deadline = time.monotonic() + 120.0
        current = None
        while time.monotonic() < deadline:
            current = live_state()
            if (
                current.get("environment") == plan["environment"]
                and current.get("recording_mode")
                and current.get("status") in ("running", "finalizing", "saved")
            ):
                applied_hz = float(current.get("sim_control_hz", 0))
                if abs(applied_hz - plan["control_hz"]) > 1.0e-6:
                    request = {
                        "id": uuid4().hex, "run_id": current.get("run_id"),
                        "environment": plan["environment"], "requested_at_ms": round(time.time() * 1000),
                    }
                    write_json_atomic(STOP_REQUEST_FILE, request)
                    raise ValueError(
                        f"Requested {plan['control_hz']:g} Hz, but {applied_hz:g} Hz was applied. Recording stopped."
                    )
                break
            if (
                current.get("environment") == plan["environment"]
                and current.get("status") == "failed"
                and int(current.get("updated_at_ms", 0)) >= launched_at_ms
            ):
                raise ValueError(current.get("error") or "The HDF5 recorder failed to start.")
            time.sleep(0.25)
        else:
            raise ValueError("The recorder did not report its applied control rate within 120 seconds.")
        export_status({
            "id": job_id, "status": "recording_started", "folder": folder,
            "dataset": f"{folder}/{dataset_name}", "control_hz": float(current["sim_control_hz"]),
            "physics_hz": float(current["physics_hz"]), "decimation": int(current["decimation"]),
        })
    except Exception as exc:
        with CONNECT_LOCK:
            LAUNCH_STATE.update(environment=None, started_at_ms=0)
        export_status({"id": job_id, "status": "error", "folder": folder, "error": str(exc)})
    finally:
        if EXPORT_LOCK.locked():
            EXPORT_LOCK.release()


def render_seed_job(job_id: str, dataset: Path, relative: str, fps: float) -> None:
    try:
        with RENDER_LOCK:
            RENDER_JOBS[job_id] = {"status": "rendering", "dataset": relative}
        relative_container = dataset.relative_to(DATASET_ROOT).as_posix()
        container_dataset = f"{RUNNER_LAB_ROOT}/datasets/waypoint_seed/{relative_container}"
        renderer = f"{RUNNER_LAB_ROOT}/scripts/sim2real/imitation_learning/render_hdf5_cameras.py"
        subprocess.run([
            "docker", "exec", RUNNER_CONTAINER, RUNNER_PYTHON, renderer,
            "--input-file", container_dataset, "--fps", f"{fps:g}", "--overwrite",
        ], check=True, capture_output=True, text=True, timeout=600)
        video = dataset.with_name(f"{dataset.stem}_camera_previews") / "demo_0_three_cameras.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"Rendered video not found: {video}")
        url = f"/api/seed-video?path={quote(video.relative_to(DATASET_ROOT).as_posix())}"
        with RENDER_LOCK:
            RENDER_JOBS[job_id] = {"status": "ready", "dataset": relative, "video_url": url}
    except Exception as exc:
        with RENDER_LOCK:
            RENDER_JOBS[job_id] = {"status": "error", "dataset": relative, "error": str(exc)}


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._reply(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _video(self, path: Path) -> None:
        size = path.stat().st_size
        start, end, status = 0, max(0, size - 1), 200
        header = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header) if header else None
        if match:
            if match.group(1):
                start = int(match.group(1))
                end = min(end, int(match.group(2))) if match.group(2) else end
            elif match.group(2):
                start = max(0, size - int(match.group(2)))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = length
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self) -> None:
        request = urlsplit(self.path)
        if request.path == "/":
            self._reply(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif request.path == "/api/catalog":
            joints, limits = joint_contract()
            self._json(200, {
                "environments": discover_environments(), "joints": joints, "limits": limits,
                "limits_source": str(SG2_URDF_FILE) if SG2_URDF_FILE.is_file() else "Cyclo Lab URDF-derived 19D limits",
                "ik_solvers": IK_SOLVER_LABELS,
            })
        elif request.path == "/api/plans":
            self._json(200, {"plans": sorted((path.name for path in PLANS.glob("*.json") if path.is_file()), reverse=True)})
        elif request.path.startswith("/api/plans/"):
            name = unquote(request.path.removeprefix("/api/plans/"))
            try:
                valid_name = validate_plan_filename(name)
            except ValueError:
                self._json(404, {"error": "Plan not found"})
                return
            if valid_name != name:
                self._json(404, {"error": "Plan not found"})
                return
            try:
                plan = validate_plan(json.loads((PLANS / name).read_text(encoding="utf-8")), discover_environments())
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(200, {"name": name, "plan": plan})
        elif request.path == "/api/state":
            self._json(200, live_state())
        elif request.path == "/api/ik-result":
            request_id = parse_qs(request.query).get("id", [""])[0]
            try:
                result = json.loads(IK_RESULT_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = None
            if result is None or result.get("id") != request_id or not request_id:
                self._json(202, {"status": "pending"})
            else:
                self._json(200, result)
        elif request.path == "/api/preview-result":
            request_id = parse_qs(request.query).get("id", [""])[0]
            try:
                result = json.loads(PREVIEW_RESULT_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = None
            if result is None or result.get("id") != request_id or not request_id:
                self._json(202, {"status": "pending"})
            else:
                self._json(200, result)
        elif request.path == "/api/reset-result":
            request_id = parse_qs(request.query).get("id", [""])[0]
            try:
                result = json.loads(RESET_RESULT_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = None
            if result is None or result.get("id") != request_id or not request_id:
                self._json(202, {"status": "pending"})
            else:
                self._json(200, result)
        elif request.path == "/api/object-result":
            request_id = parse_qs(request.query).get("id", [""])[0]
            try:
                result = json.loads(OBJECT_RESULT_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = None
            if result is None or result.get("id") != request_id or not request_id:
                self._json(202, {"status": "pending"})
            else:
                self._json(200, result)
        elif request.path == "/api/export-status":
            request_id = parse_qs(request.query).get("id", [""])[0]
            try:
                result = json.loads(EXPORT_STATUS_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = None
            if result is None or result.get("id") != request_id or not request_id:
                self._json(202, {"status": "pending"})
            else:
                self._json(200, result)
        elif request.path == "/api/seeds":
            seeds = []
            if DATASET_ROOT.is_dir():
                for path in sorted(DATASET_ROOT.rglob("*.hdf5"), key=lambda item: item.stat().st_mtime, reverse=True):
                    if path.name.endswith(".pending.hdf5"):
                        continue
                    relative = path.relative_to(DATASET_ROOT).as_posix()
                    metadata = {}
                    try:
                        metadata_path = path.with_suffix(".json")
                        if not metadata_path.is_file():
                            metadata_path = path.parent / "export.json"
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        pass
                    video = path.with_name(f"{path.stem}_camera_previews") / "demo_0_three_cameras.mp4"
                    seeds.append({
                        "relative": relative,
                        "folder": path.parent.relative_to(DATASET_ROOT).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "modified_at_ms": round(path.stat().st_mtime * 1000),
                        "control_hz": metadata.get("control_hz"),
                        "environment": metadata.get("environment"),
                        "video_url": (
                            f"/api/seed-video?path={quote(video.relative_to(DATASET_ROOT).as_posix())}"
                            if video.is_file() else None
                        ),
                    })
            self._json(200, {"seeds": seeds})
        elif request.path == "/api/render-status":
            request_id = parse_qs(request.query).get("id", [""])[0]
            if not request_id or request_id == "undefined":
                self._json(200, {
                    "status": "error",
                    "error": "Render status expired. Click View again.",
                })
                return
            with RENDER_LOCK:
                result = RENDER_JOBS.get(request_id)
            self._json(200 if result is not None else 202, result or {"status": "pending"})
        elif request.path == "/api/seed-video":
            try:
                path = seed_path(parse_qs(request.query).get("path", [""])[0], suffix=".mp4")
                if not path.is_file():
                    raise OSError("video missing")
            except (ValueError, OSError):
                self._json(404, {"error": "Seed video not found"})
                return
            self._video(path)
        elif request.path.startswith("/api/camera/"):
            label = request.path.removeprefix("/api/camera/").removesuffix(".jpg")
            state = live_state()
            frame = parse_qs(request.query).get("frame", [None])[0]
            if (
                request.path != f"/api/camera/{label}.jpg"
                or label not in CAMERA_LABELS
                or state.get("status") not in ("running", "awaiting_acceptance")
                or label not in state.get("cameras", [])
                or frame != str(state.get("camera_frame_step"))
            ):
                self._json(404, {"error": "Live camera frame unavailable"})
                return
            path = CAMERA_DIR / f"{label}.jpg"
            try:
                payload = path.read_bytes()
            except OSError:
                self._json(404, {"error": "Live camera frame unavailable"})
                return
            self._reply(200, payload, "image/jpeg")
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path not in (
            "/api/plans", "/api/ik-requests", "/api/preview-requests", "/api/reset-requests", "/api/object-requests",
            "/api/runner/connect", "/api/runner/disconnect", "/api/recording/accept",
            "/api/recording/start", "/api/seeds/render",
        ):
            self._json(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= (128_000 if self.path in ("/api/plans", "/api/preview-requests", "/api/recording/start") else 2_000):
                raise ValueError("Request size is invalid.")
            raw = json.loads(self.rfile.read(size))
            if self.path == "/api/runner/connect":
                self._post_runner_connect(raw)
                return
            if self.path == "/api/runner/disconnect":
                self._post_runner_disconnect(raw)
                return
            if self.path == "/api/ik-requests":
                self._post_ik_request(raw)
                return
            if self.path == "/api/preview-requests":
                self._post_preview_request(raw)
                return
            if self.path == "/api/reset-requests":
                self._post_reset_request(raw)
                return
            if self.path == "/api/object-requests":
                self._post_object_request(raw)
                return
            if self.path == "/api/recording/accept":
                self._post_accept(raw)
                return
            if self.path == "/api/recording/start":
                self._post_recording_start(raw)
                return
            if self.path == "/api/seeds/render":
                self._post_seed_render(raw)
                return
            if not isinstance(raw, dict):
                raise ValueError("Save Waypoints request must be an object.")
            name = validate_plan_filename(raw.get("filename"))
            plan = validate_plan(raw.get("plan"), discover_environments())
            PLANS.mkdir(exist_ok=True)
            path = PLANS / name
            if path.exists():
                self._json(409, {"error": f"A saved waypoint file named {name} already exists. Choose another filename."})
                return
            with path.open("x", encoding="utf-8") as stream:
                json.dump(plan, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            self._json(201, {
                "saved_path": str(path),
                "steps": sum(w["steps"] for w in plan["waypoints"]),
                "duration_s": sum(w["actual_duration_s"] for w in plan["waypoints"]),
            })
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except OSError as exc:
            self._json(500, {"error": f"Control file error: {exc}"})

    def _post_recording_start(self, raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ValueError("Seed Export request must be an object.")
        folder = validate_folder_name(raw.get("folder_name"))
        plan = validate_plan(raw.get("plan"), discover_environments())
        physics_hz, decimation = validate_control_rate(plan["control_hz"])
        if not EXPORT_LOCK.acquire(blocking=False):
            raise ValueError("Another Seed Export is starting.")
        job_id = uuid4().hex
        try:
            thread = threading.Thread(
                target=start_export_job, args=(job_id, plan, folder),
                name=f"seed-export-{job_id[:8]}", daemon=True,
            )
            thread.start()
        except Exception:
            EXPORT_LOCK.release()
            raise
        self._json(202, {
            "status": "stopping_preview", "id": job_id, "folder": folder,
            "control_hz": plan["control_hz"], "physics_hz": physics_hz, "decimation": decimation,
        })

    def _post_seed_render(self, raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ValueError("Seed viewer request must be an object.")
        relative = raw.get("path")
        dataset = seed_path(relative, suffix=".hdf5")
        if not dataset.is_file() or dataset.name.endswith(".pending.hdf5"):
            raise ValueError("Select a completed HDF5 seed.")
        video = dataset.with_name(f"{dataset.stem}_camera_previews") / "demo_0_three_cameras.mp4"
        if video.is_file():
            self._json(200, {
                "status": "ready", "dataset": relative,
                "video_url": f"/api/seed-video?path={quote(video.relative_to(DATASET_ROOT).as_posix())}",
            })
            return
        fps = 15.0
        try:
            metadata_path = dataset.with_suffix(".json")
            if not metadata_path.is_file():
                metadata_path = dataset.parent / "export.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            fps = float(metadata.get("control_hz", fps))
        except (OSError, ValueError, TypeError):
            pass
        if not math.isfinite(fps) or fps <= 0:
            fps = 15.0
        job_id = uuid4().hex
        with RENDER_LOCK:
            RENDER_JOBS[job_id] = {"status": "queued", "dataset": relative}
        threading.Thread(
            target=render_seed_job, args=(job_id, dataset, relative, fps),
            name=f"seed-render-{job_id[:8]}", daemon=True,
        ).start()
        self._json(202, {"status": "queued", "id": job_id, "dataset": relative})

    def _post_runner_connect(self, raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ValueError("Connect request must be an object.")
        environments = discover_environments()
        plan = validate_plan({
            "environment": raw.get("environment"),
            "control_hz": raw.get("control_hz", 15),
            "object_position_xyz_m": raw.get("object_position_xyz_m"),
            "preview": "head",
            "waypoints": [{"duration_s": 0.1, "memo": "UI connection hold", "q_des": {}}],
        }, environments)
        validate_control_rate(plan["control_hz"])
        with CONNECT_LOCK:
            state = live_state()
            if state.get("status") in ("running", "awaiting_acceptance", "starting"):
                if state.get("environment") == plan["environment"]:
                    self._json(200, {"status": "already_connected", "environment": plan["environment"]})
                    return
                raise ValueError(f"Disconnect the running environment first: {state.get('environment')}")
            stage_sg2_urdf()
            write_json_atomic(CONNECT_PLAN_FILE, plan)
            STOP_REQUEST_FILE.unlink(missing_ok=True)
            STOP_RESULT_FILE.unlink(missing_ok=True)
            try:
                container_plan = f"{RUNNER_LAB_ROOT}/.runtime/joint_waypoint_seed_ui/{CONNECT_PLAN_FILE.name}"
                subprocess.run([
                    "docker", "exec", "-d",
                    "-e", f"CYCLO_LAB_ROOT={RUNNER_LAB_ROOT}",
                    "-w", RUNNER_DIR,
                    RUNNER_CONTAINER,
                    RUNNER_PYTHON, "run_seed.py",
                    "--plan", container_plan, "--execute", "--headless", "--hold-final",
                ], check=True, capture_output=True, text=True, timeout=30)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                detail = getattr(exc, "stderr", "") or str(exc)
                raise ValueError(f"Could not start the preview runner: {detail.strip()}") from exc
            LAUNCH_STATE.update(environment=plan["environment"], started_at_ms=round(time.time() * 1000))
        self._json(202, {"status": "starting", "environment": plan["environment"]})

    def _post_runner_disconnect(self, raw: dict) -> None:
        if not isinstance(raw, dict) or not isinstance(raw.get("environment"), str):
            raise ValueError("Select an environment to disconnect.")
        with CONNECT_LOCK:
            state = live_state()
            if state.get("status") not in ("running", "awaiting_acceptance", "starting"):
                self._json(200, {"status": "disconnected", "environment": raw["environment"]})
                return
            if state.get("environment") != raw["environment"]:
                raise ValueError(f"The running environment differs from the selection: {state.get('environment')}")
            request = {
                "id": uuid4().hex,
                "run_id": state.get("run_id"),
                "environment": state["environment"],
                "requested_at_ms": round(time.time() * 1000),
            }
            write_json_atomic(STOP_REQUEST_FILE, request)
            LAUNCH_STATE.update(environment=None, started_at_ms=0)
        self._json(202, {"status": "stopping", "id": request["id"]})

    def _post_accept(self, raw: dict) -> None:
        state = live_state()
        if not isinstance(raw, dict) or state.get("status") != "awaiting_acceptance":
            raise ValueError("No HDF5 run is awaiting approval.")
        if not isinstance(raw.get("run_id"), str) or raw["run_id"] != state.get("run_id"):
            raise ValueError("The run ID does not match.")
        if raw.get("environment") != state.get("environment"):
            raise ValueError("The running environment differs from the selection.")
        request = {"run_id": raw["run_id"], "accepted_at_ms": round(time.time() * 1000)}
        ACCEPT_REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = ACCEPT_REQUEST_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request) + "\n", encoding="utf-8")
        os.replace(temporary, ACCEPT_REQUEST_FILE)
        self._json(202, {"status": "accepted"})

    def _post_ik_request(self, raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ValueError("IK request must be an object")
        state = live_state()
        if state.get("status") != "running" or not state.get("run_id"):
            raise ValueError("The selected environment's preview runner must be active.")
        if raw.get("environment") != state.get("environment"):
            raise ValueError("The running environment differs from the UI selection.")
        if raw.get("side") not in ("left", "right"):
            raise ValueError("Arm must be left or right.")
        if state.get("phase") not in ("ik_wait", "final_hold"):
            raise ValueError("Wait until Preview Motion finishes before calculating TCP IK.")
        solver = raw.get("solver", "iterative_lm")
        if solver not in IK_SOLVER_LABELS:
            raise ValueError(f"Unsupported IK solver: {solver}")
        xyz = raw.get("xyz_m")
        if not isinstance(xyz, list) or len(xyz) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or abs(value) > 10 for value in xyz
        ):
            raise ValueError("XYZ must contain three finite world-frame values in metres.")
        request = {
            "id": uuid4().hex,
            "run_id": state["run_id"],
            "environment": state["environment"],
            "side": raw["side"],
            "xyz_m": xyz,
            "solver": solver,
        }
        IK_REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = IK_REQUEST_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request) + "\n", encoding="utf-8")
        os.replace(temporary, IK_REQUEST_FILE)
        self._json(202, {"status": "pending", "id": request["id"]})

    def _post_preview_request(self, raw: dict) -> None:
        state = live_state()
        if (
            state.get("status") != "running"
            or state.get("phase") not in ("ik_wait", "final_hold")
            or not state.get("run_id")
        ):
            raise ValueError("The preview runner must be ready while holding its final pose.")
        plan = validate_plan(raw, discover_environments())
        validate_control_rate(plan["control_hz"])
        if plan["environment"] != state.get("environment"):
            raise ValueError("The running environment differs from the UI selection.")
        request = {
            "id": uuid4().hex,
            "run_id": state["run_id"],
            "environment": state["environment"],
            "plan": plan,
        }
        PREVIEW_REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = PREVIEW_REQUEST_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, PREVIEW_REQUEST_FILE)
        self._json(202, {
            "status": "pending",
            "id": request["id"],
            "steps": sum(waypoint["steps"] for waypoint in plan["waypoints"]),
            "duration_s": sum(waypoint["actual_duration_s"] for waypoint in plan["waypoints"]),
        })

    def _post_reset_request(self, raw: dict) -> None:
        state = live_state()
        if (
            not isinstance(raw, dict)
            or state.get("status") != "running"
            or state.get("phase") not in ("ik_wait", "final_hold")
            or not state.get("run_id")
        ):
            raise ValueError("The preview runner must be ready while holding its final pose.")
        if raw.get("environment") != state.get("environment"):
            raise ValueError("The running environment differs from the UI selection.")
        request = {
            "id": uuid4().hex,
            "run_id": state["run_id"],
            "environment": state["environment"],
        }
        RESET_REQUEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = RESET_REQUEST_FILE.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request) + "\n", encoding="utf-8")
        os.replace(temporary, RESET_REQUEST_FILE)
        self._json(202, {"status": "pending", "id": request["id"]})

    def _post_object_request(self, raw: dict) -> None:
        state = live_state()
        if (
            not isinstance(raw, dict)
            or state.get("status") != "running"
            or state.get("phase") not in ("ik_wait", "final_hold")
            or not state.get("run_id")
        ):
            raise ValueError("The environment must be connected and holding its pose.")
        if raw.get("environment") != state.get("environment"):
            raise ValueError("The running environment differs from the UI selection.")
        xyz = raw.get("xyz_m")
        if not isinstance(xyz, list) or len(xyz) != 3 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or abs(value) > 10.0 for value in xyz
        ):
            raise ValueError("Object XYZ must contain three finite world-frame values in metres.")
        request = {
            "id": uuid4().hex,
            "run_id": state["run_id"],
            "environment": state["environment"],
            "xyz_m": [float(value) for value in xyz],
        }
        write_json_atomic(OBJECT_REQUEST_FILE, request)
        self._json(202, {"status": "pending", "id": request["id"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.10.193")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Open http://{args.host}:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
