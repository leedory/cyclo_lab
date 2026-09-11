"""FFW F2 RGB camera configurations from the canonical URDF mounts."""

from __future__ import annotations


FFW_F2_HEAD_CAMERA_NAME = "cam_head"
FFW_F2_WRIST_LEFT_CAMERA_NAME = "cam_wrist_left"
FFW_F2_WRIST_RIGHT_CAMERA_NAME = "cam_wrist_right"
FFW_F2_OVERHEAD_LEFT_CAMERA_NAME = "cam_overhead_left"
FFW_F2_OVERHEAD_CENTER_CAMERA_NAME = "cam_overhead_center"
FFW_F2_OVERHEAD_RIGHT_CAMERA_NAME = "cam_overhead_right"
FFW_F2_CAMERA_HEIGHT = 480
FFW_F2_CAMERA_WIDTH = 640
FFW_F2_OVERHEAD_CAMERA_HEIGHT = 512
FFW_F2_OVERHEAD_CAMERA_WIDTH = 512
FFW_F2_CAMERA_HORIZONTAL_APERTURE = 20.955

_F2_HEAD_CAMERA_ROT = (0.5, 0.5, -0.5, -0.5)
_F2_WRIST_CAMERA_ROT = (0.5, -0.5, -0.5, 0.5)
_F2_OVERHEAD_POSES = {
    "left": ((-0.10, 0.40, 1.8), (0.8535534, 0.1464466, 0.3535534, -0.3535534)),
    "center": ((0.1, 0.0, 2.0), (0.7071068, 0.0, 0.7071068, 0.0)),
    "right": ((-0.10, -0.40, 1.8), (0.8535534, -0.1464466, 0.3535534, 0.3535534)),
}


def _make_pinhole_camera_cfg(
    sim_utils,
    *,
    width: int,
    height: int,
    clipping_range: tuple[float, float],
    intrinsic_matrix: tuple[float, ...] | None,
):
    focal_length = 18.0
    if intrinsic_matrix is not None:
        if len(intrinsic_matrix) != 9:
            raise ValueError("Camera intrinsic matrix must contain 9 values.")
        fx, fy = float(intrinsic_matrix[0]), float(intrinsic_matrix[4])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("Camera focal lengths must be positive.")
        focal_length = 0.5 * (fx + fy) * FFW_F2_CAMERA_HORIZONTAL_APERTURE / float(width)
    return sim_utils.PinholeCameraCfg(
        focal_length=focal_length,
        focus_distance=400.0,
        horizontal_aperture=FFW_F2_CAMERA_HORIZONTAL_APERTURE,
        vertical_aperture=FFW_F2_CAMERA_HORIZONTAL_APERTURE * float(height) / float(width),
        clipping_range=clipping_range,
    )


def make_ffw_f2_head_camera_cfg(
    *,
    update_period: float = 0.0,
    height: int = FFW_F2_CAMERA_HEIGHT,
    width: int = FFW_F2_CAMERA_WIDTH,
    clipping_range: tuple[float, float] = (0.01, 8.0),
    intrinsic_matrix: tuple[float, ...] | None = None,
):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_head_link/cam_head",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=_make_pinhole_camera_cfg(
            sim_utils, width=width, height=height,
            clipping_range=clipping_range, intrinsic_matrix=intrinsic_matrix,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=_F2_HEAD_CAMERA_ROT, convention="isaac"
        ),
    )


def make_ffw_f2_wrist_camera_cfg(
    side: str,
    *,
    update_period: float = 0.0,
    height: int = FFW_F2_CAMERA_HEIGHT,
    width: int = FFW_F2_CAMERA_WIDTH,
    clipping_range: tuple[float, float] = (0.01, 8.0),
    intrinsic_matrix: tuple[float, ...] | None = None,
):
    """Create an F2 wrist camera; optical alignment still needs real-robot validation."""

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    if side not in ("left", "right"):
        raise ValueError(f"Unsupported wrist camera side: {side}")
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/camera_{side}_link/cam_wrist_{side}",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=_make_pinhole_camera_cfg(
            sim_utils, width=width, height=height,
            clipping_range=clipping_range, intrinsic_matrix=intrinsic_matrix,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), rot=_F2_WRIST_CAMERA_ROT, convention="isaac"
        ),
    )


def make_ffw_f2_overhead_camera_cfg(
    side: str,
    *,
    update_period: float = 0.0,
    height: int = FFW_F2_OVERHEAD_CAMERA_HEIGHT,
    width: int = FFW_F2_OVERHEAD_CAMERA_WIDTH,
):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    try:
        position, rotation = _F2_OVERHEAD_POSES[side]
    except KeyError as exc:
        raise ValueError(f"Unsupported overhead camera side: {side}") from exc
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/base_link/cam_overhead_{side}",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=300.0,
            horizontal_aperture=FFW_F2_CAMERA_HORIZONTAL_APERTURE,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=position, rot=rotation, convention="world"),
    )
