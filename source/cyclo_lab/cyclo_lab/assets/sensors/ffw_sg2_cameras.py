"""FFW SG2 observation camera configurations."""

from __future__ import annotations


FFW_SG2_HEAD_CAMERA_NAME = "cam_head"
FFW_SG2_WRIST_LEFT_CAMERA_NAME = "cam_wrist_left"
FFW_SG2_WRIST_RIGHT_CAMERA_NAME = "cam_wrist_right"
FFW_SG2_OVERHEAD_LEFT_CAMERA_NAME = "cam_overhead_left"
FFW_SG2_OVERHEAD_CENTER_CAMERA_NAME = "cam_overhead_center"
FFW_SG2_OVERHEAD_RIGHT_CAMERA_NAME = "cam_overhead_right"

FFW_SG2_HEAD_CAMERA_HEIGHT = 720
FFW_SG2_HEAD_CAMERA_WIDTH = 1280
FFW_SG2_WRIST_CAMERA_HEIGHT = 640
FFW_SG2_WRIST_CAMERA_WIDTH = 480
FFW_SG2_OVERHEAD_CAMERA_HEIGHT = 512
FFW_SG2_OVERHEAD_CAMERA_WIDTH = 512

# Align ROS optical +Z with link +X while keeping the rendered image upright.
# The portrait 480x640 raster is the quarter-turned form of the native sideways
# 640x480 image, preserving its full framing without downstream rotation.
FFW_SG2_D405_CAMERA_UPRIGHT_ROT = (0.7071068, 0.0, 0.7071068, 0.0)
FFW_SG2_CAMERA_HORIZONTAL_APERTURE = 20.955
FFW_SG2_OVERHEAD_LEFT_CAMERA_POS = (-0.10, 0.40, 1.8)
FFW_SG2_OVERHEAD_LEFT_CAMERA_ROT = (0.8535534, 0.1464466, 0.3535534, -0.3535534)
FFW_SG2_OVERHEAD_CENTER_CAMERA_POS = (0.1, 0.0, 2.0)
FFW_SG2_OVERHEAD_CENTER_CAMERA_ROT = (0.7071068, 0.0, 0.7071068, 0.0)
FFW_SG2_OVERHEAD_RIGHT_CAMERA_POS = (-0.10, -0.40, 1.8)
FFW_SG2_OVERHEAD_RIGHT_CAMERA_ROT = (0.8535534, -0.1464466, 0.3535534, 0.3535534)


def camera_publish_period(publish_hz: float) -> float:
    return 1.0 / publish_hz if publish_hz > 0.0 else 0.0


def _make_pinhole_camera_cfg(
    sim_utils,
    *,
    focal_length: float,
    width: int,
    height: int,
    focus_distance: float,
    clipping_range: tuple[float, float],
    intrinsic_matrix: tuple[float, ...] | None,
):
    if intrinsic_matrix is None:
        return sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=focus_distance,
            horizontal_aperture=FFW_SG2_CAMERA_HORIZONTAL_APERTURE,
            clipping_range=clipping_range,
        )

    if len(intrinsic_matrix) != 9:
        raise ValueError(f"Camera intrinsic matrix must contain 9 values, got {len(intrinsic_matrix)}.")
    fx = float(intrinsic_matrix[0])
    fy = float(intrinsic_matrix[4])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Camera focal lengths must be positive, got fx={fx}, fy={fy}.")

    # Omniverse's standard pinhole renderer supports one focal scale and a
    # centered optical axis. Preserve the measured mean focal scale/FOV while
    # retaining the full source K in the selected robot profile for provenance.
    mean_focal_px = 0.5 * (fx + fy)
    matched_focal_length = mean_focal_px * FFW_SG2_CAMERA_HORIZONTAL_APERTURE / float(width)
    return sim_utils.PinholeCameraCfg(
        focal_length=matched_focal_length,
        focus_distance=focus_distance,
        horizontal_aperture=FFW_SG2_CAMERA_HORIZONTAL_APERTURE,
        vertical_aperture=FFW_SG2_CAMERA_HORIZONTAL_APERTURE * float(height) / float(width),
        clipping_range=clipping_range,
    )


def make_ffw_sg2_head_camera_cfg(
    *,
    update_period: float = 0.0,
    height: int = FFW_SG2_HEAD_CAMERA_HEIGHT,
    width: int = FFW_SG2_HEAD_CAMERA_WIDTH,
    clipping_range: tuple[float, float] = (0.01, 8.0),
    intrinsic_matrix: tuple[float, ...] | None = None,
):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/head_link2/zed/cam_head",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=_make_pinhole_camera_cfg(
            sim_utils,
            focal_length=10.4,
            width=width,
            height=height,
            focus_distance=200.0,
            clipping_range=clipping_range,
            intrinsic_matrix=intrinsic_matrix,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.03, 0.0),
            rot=(0.5, 0.5, -0.5, -0.5),
            convention="isaac",
        ),
    )


def make_ffw_sg2_wrist_camera_cfg(
    side: str,
    *,
    update_period: float = 0.0,
    height: int = FFW_SG2_WRIST_CAMERA_HEIGHT,
    width: int = FFW_SG2_WRIST_CAMERA_WIDTH,
    clipping_range: tuple[float, float] = (0.01, 8.0),
    intrinsic_matrix: tuple[float, ...] | None = None,
):
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    if side not in ("left", "right"):
        raise ValueError(f"Unsupported wrist camera side: {side}")

    side_prefix = "l" if side == "left" else "r"
    camera_name = FFW_SG2_WRIST_LEFT_CAMERA_NAME if side == "left" else FFW_SG2_WRIST_RIGHT_CAMERA_NAME
    arm_link_name = f"arm_{side_prefix}_link7"
    camera_frame_name = f"camera_{side_prefix}_bottom_screw_frame"
    camera_link_name = f"camera_{side_prefix}_link"

    return CameraCfg(
        prim_path=(
            "{ENV_REGEX_NS}/Robot/ffw_sg2_follower/"
            f"{arm_link_name}/{camera_frame_name}/{camera_link_name}/{camera_name}"
        ),
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=_make_pinhole_camera_cfg(
            sim_utils,
            focal_length=18.0,
            width=width,
            height=height,
            focus_distance=400.0,
            clipping_range=clipping_range,
            intrinsic_matrix=intrinsic_matrix,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=FFW_SG2_D405_CAMERA_UPRIGHT_ROT,
            convention="ros",
        ),
    )


def make_ffw_sg2_overhead_camera_cfg(
    side: str,
    *,
    update_period: float = 0.0,
    height: int = FFW_SG2_OVERHEAD_CAMERA_HEIGHT,
    width: int = FFW_SG2_OVERHEAD_CAMERA_WIDTH,
    clipping_range: tuple[float, float] = (0.05, 10.0),
):
    """Create a robot-following preview camera above one side of the SG2 base."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    if side == "left":
        camera_name = FFW_SG2_OVERHEAD_LEFT_CAMERA_NAME
        position = FFW_SG2_OVERHEAD_LEFT_CAMERA_POS
        rotation = FFW_SG2_OVERHEAD_LEFT_CAMERA_ROT
    elif side == "center":
        camera_name = FFW_SG2_OVERHEAD_CENTER_CAMERA_NAME
        position = FFW_SG2_OVERHEAD_CENTER_CAMERA_POS
        rotation = FFW_SG2_OVERHEAD_CENTER_CAMERA_ROT
    elif side == "right":
        camera_name = FFW_SG2_OVERHEAD_RIGHT_CAMERA_NAME
        position = FFW_SG2_OVERHEAD_RIGHT_CAMERA_POS
        rotation = FFW_SG2_OVERHEAD_RIGHT_CAMERA_ROT
    else:
        raise ValueError(f"Unsupported overhead camera side: {side}")

    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/ffw_sg2_follower/world/{camera_name}",
        update_period=update_period,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=300.0,
            horizontal_aperture=20.955,
            clipping_range=clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=position,
            rot=rotation,
            convention="world",
        ),
    )
