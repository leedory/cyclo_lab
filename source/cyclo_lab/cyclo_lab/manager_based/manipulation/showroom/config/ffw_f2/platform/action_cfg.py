"""Action configuration for the FFW F2 Continuous showroom template."""

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from cyclo_lab.manager_based.actions.swerve_base_action import SwerveBaseVelocityActionCfg
from cyclo_lab.robot_specs.ffw.f2 import (
    F2_SWERVE_MODULE_ANGLE_OFFSETS,
    F2_SWERVE_MODULE_X_OFFSETS,
    F2_SWERVE_MODULE_Y_OFFSETS,
    F2_SWERVE_STEERING_JOINTS,
    F2_SWERVE_WHEEL_JOINTS,
    F2_SWERVE_WHEEL_RADIUS,
    FFW_F2_HEAD_JOINT_NAMES,
    FFW_F2_LEFT_ARM_JOINT_NAMES,
    FFW_F2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_F2_LIFT_JOINT_NAMES,
    FFW_F2_RIGHT_ARM_JOINT_NAMES,
    FFW_F2_RIGHT_GRIPPER_JOINT_NAMES,
    FFW_F2_SWERVE_ANGULAR_ACCELERATION_LIMIT,
    FFW_F2_SWERVE_DRIVE_SPEED_SCALE,
    FFW_F2_SWERVE_ENABLED_SPEED_LIMITS,
    FFW_F2_SWERVE_ENABLED_WHEEL_SATURATION_SCALING,
    FFW_F2_SWERVE_LINEAR_ACCELERATION_LIMIT,
    FFW_F2_SWERVE_STEERING_ALIGNMENT_ANGLE_ERROR_THRESHOLD,
    FFW_F2_SWERVE_STEERING_ALIGNMENT_START_ANGLE_ERROR_THRESHOLD,
    FFW_F2_SWERVE_STEERING_ALIGNMENT_START_SPEED_ERROR_THRESHOLD,
    FFW_F2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
    FFW_F2_SWERVE_STEERING_LIMIT_LOWER,
    FFW_F2_SWERVE_STEERING_LIMIT_UPPER,
    FFW_F2_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
    FFW_F2_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
)


def _joint_action(joint_names: tuple[str, ...]) -> JointPositionActionCfg:
    return JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(joint_names),
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
    )


def _swerve_base_action() -> SwerveBaseVelocityActionCfg:
    return SwerveBaseVelocityActionCfg(
        asset_name="robot",
        steering_joint_names=F2_SWERVE_STEERING_JOINTS,
        wheel_joint_names=F2_SWERVE_WHEEL_JOINTS,
        module_x_offsets=F2_SWERVE_MODULE_X_OFFSETS,
        module_y_offsets=F2_SWERVE_MODULE_Y_OFFSETS,
        module_angle_offsets=F2_SWERVE_MODULE_ANGLE_OFFSETS,
        wheel_radius=F2_SWERVE_WHEEL_RADIUS,
        steering_limit_lower=FFW_F2_SWERVE_STEERING_LIMIT_LOWER,
        steering_limit_upper=FFW_F2_SWERVE_STEERING_LIMIT_UPPER,
        wheel_speed_limit_lower=FFW_F2_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
        wheel_speed_limit_upper=FFW_F2_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
        steering_angular_velocity_limit=FFW_F2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
        enabled_speed_limits=FFW_F2_SWERVE_ENABLED_SPEED_LIMITS,
        linear_acceleration_limit=FFW_F2_SWERVE_LINEAR_ACCELERATION_LIMIT,
        angular_acceleration_limit=FFW_F2_SWERVE_ANGULAR_ACCELERATION_LIMIT,
        steering_alignment_angle_error_threshold=(
            FFW_F2_SWERVE_STEERING_ALIGNMENT_ANGLE_ERROR_THRESHOLD
        ),
        steering_alignment_start_angle_error_threshold=(
            FFW_F2_SWERVE_STEERING_ALIGNMENT_START_ANGLE_ERROR_THRESHOLD
        ),
        steering_alignment_start_speed_error_threshold=(
            FFW_F2_SWERVE_STEERING_ALIGNMENT_START_SPEED_ERROR_THRESHOLD
        ),
        enabled_wheel_saturation_scaling=(
            FFW_F2_SWERVE_ENABLED_WHEEL_SATURATION_SCALING
        ),
        drive_speed_scale=FFW_F2_SWERVE_DRIVE_SPEED_SCALE,
    )


@configclass
class ContinuousShowroomActionsCfg:
    """Canonical 19 F2 joint targets followed by 3D body velocity."""

    arm_l_action: JointPositionActionCfg = _joint_action(FFW_F2_LEFT_ARM_JOINT_NAMES)
    gripper_l_action: JointPositionActionCfg = _joint_action(FFW_F2_LEFT_GRIPPER_JOINT_NAMES)
    arm_r_action: JointPositionActionCfg = _joint_action(FFW_F2_RIGHT_ARM_JOINT_NAMES)
    gripper_r_action: JointPositionActionCfg = _joint_action(FFW_F2_RIGHT_GRIPPER_JOINT_NAMES)
    head_action: JointPositionActionCfg = _joint_action(FFW_F2_HEAD_JOINT_NAMES)
    lift_action: JointPositionActionCfg = _joint_action(FFW_F2_LIFT_JOINT_NAMES)
    base_action: SwerveBaseVelocityActionCfg = _swerve_base_action()
