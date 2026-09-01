"""Canonical public data contracts for the mobile FFW SG2.

The normal Task525 contract is joint19 + body-velocity3.  The local
locomanipulation SDG adapter also has 22 values, but its arm fields are EEF
poses.  Keeping those contracts separate prevents silent training corruption.
"""

from __future__ import annotations

from .joints import FFW_SG2_ACTION_JOINT_NAMES


FFW_SG2_BASE_ACTION_NAMES = ("linear_x", "linear_y", "angular_z")
FFW_SG2_EEF_POSE_COMPONENTS = ("x", "y", "z", "qw", "qx", "qy", "qz")
FFW_SG2_SDG_ACTION_NAMES = (
    *(f"left_eef_{component}_robot_root" for component in FFW_SG2_EEF_POSE_COMPONENTS),
    "gripper_l_joint1",
    *(f"right_eef_{component}_robot_root" for component in FFW_SG2_EEF_POSE_COMPONENTS),
    "gripper_r_joint1",
    "head_joint1",
    "head_joint2",
    "lift_joint",
    *FFW_SG2_BASE_ACTION_NAMES,
)
FFW_SG2_TASK525_HYBRID_ACTION_NAMES = (
    *FFW_SG2_ACTION_JOINT_NAMES[:8],
    *(f"right_eef_{component}_robot_root" for component in FFW_SG2_EEF_POSE_COMPONENTS),
    "gripper_r_joint1",
    "head_joint1",
    "head_joint2",
    "lift_joint",
    *FFW_SG2_BASE_ACTION_NAMES,
)

FFW_SG2_MOBILE_ACTION_NAMES = (
    *FFW_SG2_ACTION_JOINT_NAMES,
    *FFW_SG2_BASE_ACTION_NAMES,
)

FFW_SG2_JOINT_ACTION_UNITS = ("rad",) * 18 + ("m",)
FFW_SG2_BASE_ACTION_UNITS = ("m/s", "m/s", "rad/s")
FFW_SG2_EEF_POSE_UNITS = (
    "m",
    "m",
    "m",
    "unitless",
    "unitless",
    "unitless",
    "unitless",
)
FFW_SG2_SDG_ACTION_UNITS = (
    *FFW_SG2_EEF_POSE_UNITS,
    "rad",
    *FFW_SG2_EEF_POSE_UNITS,
    "rad",
    "rad",
    "rad",
    "m",
    *FFW_SG2_BASE_ACTION_UNITS,
)
FFW_SG2_TASK525_HYBRID_ACTION_UNITS = (
    *("rad",) * 8,
    *FFW_SG2_EEF_POSE_UNITS,
    "rad",
    "rad",
    "rad",
    "m",
    *FFW_SG2_BASE_ACTION_UNITS,
)

FFW_SG2_MOBILE_ACTION_UNITS = (
    *FFW_SG2_JOINT_ACTION_UNITS,
    *FFW_SG2_BASE_ACTION_UNITS,
)

FFW_SG2_JOINT_ACTION_DIM = len(FFW_SG2_ACTION_JOINT_NAMES)
FFW_SG2_BASE_ACTION_DIM = len(FFW_SG2_BASE_ACTION_NAMES)
FFW_SG2_MOBILE_ACTION_DIM = len(FFW_SG2_MOBILE_ACTION_NAMES)
FFW_SG2_SDG_ACTION_DIM = len(FFW_SG2_SDG_ACTION_NAMES)


def converted_action_contract_metadata(
    action_representation: str, action_dim: int
) -> dict[str, object]:
    """Describe an SG2 action tensor produced by an offline converter."""

    if action_representation not in ("joint", "ik"):
        raise ValueError(
            "action_representation must be 'joint' or 'ik', "
            f"got {action_representation!r}"
        )
    if action_dim not in (FFW_SG2_JOINT_ACTION_DIM, FFW_SG2_MOBILE_ACTION_DIM):
        raise ValueError(f"SG2 converted actions must be 19D or 22D, got {action_dim}D")

    mobile = action_dim == FFW_SG2_MOBILE_ACTION_DIM
    if action_representation == "joint":
        return {
            "robot_contract_id": (
                "ffw_sg2_rev1_mobile_22d_v1"
                if mobile
                else "ffw_sg2_rev1_fixed_base_19d_v1"
            ),
            "action_names": list(
                FFW_SG2_MOBILE_ACTION_NAMES
                if mobile
                else FFW_SG2_ACTION_JOINT_NAMES
            ),
            "action_units": list(
                FFW_SG2_MOBILE_ACTION_UNITS
                if mobile
                else FFW_SG2_JOINT_ACTION_UNITS
            ),
            "action_semantics": (
                "pre_step_joint_position_19_plus_body_velocity_3"
                if mobile
                else "pre_step_absolute_joint_position_command"
            ),
            "eef_action_frame": "none",
            "mimic_trajectory_source": "none",
        }

    return {
        "robot_contract_id": (
            "ffw_sg2_task525_locomanipulation_sdg_eef22_v1"
            if mobile
            else "ffw_sg2_mimic_dual_eef19_v1"
        ),
        "action_names": list(FFW_SG2_SDG_ACTION_NAMES[:action_dim]),
        "action_units": list(FFW_SG2_SDG_ACTION_UNITS[:action_dim]),
        "action_semantics": (
            "observed_achieved_dual_eef_pose16_plus_passive_joint3"
            + ("_plus_body_velocity3" if mobile else "")
        ),
        "eef_action_frame": "robot_root",
        "mimic_trajectory_source": "achieved_eef_pose",
    }


def is_mobile_action_cfg(actions_cfg) -> bool:
    """Return whether an Isaac Lab action cfg owns the 3D base action term."""

    return getattr(actions_cfg, "base_action", None) is not None


def sdg_ik_sides(actions_cfg) -> tuple[bool, bool]:
    """Return whether the left/right arm terms accept 7D EEF poses."""

    return tuple(
        type(getattr(actions_cfg, f"arm_{side}_action", None))
        .__name__
        .endswith("DifferentialInverseKinematicsActionCfg")
        for side in ("l", "r")
    )


def is_sdg_ik_action_cfg(actions_cfg) -> bool:
    """Return whether either arm term accepts a 7D EEF pose."""

    return any(sdg_ik_sides(actions_cfg))


def hdf5_contract_metadata(actions_cfg) -> dict[str, object]:
    """Build shape/name/unit/frame metadata from an environment action cfg."""

    mobile = is_mobile_action_cfg(actions_cfg)
    sdg_ik = is_sdg_ik_action_cfg(actions_cfg)
    ik_left, ik_right = sdg_ik_sides(actions_cfg)
    if mobile and ik_left and ik_right:
        action_names = FFW_SG2_SDG_ACTION_NAMES
        action_units = FFW_SG2_SDG_ACTION_UNITS
        state_names = FFW_SG2_MOBILE_ACTION_NAMES
        state_units = FFW_SG2_MOBILE_ACTION_UNITS
        contract_id = "ffw_sg2_task525_locomanipulation_sdg_eef22_v1"
        action_semantics = (
            "pre_step_dual_eef_pose16_plus_passive_joint3_plus_body_velocity3"
        )
        state_semantics = "measured_joint_position_19_plus_body_velocity_3"
    elif mobile and ik_right and not ik_left:
        action_names = FFW_SG2_TASK525_HYBRID_ACTION_NAMES
        action_units = FFW_SG2_TASK525_HYBRID_ACTION_UNITS
        state_names = FFW_SG2_MOBILE_ACTION_NAMES
        state_units = FFW_SG2_MOBILE_ACTION_UNITS
        contract_id = "ffw_sg2_task525_locomanipulation_sdg_eef_hybrid22_v1"
        action_semantics = (
            "pre_step_left_joint8_plus_right_eef_pose7_gripper1_plus_passive_joint3_plus_body_velocity3"
        )
        state_semantics = "measured_joint_position_19_plus_body_velocity_3"
    elif mobile:
        action_names = state_names = FFW_SG2_MOBILE_ACTION_NAMES
        action_units = state_units = FFW_SG2_MOBILE_ACTION_UNITS
        contract_id = "ffw_sg2_rev1_mobile_22d_v1"
        action_semantics = "pre_step_joint_position_19_plus_body_velocity_3"
        state_semantics = "measured_joint_position_19_plus_body_velocity_3"
    else:
        action_names = state_names = FFW_SG2_ACTION_JOINT_NAMES
        action_units = state_units = FFW_SG2_JOINT_ACTION_UNITS
        contract_id = "ffw_sg2_rev1_fixed_base_19d_v1"
        action_semantics = "pre_step_absolute_joint_position_command"
        state_semantics = "measured_joint_position_19"

    return {
        "robot_contract_id": contract_id,
        "action_names": list(action_names),
        "observation_state_names": list(state_names),
        "action_units": list(action_units),
        "observation_state_units": list(state_units),
        "action_semantics": action_semantics,
        "observation_state_semantics": state_semantics,
        "joint_order": "arm_l7,gripper_l,arm_r7,gripper_r,head2,lift",
        "base_action_frame": "robot_body" if mobile else "none",
        "base_state_frame": "robot_body" if mobile else "none",
        "eef_action_frame": "robot_root" if sdg_ik else "none",
        "observation_state_paths": (
            ["obs/joint_pos", "obs/base_velocity_body"]
            if mobile
            else ["obs/joint_pos"]
        ),
    }
