"""Rigid ivory destination mat asset."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from cyclo_lab.assets.object import CYCLO_LAB_OBJECT_ASSETS_DATA_DIR


IVORY_TABLE_MAT_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{CYCLO_LAB_OBJECT_ASSETS_DATA_DIR}/object/ivory_table_mat.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=False,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.001),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)
