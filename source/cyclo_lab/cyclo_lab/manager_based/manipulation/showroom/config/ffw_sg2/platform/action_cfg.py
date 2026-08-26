"""Shared Continuous and episodic SG2 showroom action configurations."""

from isaaclab.utils import configclass

from cyclo_lab.manager_based.actions import (
    FFWSG2JointPositionActionsCfg,
    FFWSG2MobileActionsCfg,
)


@configclass
class ContinuousShowroomActionsCfg(FFWSG2MobileActionsCfg):
    """Canonical 19 SG2 joint targets followed by 3D base velocity."""


@configclass
class EpisodicShowroomActionsCfg(FFWSG2JointPositionActionsCfg):
    """Canonical fixed-base 19D SG2 joint action."""
