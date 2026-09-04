"""Mobile-base, horizontal coffee transport environment."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.robotis_showroom import iter_robotis_showroom_object_cfgs

from ....platform.action_cfg import ContinuousShowroomActionsCfg
from ....randomization.cfg import ShowroomRandomizationCfg
from ...common import EpisodicShowroomTaskEnvCfg
from ..coffee_transport_common import (
    COFFEE_TRANSPORT_HOME_POS,
    COFFEE_TRANSPORT_HOME_ROT,
    SOURCE_SPAWN_SQUARES,
    coffee_transport_success,
    iter_static_task_jelly_bag_cfgs,
    make_orange_coffee_can_cfg,
    reset_horizontal_coffee_transport,
)
from .spec import TASK_000002_SPEC


@configclass
class Task000002EnvCfg(EpisodicShowroomTaskEnvCfg):
    """22D task: base starts uniformly along the route; lift and head stay fixed."""

    task_spec = TASK_000002_SPEC
    randomization = ShowroomRandomizationCfg()
    actions = ContinuousShowroomActionsCfg()

    def __post_init__(self):
        for square in SOURCE_SPAWN_SQUARES:
            setattr(self.scene, square.name, make_orange_coffee_can_cfg(square))
        super().__post_init__()
        self.scene.robot.init_state.pos = COFFEE_TRANSPORT_HOME_POS
        self.scene.robot.init_state.rot = COFFEE_TRANSPORT_HOME_ROT
        self.actions.base_action.linear_deadband = 0.01
        self.actions.base_action.angular_deadband = 0.01
        for object_name, _ in iter_robotis_showroom_object_cfgs():
            if hasattr(self.scene, object_name):
                delattr(self.scene, object_name)
        for object_name, object_cfg in iter_static_task_jelly_bag_cfgs():
            setattr(self.scene, object_name, object_cfg)
        self.events.reset_task000002 = EventTerm(func=reset_horizontal_coffee_transport, mode="reset")
        self.terminations.success = DoneTerm(func=coffee_transport_success, params={"destination": "right"})


@configclass
class Task000002RandomEnvCfg(Task000002EnvCfg):
    """Alias retained for recorder tooling; every reset already samples its task axes."""

    env_name = "Cyclo-Real-Showroom-Task000002-Random-FFW-SG2-v0"

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
