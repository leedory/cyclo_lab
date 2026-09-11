"""Mobile-base, horizontal coffee transport environment."""

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ....platform.action_cfg import ContinuousShowroomActionsCfg
from ...common import EpisodicShowroomTaskEnvCfg
from ..coffee_transport_common import (
    coffee_transport_success,
)
from ..environment import (
    add_coffee_transport_cans,
    configure_coffee_randomization_events,
    configure_coffee_transport_scene,
    make_episode_home_reset_event,
)
from .profiles import (
    TASK000002_DETERMINISTIC,
    TASK000002_RECORD_RANDOMIZED,
    Task000002RandomizationCfg,
    validate_task000002_randomization_cfg,
)
from .spec import TASK_000002_SPEC


@configclass
class Task000002EnvCfg(EpisodicShowroomTaskEnvCfg):
    """22D horizontal transport with a HOME-relative initial route pose."""

    task_spec = TASK_000002_SPEC
    randomization: Task000002RandomizationCfg = TASK000002_DETERMINISTIC
    actions = ContinuousShowroomActionsCfg()

    def __post_init__(self):
        add_coffee_transport_cans(self.scene)
        super().__post_init__()

        configure_coffee_transport_scene(self.scene)
        self.actions.base_action.linear_deadband = 0.01
        self.actions.base_action.angular_deadband = 0.01
        validate_task000002_randomization_cfg(self.randomization)
        configure_coffee_randomization_events(self.events, self.randomization)
        self.events.reset_episode_home_and_base_start = make_episode_home_reset_event(
            self.randomization.episode_home,
            base_route_start=self.randomization.base_route_start,
        )

        self.terminations.success = DoneTerm(
            func=coffee_transport_success,
            params={"destination": "right"},
        )


@configclass
class Task000002RandomEnvCfg(Task000002EnvCfg):
    """Recorder preset with reviewed HOME-relative route randomization enabled."""

    env_name = "Cyclo-Real-Showroom-Task000002-Random-FFW-SG2-v0"
    randomization: Task000002RandomizationCfg = TASK000002_RECORD_RANDOMIZED

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
