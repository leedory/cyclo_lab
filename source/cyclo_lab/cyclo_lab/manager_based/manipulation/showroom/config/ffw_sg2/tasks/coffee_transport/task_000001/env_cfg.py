"""Fixed-base, vertical coffee transport environment."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

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
from ..reset_events import (
    randomize_lift_start,
)
from .profiles import (
    TASK000001_DETERMINISTIC,
    TASK000001_RECORD_RANDOMIZED,
    Task000001RandomizationCfg,
    validate_task000001_randomization_cfg,
)
from .spec import TASK_000001_SPEC


@configclass
class Task000001EnvCfg(EpisodicShowroomTaskEnvCfg):
    """19D vertical transport with profile-owned reset randomization."""

    task_spec = TASK_000001_SPEC
    randomization: Task000001RandomizationCfg = TASK000001_DETERMINISTIC

    def __post_init__(self):
        add_coffee_transport_cans(self.scene)
        super().__post_init__()

        configure_coffee_transport_scene(self.scene)
        validate_task000001_randomization_cfg(self.randomization)
        configure_coffee_randomization_events(self.events, self.randomization)
        self.events.reset_episode_home = make_episode_home_reset_event(
            self.randomization.episode_home
        )

        lift_start = self.randomization.lift_start
        if lift_start.enabled:
            self.events.randomize_lift_start = EventTerm(
                func=randomize_lift_start,
                mode="reset",
                params={"position_range_m": lift_start.position_range_m},
            )

        self.terminations.success = DoneTerm(
            func=coffee_transport_success,
            params={"destination": "down"},
        )


@configclass
class Task000001RandomEnvCfg(Task000001EnvCfg):
    """Recorder preset with reviewed physical reset randomization enabled."""

    env_name = "Cyclo-Real-Showroom-Task000001-Random-FFW-SG2-v0"
    randomization: Task000001RandomizationCfg = TASK000001_RECORD_RANDOMIZED

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
