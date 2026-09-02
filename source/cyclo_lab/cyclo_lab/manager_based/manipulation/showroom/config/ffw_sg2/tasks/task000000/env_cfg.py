"""Task000000의 deterministic/random 실행 환경 템플릿."""

from isaaclab.utils import configclass

from ...randomization.cfg import ShowroomRandomizationCfg
from ..common import EpisodicShowroomTaskEnvCfg, ShowroomTaskSpec
from .profiles import (
    TASK000000_RECORD_DETERMINISTIC,
    TASK000000_RECORD_RANDOM,
)
from .spec import TASK_000000_SPEC


@configclass
class Task000000EnvCfg(EpisodicShowroomTaskEnvCfg):
    """Task000000의 재현 가능한 기본 환경."""

    task_spec: ShowroomTaskSpec = TASK_000000_SPEC
    randomization: ShowroomRandomizationCfg = TASK000000_RECORD_DETERMINISTIC

    def __post_init__(self):
        # task 전용 대상 물체를 새로 만드는 경우에는 super() 호출 전에
        # self.scene에 넣어야 공통 코드의 target_object 검사를 통과한다.
        super().__post_init__()

        # 공통 scene 구성이 끝난 뒤에만 가능한 task 전용 조정은 여기에 둔다.
        # 규모가 커지면 object_cfg.py, success_terms.py처럼 역할별 파일로 분리한다.


@configclass
class Task000000RandomEnvCfg(Task000000EnvCfg):
    """기본 환경에 검토된 reset randomization을 적용한 환경."""

    env_name: str = "Cyclo-Real-Showroom-Task000000-Random-FFW-SG2-v0"
    randomization: ShowroomRandomizationCfg = TASK000000_RECORD_RANDOM

    def __post_init__(self):
        # 공통 초기화가 spec의 기본 env_name을 넣으므로 random ID를 잠시 보관한다.
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
