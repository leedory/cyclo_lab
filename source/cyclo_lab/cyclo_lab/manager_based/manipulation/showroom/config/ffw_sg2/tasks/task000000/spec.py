"""Task000000의 정체성과 공통 실행 조건을 적는 템플릿."""

from cyclo_lab.assets.environments.robotis_showroom import (
    ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
)
from cyclo_lab.robot_specs.ffw.sg2 import DEFAULT_SG2_CAMERA_PROFILE

from ..common import ShowroomTaskSpec


# scene cfg에 등록된 Python 속성 이름을 정확히 적는다.
# 예: "peanut_mix_bag_02", "coffee_can_green"
TASK_000000_TARGET_OBJECT = "<scene_object_name>"

TASK_000000_SPEC = ShowroomTaskSpec(
    # 번호는 항상 여섯 자리 문자열로 둔다. task 525라면 "000525"다.
    task_id="000000",
    env_name="Cyclo-Real-Showroom-Task000000-FFW-SG2-v0",
    # 데이터와 policy가 실제로 수행해야 할 행동을 짧고 구체적으로 적는다.
    instruction="<describe the task in one clear sentence>",
    target_object=TASK_000000_TARGET_OBJECT,
    # 주로 조작할 팔. 허용 값은 "left" 또는 "right"다.
    target_side="<left_or_right>",
    environment_usd_path=ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
    robot_profile=DEFAULT_SG2_CAMERA_PROFILE,
    # 기본값은 control/camera 15 Hz, physics 30 Hz, episode 120초다.
    # 다른 값이 꼭 필요할 때만 control_hz, physics_hz 등을 여기서 덮어쓴다.
)
