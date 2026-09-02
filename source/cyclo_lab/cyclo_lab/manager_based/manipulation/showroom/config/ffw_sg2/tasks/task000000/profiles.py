"""Task000000에서 선택할 reset randomization 템플릿."""

import math

from ...randomization.cfg import (
    CameraRandomizationCfg,
    LightingRandomizationCfg,
    ObjectPoseRandomizationCfg,
    PresenceRandomizationCfg,
    RobotRootRandomizationCfg,
    ShelfAppearanceRandomizationCfg,
    ShowroomGenerationRandomizationCfg,
    ShowroomRandomizationCfg,
    TargetPoseRandomizationCfg,
    WallAppearanceRandomizationCfg,
)
from .spec import TASK_000000_SPEC


# 기준 동작을 먼저 재현할 수 있도록 아무 축도 흔들지 않는 설정을 유지한다.
TASK000000_RECORD_DETERMINISTIC = ShowroomRandomizationCfg()

# 아래 범위는 작은 시작 예시다. 실제 task의 여유 공간과 성공 조건을 확인한 뒤
# 필요한 축만 남기고 범위를 조정한다.
TASK000000_RECORD_RANDOM = ShowroomRandomizationCfg(
    robot_root=RobotRootRandomizationCfg(
        enabled=True,
        depth_x_max_m=0.010,
        lateral_y_max_m=0.010,
        yaw_max_rad=math.radians(2.0),
    ),
    objects=ObjectPoseRandomizationCfg(
        enabled=True,
        object_names=(TASK_000000_SPEC.target_object,),
        x_max_m=0.005,
        y_max_m=0.005,
        yaw_max_rad=math.radians(5.0),
    ),
)


# Mimic seed는 원본 demo를 정확히 다시 읽고 구간을 나누는 용도이므로 고정한다.
TASK000000_MIMIC_SEED = ShowroomGenerationRandomizationCfg()

# 없어져도 task 의미가 바뀌지 않는 방해 물체만 넣는다. 빈 tuple이면 presence
# randomization은 자동으로 꺼진다.
TASK000000_OPTIONAL_DISTRACTORS: tuple[str, ...] = ()

# 생성 데이터용 예시다. 자세/조명/재질/카메라를 한꺼번에 크게 흔들지 말고,
# 축별 검증을 거친 범위만 남긴다. target_pose는 현재 선반 위 물체용 event이므로
# 다른 지지면을 쓰는 task라면 전용 reset event로 교체한다.
TASK000000_MIMIC_GENERATION = ShowroomGenerationRandomizationCfg(
    target_pose=TargetPoseRandomizationCfg(
        enabled=True,
        lateral_y_max_m=0.005,
        yaw_max_rad=math.radians(5.0),
    ),
    presence=PresenceRandomizationCfg(
        enabled=bool(TASK000000_OPTIONAL_DISTRACTORS),
        object_names=TASK000000_OPTIONAL_DISTRACTORS,
        disappearance_probability=0.25,
    ),
    lighting=LightingRandomizationCfg(enabled=True),
    shelf=ShelfAppearanceRandomizationCfg(enabled=True),
    wall=WallAppearanceRandomizationCfg(enabled=True),
    camera=CameraRandomizationCfg(
        enabled=True,
        camera_names=TASK_000000_SPEC.policy_cameras,
    ),
)
