"""Randomization profiles owned by showroom Task000525.

Coffee position sampling is wired directly by Task000525RandomEnvCfg because
each object has a different reviewed center rectangle.  The shared showroom
profiles still own the task-independent visual axes.
"""

from ...randomization.cfg import (
    CameraRandomizationCfg,
    LightingRandomizationCfg,
    ShowroomGenerationRandomizationCfg,
    ShowroomRandomizationCfg,
    WallAppearanceRandomizationCfg,
)
from .spec import TASK_000525_SPEC


TASK000525_RECORD_DETERMINISTIC = ShowroomRandomizationCfg()

# The B-region position event is added by Task000525RandomEnvCfg after the
# common profile builder has installed its deterministic reset terms.
TASK000525_RECORD_RANDOM = ShowroomRandomizationCfg()

TASK000525_MIMIC_SEED = ShowroomGenerationRandomizationCfg()

# TODO(task000525): add appearance-to-region shuffling and active-arm dispatch
# only after their episode metadata and Mimic contracts are designed.
TASK000525_MIMIC_GENERATION = ShowroomGenerationRandomizationCfg(
    lighting=LightingRandomizationCfg(enabled=True),
    wall=WallAppearanceRandomizationCfg(enabled=True),
    camera=CameraRandomizationCfg(
        enabled=True,
        camera_names=TASK_000525_SPEC.policy_cameras,
    ),
)

# Strict visual replay: no robot, coffee, furniture, or collision changes.
TASK000525_AUGMENT_VISUAL = ShowroomGenerationRandomizationCfg(
    lighting=LightingRandomizationCfg(enabled=True),
    wall=WallAppearanceRandomizationCfg(enabled=True),
    camera=CameraRandomizationCfg(
        enabled=True,
        camera_names=TASK_000525_SPEC.policy_cameras,
    ),
)
