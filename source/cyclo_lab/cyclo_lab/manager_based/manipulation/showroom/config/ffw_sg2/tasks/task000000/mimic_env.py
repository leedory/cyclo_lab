"""Task000000용 Mimic 좌표계 adapter 예시."""

from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
import torch

from cyclo_lab.manager_based.manipulation.common import FFWSG2MimicEnv


class Task000000MimicEnv(FFWSG2MimicEnv):
    """scene 객체 자세를 로봇 root 기준으로 Mimic에 전달한다."""

    def get_object_poses(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        # EEF pose와 객체 pose는 반드시 같은 기준 좌표계를 사용해야 한다.
        if env_ids is None:
            env_ids = slice(None)

        scene_state = self.scene.get_state(is_relative=True)
        robot_root = scene_state["articulation"]["robot"]["root_pose"]
        root_pos = robot_root[env_ids, :3]
        root_quat = robot_root[env_ids, 3:7]

        object_poses = {}
        for name, state in scene_state["rigid_object"].items():
            position, quaternion = PoseUtils.subtract_frame_transforms(
                root_pos,
                root_quat,
                state["root_pose"][env_ids, :3],
                state["root_pose"][env_ids, 3:7],
            )
            object_poses[name] = PoseUtils.make_pose(
                position, PoseUtils.matrix_from_quat(quaternion)
            )
        return object_poses
