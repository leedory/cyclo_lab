# Task000525 16-env visual augmentation demo

이 데모는 `task_000525_all_subtasks_original_plus_visual_aug_450_v30`을 만들 때
사용한 source HDF5 exact-state replay와
`TASK000525_VISUAL_REPLAY_AUGMENTATION` profile 중 third-person 화면에서 보이는
lighting, wall, coffee-label yaw 축을 재사용한다.

- 16개 showroom environment를 8 m 간격의 4 x 4 overview로 표시한다.
- 모든 env의 `LeftWall_01`, `BackWall_01`, `WallBackground`는 숨긴다.
  기존 `LeftWall`, `BackWall`은 그대로 두고 wall visual augmentation도 유지한다.
- 각 env는 서로 다른 성공 trajectory를 frame 0부터 all-subtasks 끝까지 재생한다.
- 한 env가 자기 episode를 끝내면 다른 env를 기다리지 않고 새 full episode로
  돌아가며 local light, wall, coffee-label yaw를 새로 sampling한다.
- 데모에서는 정책 카메라 3개를 생성하지 않고 third-person viewport만 렌더한다.
- DomeLight는 USD stage 전체에서 공유되므로 개별 env reset 때는 유지된다. 16개
  env가 같은 frame에 모두 reset될 때만 global dome도 다시 sampling한다.
- 데이터나 영상을 저장하지 않고 창을 닫거나 `Ctrl-C`를 누를 때까지 반복한다.

## 실행

host에서 X11을 포함해 container를 시작하고 들어간다.

```bash
cd /home/robotis-ai/cyclo_lab
./docker/container.sh start
./docker/container.sh enter
```

container 안에서 1배속 또는 5배속 launcher를 실행한다.

```bash
cd /workspace/cyclo_lab

# 1배속: recorded 15 FPS timeline을 wall-clock 1배속으로 재생
./scripts/sim2real/imitation_learning/tasks/Task000525_demo/run_task000525_demo.sh

# 5배속: recorded 15 FPS timeline을 wall-clock 5배속(75 source FPS)으로 재생
./scripts/sim2real/imitation_learning/tasks/Task000525_demo/run_task000525_demo_5x.sh
```

두 launcher 모두 기본 16 env이며 각 env는 독립적인 episode cursor와 loop count를
갖는다. 렌더가 목표 FPS보다 느려도 source cursor가 wall-clock을 따라가므로 5배속이
유지된다. 새 episode는 항상 frame 0부터 시작한다.

자주 쓰는 옵션:

```bash
# 각 env가 source episode 번호를 순서대로 선택
./scripts/sim2real/imitation_learning/tasks/Task000525_demo/run_task000525_demo.sh \
  --sequential

# 원하는 정수 배속
./scripts/sim2real/imitation_learning/tasks/Task000525_demo/run_task000525_demo.sh \
  --speed 3

# rendering 가능한 최대 속도
./scripts/sim2real/imitation_learning/tasks/Task000525_demo/run_task000525_demo_5x.sh \
  --no-throttle

# 오른쪽 close-up에 표시할 env 변경
./scripts/sim2real/imitation_learning/tasks/Task000525_demo/run_task000525_demo_5x.sh \
  --zoom-envs 1 6 11

# close-up panel 없이 main overview만 표시
./scripts/sim2real/imitation_learning/tasks/Task000525_demo/run_task000525_demo_5x.sh \
  --no-zoom-panel
```

기본 source는 provenance에 기록된 성공 trajectory 파일이다.

```text
datasets/task_000525_trajectory_ccw_rootstable_success_50_v2.hdf5
```

다른 위치에서는 `--input-file /absolute/path/to/source.hdf5`로 지정한다.
두 GUI launcher 모두 의도적으로 `--headless`를 전달하지 않는다.
