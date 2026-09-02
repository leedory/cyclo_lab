# task000000 - 새 쇼룸 task 템플릿

작성 및 확인일: 2026-09-01

이 폴더는 실행용 task가 아니라 복사용 뼈대다. 실수로 Gym 환경에 노출되지
않도록 상위 `ffw_sg2/__init__.py`에도 등록되어 있지 않다.

이 문서는 현재 task API와 Task000458/Task000525 구조를 기준으로 작성했다.
공통 환경, randomization, Mimic 인터페이스가 바뀌면 이 템플릿도 함께 고칠 수
있다. 새 task를 시작할 때는 최근 task와 이 템플릿의 차이를 한 번 확인한다.

## 만드는 순서

1. 이 폴더를 `task_000526`처럼 복사한다. 실제 Python package 이름에는
   `task_` 뒤에 여섯 자리 번호를 붙인다. 예를 들어 task 525는 `task_000525`다.
2. 복사본 안의 `000000`, `Task000000`, `TASK_000000`, `TASK000000`을 새 번호에
   맞게 모두 바꾼다.
3. `spec.py`에서 대상 물체, 작업 지시문, 사용할 팔을 정한다. `target_object`는
   설명용 이름이 아니라 scene에 등록된 정확한 객체 이름이어야 한다.
4. `profiles.py`에서 deterministic 설정을 먼저 확인한 뒤 필요한 randomization만
   켠다.
5. task 전용 물체, observation, action, 성공 판정이 필요하면 `env_cfg.py`와 별도
   helper 파일에 추가한다. 공통 동작은 가능한 한 `../common.py`를 그대로 쓴다.
6. Mimic을 사용한다면 `mimic_cfg.py`의 subtask 신호와 성공 판정을 실제 task
   기준으로 구현한다. 객체 좌표계 처리는 `mimic_env.py` 예시를 따른다.
7. 마지막으로 `ffw_sg2/__init__.py`에 사용할 Gym ID를 등록한다.

## 파일 역할

| 파일 | 수정할 내용 |
| --- | --- |
| `spec.py` | task 번호, 대상, 지시문, 주 사용 팔 |
| `profiles.py` | 기록 및 생성 시 허용할 randomization 범위 |
| `env_cfg.py` | 기본 scene과 task 전용 구성 |
| `mimic_cfg.py` | subtask 경계, 생성 횟수, 보간 및 noise |
| `mimic_env.py` | Mimic에서 사용할 EEF/객체 좌표계 보정 |

등록 형태는 아래와 같다. 번호와 import 경로를 새 task에 맞게 바꿔 넣는다.

```python
gym.register(
    id="Cyclo-Real-Showroom-Task000000-FFW-SG2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tasks.task_000000.env_cfg:Task000000EnvCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Cyclo-Real-Showroom-Task000000-Random-FFW-SG2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tasks.task_000000.env_cfg:Task000000RandomEnvCfg"
        ),
    },
    disable_env_checker=True,
)
```

처음에는 deterministic 환경을 한 번 reset해 대상 물체와 카메라가 보이는지
확인한다. 그 다음 random 환경을 여러 번 reset해 물체가 지지면 밖으로 나가거나
로봇과 겹치지 않는지 확인한다.

## Mimic 사용 시 주의

`mimic_cfg.py`의 `<replace_with_..._signal>`은 실제 observation term 이름으로
바꿔야 한다. 각 신호는 한 프레임의 우연한 접촉이 아니라 grasp, 이동, release
같은 단계가 확실히 끝났음을 나타내야 한다. 최종 성공 판정도 별도로 연결한다.

Seed와 Generate 환경은 모두 `Task000000MimicEnv`를 entry point로 사용하고,
각각 `Task000000MimicSeedEnvCfg`, `Task000000MimicGenerateEnvCfg`를 연결한다.
더 복잡한 예시는 `task_000458`의 `mimic_cfg.py`와 `mimic_env.py`를 참고한다.
