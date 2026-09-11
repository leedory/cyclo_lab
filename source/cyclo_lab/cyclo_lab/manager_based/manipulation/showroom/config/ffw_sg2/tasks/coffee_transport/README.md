# Coffee transport tasks

> **Task-specific production status:** Task 000002 keeps the approved v12
> commanded-target arm-hold contract. Task 000001 v12 is superseded because it
> recorded intermediate absolute lift targets. Its endpoint-only v13 code is
> intentionally marked `production_approved=False` until a new physical probe
> is generated and reviewed.

Both profiles hold each arm's existing seven-joint commanded
target during the initial lift/base normalization. It intentionally preserves
the gravity/drive equilibrium offset between command and actual joint position;
capturing the actual position as the command caused the v11b arm transient. The
hold is off by default and is always released by `finally` before the first
demonstrated pickup. Each episode records commanded targets as
`normalization_external_joint_hold_{left,right}_target_rad_json`, hold-start
actual positions as
`normalization_external_joint_hold_{left,right}_start_actual_rad_json`, and the
held step count as `normalization_external_joint_hold_steps`. The existing
first-ten-frame arm gate remains `0.010 rad`. `--experimental-realized-initial-pose`
is retained only as a deprecated no-op so the recorded probe commands remain
reproducible.

Temporary Task 000001 and 000002 share one object layout and episode-local HOME
contract. The numeric IDs can be renamed later without flattening the package.

## Ownership

| File | Owns |
| --- | --- |
| `coffee_transport_common.py` | measured geometry, assets, final success |
| `environment.py` | shared scene and reset-event wiring |
| `randomization_cfg.py` | shared typed physical and visual randomization axes and bounds |
| `reset_events.py` | can visual yaw, pose, episode HOME, lift, and base-route reset sampling |
| `generation_contract.py` | can order, arm assignment, subtask phases, retarget frames |
| `quality_contract.py` | generation body-grasp metrics and runtime placement gates |
| `seed_automation.py` | online G-key inverse replay and fail-closed state machine |
| `task_000001/spec.py` | vertical task identity and instruction |
| `task_000001/profiles.py` | vertical seed/record/physical/visual profiles |
| `task_000001/env_cfg.py` | vertical-only lift and termination wiring |
| `task_000002/spec.py` | horizontal task identity and instruction |
| `task_000002/profiles.py` | horizontal seed/record/physical/visual profiles |
| `task_000002/env_cfg.py` | mobile-only base and termination wiring |

Future runtime boundary metrics should go in a shared `generation_terms.py` only
when both tasks use the exact same measurement. Dataset campaign counts are not
runtime configuration and belong in experiment manifests, not env classes.

## Profiles

- `*_DETERMINISTIC` / `*_SEED`: fixed can centers, authored HOME, normalized actuator.
- `*_RECORD_RANDOMIZED`: can squares, HOME X/Y/yaw, and the task-specific initial axis.
- `*_PHYSICAL_TRAJECTORY_GENERATION`: collision-relevant randomization only.
- `*_VISUAL_REPLAY_AUGMENTATION`: lighting, wall, policy cameras, and orange-can label yaw.

Task000002 samples HOME first and caches it. Its initial base pose is then sampled as
`HOME + (0, route_offset_y, 0)`, so the normalize phase always returns to that same
episode's valid `+-2 cm`, `+-1 degree` HOME rather than to a second random pose.

## Fixed behavior

- Can order: right, left.
- Active arm: right for the right can, left for the left can.
- Head: default and held.
- Task 000001: lift starts in [-0.28, 0] m, normalizes to 0, transports by lift.
- Task 000002: start is cached HOME plus world +Y offset, normalizes to that same HOME,
  and transports by closed-loop world-pose base control.

## Production trajectory generation

The standard launcher is production-enabled for both tasks:

```bash
./scripts/sim2real/imitation_learning/run_coffee_transport_trajectory_generation.sh \
  000001 --input-file <seed.hdf5> --output-file <physical50.hdf5> \
  --num-successes 50 --max-attempts 400 --seed <campaign-seed> --headless
```

Use `000002` with its own seed/output/campaign seed for the mobile-base task. Every
dataset and episode records the input seed HDF5 SHA-256, production promotion
ID, both physical-probe seeds and hashes, and the unchanged cylinder collision
profile/hash. Existing output paths are refused rather than overwritten.

## Inspecting policy state and action labels

`view_lerobot_episode.py` renders one LeRobot v3 episode with a named table of
the measured `observation.state`, supervised `action`, and their difference.
It also writes a frame-exact CSV and a JSON summary beside the MP4. For example:

```bash
python scripts/sim2real/imitation_learning/tasks/coffee_transport/view_lerobot_episode.py \
  --dataset /workspace/dataset/robotis/task_000001_coffee_transport_headcam_v30 \
  --episode 2 \
  --signal lift_joint \
  --output /tmp/task000001_episode2_lift.mp4
```

Omit `--signal` to display every named policy dimension. Use `--camera` for a
specific video feature and `--start-frame`/`--end-frame` for a smaller window.
`--signals-dataset` may supply the state/action rows from another same-length
dataset while retaining the first dataset's video.

## Nominal seed acquisition

Run one task at a time from the repository root:

```bash
./scripts/sim2real/imitation_learning/record_coffee_transport_seed.sh 000001
./scripts/sim2real/imitation_learning/record_coffee_transport_seed.sh 000002
```

Task 000001 writes
`datasets/task_000001_orange_transport_two_can_v4_seed.hdf5`; Task 000002 keeps
`datasets/task_000002_orange_transport_two_can_v3_seed.hdf5`. The prior seed
files are never appended to.

The seed launcher intentionally rejects `Random` environments. Isaac Sim itself runs
headless, while the separate operator-camera dashboard remains visible. Before `B`,
and again after either `R` or a completed `N`, the recorder commands the exact
configured 19D HOME pose (including lift 0) and zero base velocity.

Press `B` once; both A3 arms are forced to IDLE and the robot keeps exact HOME.
Press the right-arm tact once; control is enabled immediately, so teleoperate only approach, body grasp, and extraction
for the right can. Press `G`; the recorder locks manual inputs, blends both arms to the fixed
episode HOME while preserving the active grasp and head, waits for 10 stable frames,
moves the task axis, replays pickup backward, releases, and returns the axis to source.
Task000001 records the measured lift position as state while commanding the
absolute destination endpoint from the first motion frame: `-0.28` m downward
and `0.0` m upward. It does not record intermediate lift action targets.
Task000002 remains closed-loop body-velocity control. After release,
the inverse approach must finish at exact destination-side HOME and settle there before
the axis returns. At source the robot holds HOME until a fresh left-arm tact immediately enables the left pickup. Press `N` only
after the second automatic return reports completion. `N` rechecks every can; `R`
discards the entire episode and restores HOME.

G does not use the body-carry quality measurement as an acceptance gate; it only
requires a recorded active-gripper close transition. At G, the active A3 arm is explicitly returned to IDLE; the last commanded closed
gripper value, rather than lagging measured state, is retained. The automatic HOME
state uses a fixed target and excludes both grippers from its measured settle check.
Placement is observed for at most one second and can finish early after eight stable
frames inside the translated named square, support-height error at most 0.01 m, tilt
at most 10 degrees, settled velocity, and EEF clearance of 0.10 m. A failed interim
check does not strand the lift or base: the axis returns to source and `N` performs the
strict final two-can check.
