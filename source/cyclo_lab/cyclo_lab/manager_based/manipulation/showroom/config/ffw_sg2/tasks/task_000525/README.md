# Task000525

Task000525 moves the green coffee can from kolbjorn_cabinet_02 to the fixed
ivory mat on central_dining_set with the SG2 right arm and mobile base.

## Runtime contract

- robot asset: FFW_SG2_softgripper.usd;
- target: coffee_can_green;
- can order at reset: black, brown, green, orange from low to high world Y;
- approved can-center layout: candidate B;
- action width: joint19 plus body-frame [vx, vy, wz] = 22;
- control/output/camera rate: 15 Hz;
- physics rate: 30 Hz, giving two physics substeps per output action;
- active manipulation arm: right.

There is no manipulation source-frame repeat option. One source frame advances
per 15 Hz environment step. Posture-biased IK is still recomputed at both
30 Hz physics substeps. If the repeat-free campaign is materially worse than
the 50/100 repeat-3 baseline, the preferred fallback is physics 45 Hz with
control/output/camera still at 15 Hz (decimation 3), not duplicated 30/45 Hz
dataset rows.

The 2026-09-01 fixed 10-attempt repeat-free check passed 6/10 runs. All four
failures lost the can before the carry checkpoint; none was a planner or
release failure. Every trajectory, action, camera, and fixed-size recorded path
length agreed, the measured output rate was 15 Hz, and the manipulation cursor
never held a source frame. This is not worse than the 50/100 repeat-3 baseline,
so the 45 Hz physics fallback is not enabled.
Decision rationale: ../agents/docs/2026-09-01_11-28-31_task000525_mimicgen_action_staircase_and_smoothing_options.md
and its KR counterpart.

## Authoritative files

| Question | Authoritative file |
| --- | --- |
| identity, target, cameras, rates | spec.py |
| B-region geometry and selected layout | layout.py |
| complete randomization profiles | profiles.py |
| reset wiring and base environment | env_cfg.py |
| runtime EEF frame policy | generation_contract.py |
| generation environment and success metrics | locomanipulation_sdg_env.py |
| generation CLI/state machine/planner | scripts/.../task_000525/generate_trajectories.py |
| seed-recording route controller | online_dijkstra.py |
| visual-only can-label yaw | appearance_events.py |
| mat pose, collider, and fixed joint | destination_mat.py |
| Task525-only arm hold tuning | robot_stability.py |

Task525 uses generation_contract.py and a local locomanipulation trajectory
generator directly; it does not expose an Isaac Lab Mimic compatibility shim.

## Randomization profiles

All Task525 axes are selected in profiles.py. Environment classes do not carry
separate coffee-position or coffee-yaw booleans.

| Profile | Physical axes | Appearance axes |
| --- | --- | --- |
| TASK000525_DETERMINISTIC | none | none |
| TASK000525_RECORD_RANDOMIZED | four can centers in B regions | can visual yaw |
| TASK000525_PHYSICAL_TRAJECTORY_GENERATION | root X/Y +/-30 mm, yaw +/-2.5 degrees, can B regions | none |
| TASK000525_VISUAL_REPLAY_AUGMENTATION | none | lighting, wall, cameras, can visual yaw |

The canonical Gym ID is
Cyclo-Real-Showroom-Task000525-Trajectory-Generation-FFW-SG2-v0.

## Geometry

The cabinet support is approximately 300.0 mm deep and 791.4 mm wide. The B
rectangles constrain can centers:

- low/high Y outer margins: 120/120 mm;
- adjacent center-rectangle gap: 100 mm;
- back/front X margins: 120/30 mm;
- each rectangle: approximately 150.0 x 62.8 mm;
- minimum adjacent can-surface gap: 33 mm;
- lateral can-body-to-edge clearance: 86.5 mm;
- upright can origin Z: 1.350128932 m.

The 30 mm front center margin permits a maximum 3.5 mm footprint overhang. This
is intentional in the approved layout.

The rigid 250 x 210 x 2 mm ivory mat is at world center
(-0.8607353544, -0.6178486049, 0.7591681609), WXYZ
(0.7071067812, 0, 0, 0.7071067812). A prestartup fixed joint attaches it to the
dining-table rigid body.

## EEF reference frames

These describe the implemented generator, not future Mimic drafts.

| Phase | Implemented EEF reference |
| --- | --- |
| grasp | fixed source and randomized target-can frames captured after reset |
| cabinet clear / carry-home | blend from fixed can frame to current robot root |
| navigation and approach | current robot root |
| place, release, empty return | recorded source base retargeted to current robot root |

The place/release sequence is intentionally unchanged in this refactor.
infer_dropoff_replay_step(), pre-place lift descent, handle_drop_off_state(),
and the final release gate retain their existing behavior.

## Route-planning timing

Recording and trajectory generation both:

1. plan at activation/reset as a preflight validity check;
2. after carry-home settles, plan again from the measured root immediately
   before base motion.

Generated metadata includes both planning times, waypoint counts, and path
lengths. Seed recording still performs bounded off-path replanning.

## Success metrics

The runtime source of truth is evaluate_task525_carry_checkpoint() and
evaluate_task525_final_checkpoint() in locomanipulation_sdg_env.py.

Carry checkpoint:

- right TCP-to-can distance <= 0.080 m;
- can displacement from randomized initial pose >= 0.200 m;
- non-gripper joint19 maximum carry/home error <= 0.150 rad or m.

Final checkpoint:

- full can footprint remains in the mat with 5 mm extra edge margin;
- mat-local can-origin Z is in [0.020, 0.090] m;
- can linear speed <= 0.030 m/s;
- can angular speed <= 0.250 rad/s;
- right TCP-to-can distance >= 0.100 m.

HDF5 episode attributes store success, failure_reason, and every numeric metric
with a quality_ prefix, including planner provenance.

## Coffee-can texture rotation

Visual augmentation authors only
/Visual/SharedMesh/xformOp:rotateZ. It never writes rigid pose, velocity, or
collision data. Exact sampled radians and degrees for all four cans are stored
in each replay manifest. Replay also requires robot and four-can rigid roots to
remain bit-identical immediately across profile application.

Existing runtime verification found bit-identical roots, unchanged three
collision prims per can, and non-accumulating visual rotation. Raw UV
translation is not used because the current atlas also contains top/bottom
islands.

A 2026-09-01 one-episode visual-replay smoke test produced 209 frames at 15 Hz,
recorded independent radian/degree yaw samples for all four cans, and reported
exactly zero protected-root error for the robot and every can.

## Robot stability

Task525 enables PhysX stabilization and scales only arm stiffness by 2 and
damping by sqrt(2). Effort limits, lift, base, Task458, and shared SG2 dynamics
are unchanged. Measured maximum static arm settle improved from 0.01839 rad
(1.05 degrees) to 0.01057 rad (0.61 degrees).

Evidence:
../agents/docs/2026-08-28_14-46-33_task000525_runtime_and_teleop.md and its KR
counterpart. Unit coverage is test_task_000525_robot_stability.py.

## Contact-last solver order

`solve_articulation_contact_last` deliberately remains disabled. A current-code
30 Hz physics / 15 Hz control A/B changed only this boolean; contact-last lost
the can in all three fresh-process reproductions, while the baseline retained
it. In the latest run can/hand translation RMS increased from 0.285 to 6.469 mm
and the can reached the floor at source step 120. Reconsider it only after the
paired acceptance gates in the report pass.

Evidence:
../agents/experiments/2026-09-01_13-52-02_task000525_contact_last_current_audit/README.md

## Commands

Record a continuous seed:

~~~bash
./scripts/sim2real/imitation_learning/record_task000525_mobile_demo.sh \
  --dataset_file /workspace/cyclo_lab/datasets/task_000525_mobile_seed.hdf5 \
  --num_demos 1 --camera_view operator --render_episode_cameras
~~~

Generate repeat-free 15 Hz trajectories:

~~~bash
./scripts/sim2real/imitation_learning/run_task000525_trajectory_generation.sh \
  --dataset /workspace/cyclo_lab/datasets/task_000525_mobile_seed_v2.hdf5 \
  --output_file /workspace/cyclo_lab/datasets/task_000525_trajectory_15hz.hdf5 \
  --num_runs 3 --max_attempts 10 --device cpu --headless --enable_cameras
~~~

## Deliberately deferred

- appearance-to-region permutation and target-side-dependent arm selection;
- carrying-object swept-volume validation for the complete route;
- watertight/component-aware soft-finger collision proxies.
