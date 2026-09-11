# Coffee Transport ACT Data Pipeline

The policy dataset keeps each successful two-can trajectory as one complete
episode. Native physical episodes and exact-state visual replays have identical
state/action arrays; visual replay changes pixels only.

## Policy contract

Both tasks use the canonical 15 Hz head camera (`cam_left_head`, 672 x 376).
Head joints remain fixed and are omitted from the named state/action vectors.

- Task 000001: left arm 7, left gripper 1, right arm 7, right gripper 1,
  lift 1 (17D).
- Task 000002: left arm 7, left gripper 1, right arm 7, right gripper 1,
  body-frame `linear_x`, `linear_y`, and `angular_z` (19D).

The physical generator emits EEF commands, not deployable arm joint commands.
For source frame `t`, the ACT action is the realized joint target stored at
`obs/joint_pos_target[t + 1]`. The corresponding state and image are the
pre-step values at `t`; therefore the final source frame is intentionally
removed. Task 000002's full planar-base state is computed from the causal
pre-step root velocity and root quaternion. Its action preserves all three
generated body-frame commands at `t`, including values that happen to be zero.

## Visual augmentation

Each of 50 physical successes is replayed eight times from every recorded
pre-step articulation and rigid-object state. The following pixel-only axes are
sampled:

- dome-light brightness and color,
- solid background-wall color,
- head-camera focal scale and local roll/pitch/yaw,
- independent label/texture yaw for the right and left orange cans.

The robot and both can rigid-root poses are checked bit-for-bit before and after
appearance mutation. The staging audit also requires every visual replay's
task-specific arrays to be exactly equal to its native source arrays.

## Build one dataset

Run only after the task's 50-success physical HDF5 is complete and no other
Isaac job owns the GPU:

```bash
scripts/sim2real/imitation_learning/tasks/coffee_transport/run_policy_dataset_pipeline.sh \
  000001 datasets/<physical-000001-50>.hdf5 \
  <campaign-dir>
```

The staging boundary verifies the reviewed task-specific generation metadata
as well as the exact runtime-rate provenance. In particular, the superseded
Task 000001 v12 data cannot enter this pipeline; Task 000001 requires a
physically audited and promoted endpoint-lift v13 dataset.
Task 000002 writes new staging, dataset, and model artifacts with a
`full_base` suffix so a legacy lateral-only artifact can never be reused by
path.

The runner locates Cyclo Lab from its own path and expects the S2R repository
next to it. Set `CYCLO_LAB_ROOT`, `CYCLO_S2R_ROOT`, or the container-name
variables at the top of the script when a checkout uses another layout.

The runner exports 50 native episodes, renders 400 visual episodes, audits the
visual-only invariant, merges 450 staging episodes, writes LeRobot v3.0, audits
all vectors/videos/metadata, and creates non-destructive symlinks below the
campaign directory. Visual rendering is resumable at five-episode boundaries.

## Train ACT

The canonical recipes in the S2R repository are:

- `training/task_000001/act.yaml`
- `training/task_000002/act.yaml`

After the corresponding LeRobot dataset exists:

```bash
cd <cyclo-intelligence-s2r>
python3 tools/training/act_recipe.py validate training/task_000001/act.yaml
python3 tools/training/act_recipe.py run training/task_000001/act.yaml
```

The campaign wrapper also validates, plans, trains, checks that the deployment contract exists, and links the completed policy into the experiment:

```bash
scripts/sim2real/imitation_learning/tasks/coffee_transport/run_act_training.sh \
  000001 <campaign-dir>
```

Each recipe uses head-only ResNet-18 ACT, a 30-frame chunk/action horizon,
50,000 steps, batch size 32, mean/std state and action normalization, and
normalized Gaussian state noise (`0.01`) only on arm/gripper state. Lift and
planar-base state are not noised.
