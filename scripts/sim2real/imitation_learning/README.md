# Imitation-learning tools

Keep reusable recording and format-conversion code in the shared directories.
Task-specific experiments belong under `tasks/task_<id>/`; they should not add
task IDs or scene assumptions to the generic converters.

## Replay augmentation pipeline

Replay staging is an intermediate, reviewable dataset made of one manifest,
per-episode policy arrays, and already encoded camera videos. It separates the
Isaac-dependent render step from the final dataset writer:

1. `data_converter/isaac_hdf5_to_replay_staging.py` exports the original
   recorded observations without rerunning physics.
2. `tasks/task_000458/replay_with_randomization.py` reruns Task000458 actions
   in Isaac and writes randomized observations using the same staging schema.
3. Review the manifests and videos. Task000458 keeps its contract-aware HDF5
   viewer and three-way replay comparison in `tasks/task_000458/`.
4. `data_converter/merge_replay_staging.py` combines accepted staging
   directories by hard-linking their arrays and videos when possible. It exists
   so accepted original and augmented episodes can be assembled without
   rerunning Isaac or re-encoding videos.
5. `data_converter/replay_staging_to_lerobot_v30.py` validates the selected
   staging set and invokes the Cyclo Intelligence LeRobot v3 writer.

Every intermediate and final output remains marked `training_ready=false`
until a reviewer explicitly accepts it. The staging scripts refuse to overwrite
an existing output directory.
