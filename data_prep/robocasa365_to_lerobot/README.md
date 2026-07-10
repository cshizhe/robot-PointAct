# RoboCasa365 to LeRobot Point Dataset

This converter uses action rollout replay for RoboCasa365 PandaOmron demos and writes a LeRobot dataset plus a fused 3-view point-cloud LMDB.

Note: `robocasa365` simulator cannot be properly ran in H100 GPU.

Run it in two stages:

- `robocasa365`: read the source parquet actions, replay them in simulation, and cache replay outputs.
- `pointact`: write the cached replay into LeRobot format.

## 1. Replay Actions in Simulation

Run in the `robocasa365` environment, where RoboCasa / robosuite / MuJoCo work.

```bash
conda activate robocasa365
export PYTHONPATH=/home/shichen/codes/GroundedVLA/PointAct:$PYTHONPATH

cd /home/shichen/codes/GroundedVLA/PointAct/pointact/robot_envs/robocasa365_utils/robocasa365_to_lerobot

taskname=OpenDrawer
input_dir=/scratch/shichen/datasets/robot_data/robocasa365/v1.0/target/atomic/$taskname/20250816

python replay.py --input-dir $input_dir/lerobot --cache-dir $input_dir/replay_cache --resume
```

Add `--episodes 0` for a small debug run.
Add `--resume` to skip already-completed episode caches and continue the rest.

This writes per-episode replay caches under `replay_cache/episodes`. Each episode
has one metadata/image/action `.npz` and one fused point-cloud `_points.npy`.

## 2. Write LeRobot Dataset

Run in the `pointact` environment.

```bash
conda activate pointact

export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

python convert.py \
  --cache-dir $input_dir/replay_cache \
  --output-dir /scratch/shichen/datasets/robot_data/robocasa365/lerobot_point_lmdb/${taskname} \
  --repo-id ${taskname}
```

`convert.py` writes one LMDB key per timestep and commits point-cloud writes every
50 frames by default. Use `--lmdb-commit-every N` to tune this, or
`--lmdb-commit-every 0` for one transaction per episode. It also enables LeRobot
async image writing with `--image-writer-threads 12` by default.

The final dataset is written to:

```text
lerobot_point_lmdb/OpenDrawer/
  data/
  videos/
  meta/
  points_3views/
  cache_meta.json
  replay_summary.jsonl
```

## Conventions

- Source action order: `base_motion, control_mode, eef_position, eef_rotation_axisangle, gripper_close`.
- Simulator action order: `eef_position, eef_rotation_quat_xyzw, gripper_close, base_motion, control_mode`.
- Saved LeRobot action order: `eef_position, eef_rotation_quat_xyzw, gripper_close, base_motion, control_mode`.
- Saved image keys are `observation.images.left_image`, `observation.images.right_image`,
  and `observation.images.wrist_image`, matching `RoboCasa365Env.get_observation`.
- Task text is stored through the LeRobot episode task field; no extra
  `annotation.human.*` features are written.
- Saved `observation.state` follows `RoboCasa365Env.get_observation`:
  `eef_position_relative, eef_rotation_relative_quat, gripper_qpos, base_position, base_rotation_quat`.
- When replay reaches `done`, the cache appends the terminal observation with a no-op
  action whose quaternion is xyzw identity `[0, 0, 0, 1]`, keeping image, point cloud,
  action, reward, done, and state lengths identical.
- Replay point clouds are fused from left, right, and wrist cameras, transformed to
  robot base frame, cropped to the workspace, voxelized, and saved per episode as
  one `(sum_N, 6)` float32 `xyzrgb` `.npy`, with frame offsets in the `.npz`.
- `convert.py` writes those replay point clouds into the final `points_3views/` LMDB.


## Troubleshotting

Some robocasa tasks contain transparent objects, such as the bottle in `PickPlaceCounterToCabinet`. The depth is not accurate.
