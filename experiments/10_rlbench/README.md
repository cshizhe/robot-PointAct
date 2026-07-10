# RLBench Benchmark

This directory contains the installation, data preparation, training, and evaluation instructions for the RLBench benchmark.


## Install RLBench

```bash
conda create -n rlbench python==3.10
```

Follow [instructions](https://github.com/vlc-robot/robot-3dlotus/blob/main/INSTALL.md) to install Pyrep and RLBench.

## RLBench Data Generation

The processed RLBench-10Task training data used in our [PointACT](https://arxiv.org/abs/2605.21414) paper can be downloaded directly:
```bash
hf download --repo-type dataset cshizhe/RLBench-10Task \
    --local-dir robot_data/rlbench/lerobot_point_lmdb
```

To generate your own RLBench data, follow the steps below.

1. Follow the [robot-3dlotus data generation pipeline](https://github.com/vlc-robot/robot-3dlotus/blob/main/DATAGEN.md) to generate microsteps and RGB-D keysteps.

2. Convert the keystep data to LeRobot v2.1 format:
```bash
conda activate pointact

export SVT_LOG=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=TRUE
export HDF5_USE_FILE_LOCKING=FALSE

cd data_prep/rlbench_to_lerobot

keystep_dir=$SCRATCH/datasets/robot_data/rlbench/train_dataset/keysteps_bbox
microstep_dir=$SCRATCH/datasets/robot_data/rlbench/train_dataset/microsteps
task_instruction_file=$HOME/codes/robot-3dlotus/assets/taskvars_instructions_new.json
output_dir=$SCRATCH/datasets/robot_data/rlbench/lerbot_point_lmdb
repo_id=push_button+0

python convert_keystep_to_lerobot_v21.py \
    --keystep_dir ${keystep_dir} \
    --task_instruction_file ${task_instruction_file} \
    --taskvars push_button+0 \
    --keep_stop_action \
    --export_point_cloud --microstep_dir ${microstep_dir} \
    --output_dir ${output_dir} --repo_id ${repo_id}
```

3. Generate state/action statistics:

```bash
python data_prep/prepare_robot_state_action_stats.py \
    --dataset_dirs robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps \
    --output_file robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps/robot_state_action_stats/rot6d_points_frontview.json \
    --point_cloud_dir points_frontview \
    --state_xyz_slice 0 3 \
    --action_xyz_slice 0 3 \
    --state_rotation_slice 3 7 \
    --action_rotation_slice 3 7 \
    --rotation_type quat \
    --target_rotation_type rot6d

python data_prep/prepare_robot_state_action_stats.py \
    --dataset_dirs robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps \
    --output_file robot_data/rlbench/lerobot_point_lmdb/hybridvla_10tasks_train_keysteps/robot_state_action_stats/rot6d.json \
    --state_rotation_slice 3 7 \
    --action_rotation_slice 3 7 \
    --rotation_type quat \
    --target_rotation_type rot6d
```

## Training

We support EO1, EO1-Point, QwenGR00T, QwenGR00T-Point, Pi0, and PointAct.
For PointAct, you can switch between classification and regression action heads. You can also remove images from the VLM by setting `video_key_ids_for_vlm: []` in the data configuration file. In RLBench, the 3D point cloud alone is often sufficient for most tasks, so removing images can substantially speed up training, roughly 13 hours on 1 H100 GPU, while keeping comparable performance.

Before launching a run, update the relevant configuration files in `experiments/10_rlbench/data_configs`.
SLURM launch examples are available in `job_scripts`.

We train all models with an effective batch size of 128 using 1 or 2 H100 GPUs.

```bash
# EO1
bash experiments/10_rlbench/train_eo1.sh
# EO1-Point
bash experiments/10_rlbench/train_eo1_point.sh
# QwenGR00T
bash experiments/10_rlbench/train_vla2.sh
# QwenGR00T-Point
bash experiments/10_rlbench/train_vla2.sh
# Pi0
bash experiments/10_rlbench/train_pi0.sh
# PointACT
bash experiments/10_rlbench/train_pointact_clf_concerto.sh
bash experiments/10_rlbench/train_pointact_clf_utonia.sh
```

## Inference

We use client-server evaluation so the policy can run in the `pointact` environment while the RLBench simulator runs in its own environment.

```bash
bash experiments/10_rlbench/eval_hybridvla_10tasks.sh <YOUR-EXPR-DIR> <CKPT-STEP> <SEED> <PRED-ROTATION-TYPE> "--args.num_episodes 100"
```

