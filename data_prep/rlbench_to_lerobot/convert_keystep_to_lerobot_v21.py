import argparse
import inspect
import json
import os
import pickle as pkl
import shutil

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from pointact.utils.depth import get_real_metric_depth, image_to_float_array
from pointact.utils.point_cloud import get_voxelized_point_cloud_from_rgb_depth
from rlbench_utils.camera_depth_utils import extract_camera_info
from rlbench_utils.config import RLBENCH_FEATURES

msgpack_numpy.patch()


DEPTH_RGB_SCALE_FACTOR = 2**24 - 1

CAMERA_NAMES = ("left_shoulder", "right_shoulder", "wrist", "front")

POINT_WORKSPACE = {
    "X_BBOX": [-0.5, 1.5],  # 0 is the robot base
    "Y_BBOX": [-1, 1],  # 0 is the robot base
    "Z_BBOX": [0.7505, 2],  # 0 is the floor, 0.75 table height
}

POINT_CLOUD_OUTPUTS = {
    "observation.points.frontview": {
        "dirname": "points_frontview",
        "camera_names": ("front",),
        "label": "front view",
    },
    "observation.points.4views": {
        "dirname": "points_4views",
        "camera_names": CAMERA_NAMES,
        "label": "multi view",
    },
}


def resolve_taskvars(keystep_dir, taskvars=None, taskvar_file=None):
    if taskvar_file is not None:
        with open(taskvar_file) as f:
            return json.load(f)
    if taskvars is not None:
        return taskvars
    return [x for x in os.listdir(keystep_dir) if "+" in x]


def get_micro_taskvar_dir(microstep_dir, taskvar):
    task, variation = taskvar.split("+", 1)
    return os.path.join(microstep_dir, task, f"variation{variation}", "episodes")


def build_point_clouds(value, micro_taskvar_dir, episode_key, voxel_size):
    robot_state_file = os.path.join(micro_taskvar_dir, episode_key, "low_dim_obs.pkl")
    with open(robot_state_file, "rb") as f:
        robot_states = pkl.load(f)

    point_clouds = {field: [] for field in POINT_CLOUD_OUTPUTS}
    for t, frameid in enumerate(value["key_frameids"]):
        pose_diff = np.abs(value["action"][t][:7] - robot_states[frameid].gripper_pose)
        assert np.isclose(np.mean(pose_diff), 0, atol=1e-5), np.mean(pose_diff)

        camera_info = extract_camera_info(robot_states[frameid].misc, convert_camera=True)
        depths = {}
        intrinsics = {}
        extrinsics = {}
        for cam_name in CAMERA_NAMES:
            depth_path = os.path.join(
                micro_taskvar_dir, episode_key, f"{cam_name}_depth", f"{frameid}.png"
            )
            with Image.open(depth_path) as image:
                depth_image = np.array(image)
            depth = image_to_float_array(
                depth_image, scale_factor=DEPTH_RGB_SCALE_FACTOR
            )
            depth = get_real_metric_depth(
                depth, camera_info[cam_name]["near"], camera_info[cam_name]["far"]
            )
            depths[cam_name] = depth
            intrinsics[cam_name] = camera_info[cam_name]["intrinsics"]
            extrinsics[cam_name] = camera_info[cam_name]["extrinsics"]

        rgbs = {cam_name: value["rgb"][t, i] for i, cam_name in enumerate(CAMERA_NAMES)}
        for field, spec in POINT_CLOUD_OUTPUTS.items():
            camera_names = spec["camera_names"]
            point_clouds[field].append(
                get_voxelized_point_cloud_from_rgb_depth(
                    [rgbs[cam_name] for cam_name in camera_names],
                    [depths[cam_name] for cam_name in camera_names],
                    [intrinsics[cam_name] for cam_name in camera_names],
                    [extrinsics[cam_name] for cam_name in camera_names],
                    POINT_WORKSPACE,
                    voxel_size=voxel_size,
                )
            )

    return point_clouds


def load_keystep_episodes(
    keystep_dir,
    taskvars,
    keep_stop_action=False,
    export_point_cloud=False,
    microstep_dir=None,
    voxel_size=0.01,
):
    print(f"#taskvars {len(taskvars)}")

    for taskvar in taskvars:
        taskvar_dir = os.path.join(keystep_dir, taskvar)
        micro_taskvar_dir = None
        if export_point_cloud:
            micro_taskvar_dir = get_micro_taskvar_dir(microstep_dir, taskvar)

        with lmdb.open(taskvar_dir, readonly=True) as lmdb_env:
            with lmdb_env.begin() as txn:
                num_episodes = txn.stat()["entries"]
                # for key, value in txn.cursor(): # the order is wrong
                for episode_id in range(num_episodes):
                    episode_key = f"episode{episode_id}"
                    value = txn.get(episode_key.encode("ascii"))
                    value = msgpack.unpackb(value)
                    demo_len = len(value["key_frameids"])

                    gripper_poses = np.array(value["action"], dtype=np.float32)
                    state = gripper_poses
                    action = np.concatenate([gripper_poses[1:], gripper_poses[-1:]], 0)

                    point_clouds = {}
                    if export_point_cloud:
                        point_clouds = build_point_clouds(
                            value, micro_taskvar_dir, episode_key, voxel_size
                        )

                    if keep_stop_action:
                        end_idx = None
                    else:
                        end_idx = -1
                        demo_len = demo_len - 1

                    episode = {
                        "observation.images.left_shoulder_image": value["rgb"][
                            :end_idx, 0
                        ],
                        "observation.images.right_shoulder_image": value["rgb"][
                            :end_idx, 1
                        ],
                        "observation.images.wrist_image": value["rgb"][:end_idx, 2],
                        "observation.images.front_image": value["rgb"][:end_idx, 3],
                        "observation.state": np.array(
                            state[:end_idx], dtype=np.float32
                        ),
                        "action": np.array(action[:end_idx], dtype=np.float32),
                    }
                    for field, values in point_clouds.items():
                        episode[field] = values[:end_idx]

                    yield (
                        taskvar,
                        [
                            {key: value[i] for key, value in episode.items()}
                            for i in range(demo_len)
                        ],
                    )


def open_point_cloud_lmdbs(repo_dir):
    return {
        field: lmdb.open(
            os.path.join(repo_dir, spec["dirname"]),
            map_size=int(1024**4),
        )
        for field, spec in POINT_CLOUD_OUTPUTS.items()
    }


def write_point_clouds(point_lmdb_envs, point_key, point_clouds):
    for field, point_cloud in point_clouds.items():
        with point_lmdb_envs[field].begin(write=True) as txn:
            txn.put(point_key.encode("ascii"), msgpack.packb(point_cloud))


def print_point_stats(point_counts):
    for field, counts in point_counts.items():
        if not counts:
            continue
        counts = np.array(counts)
        label = POINT_CLOUD_OUTPUTS[field]["label"]
        print(
            f"npoints {label}",
            np.min(counts),
            np.max(counts),
            np.mean(counts),
        )


def remove_existing_repo_dir(repo_dir):
    if not os.path.exists(repo_dir):
        return

    answer = input(f"{repo_dir} exists. Delete it? Type y to confirm: ")
    if answer.strip() != "y":
        raise SystemExit(f"Abort: {repo_dir} already exists.")
    shutil.rmtree(repo_dir)


def create_lerobot_dataset(args, repo_dir):
    supported = set(inspect.signature(LeRobotDataset.create).parameters)
    create_kwargs = {
        "repo_id": args.repo_id,
        "root": repo_dir,
        "fps": 20,
        "robot_type": "franka",
        "features": RLBENCH_FEATURES,
        "image_writer_processes": args.image_writer_processes,
        "image_writer_threads": args.image_writer_threads,
    }
    return LeRobotDataset.create(
        **{key: value for key, value in create_kwargs.items() if key in supported}
    )


def main(args):

    repo_dir = os.path.join(args.output_dir, args.repo_id)
    remove_existing_repo_dir(repo_dir)

    dataset = create_lerobot_dataset(args, repo_dir)

    with open(args.task_instruction_file) as f:
        taskvar2instrs = json.load(f)

    taskvars = resolve_taskvars(
        args.keystep_dir,
        taskvars=args.taskvars,
        taskvar_file=args.taskvar_file,
    )

    for taskvar in taskvars:
        if taskvar not in taskvar2instrs:
            print(taskvar)
        assert taskvar in taskvar2instrs

    raw_dataset = load_keystep_episodes(
        args.keystep_dir,
        taskvars=taskvars,
        keep_stop_action=args.keep_stop_action,
        export_point_cloud=args.export_point_cloud,
        microstep_dir=args.microstep_dir,
        voxel_size=args.voxel_size,
    )

    point_lmdb_envs = {}
    point_counts = {}
    if args.export_point_cloud:
        point_lmdb_envs = open_point_cloud_lmdbs(repo_dir)
        point_counts = {field: [] for field in POINT_CLOUD_OUTPUTS}

    try:
        for episode_index, (taskvar, episode_data) in enumerate(raw_dataset):
            # multiple instructions: use special token <br> to separate sentences
            task_instruction = "<br>".join(taskvar2instrs[taskvar])

            for frame_data in episode_data:
                point_clouds = {}
                if args.export_point_cloud:
                    point_clouds = {
                        field: frame_data.pop(field)
                        for field in POINT_CLOUD_OUTPUTS
                    }
                    for field, point_cloud in point_clouds.items():
                        point_counts[field].append(len(point_cloud))

                dataset.add_frame(
                    frame_data,
                    task=task_instruction,
                )

                if args.export_point_cloud:
                    episode_buffer = dataset.episode_buffer
                    point_key = "{episode_id}-{step_id}".format(
                        episode_id=episode_buffer["episode_index"],
                        step_id=episode_buffer["frame_index"][-1],
                    )
                    write_point_clouds(point_lmdb_envs, point_key, point_clouds)

            dataset.save_episode()
            print(
                f"process done for {dataset.repo_id}, "
                f"episode {episode_index}, len {len(episode_data)}"
            )
            if args.export_point_cloud:
                print_point_stats(point_counts)

        if args.export_point_cloud:
            total_points = sum(len(counts) for counts in point_counts.values())
            print("#data", total_points // len(POINT_CLOUD_OUTPUTS))
            print_point_stats(point_counts)
    finally:
        for env in point_lmdb_envs.values():
            env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keystep_dir", required=True)
    parser.add_argument("--task_instruction_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--microstep_dir", default=None)
    parser.add_argument("--taskvar_file", default=None, type=str)
    parser.add_argument("--taskvars", default=None, nargs="+", type=str)
    parser.add_argument("--keep_stop_action", action="store_true", default=False)
    parser.add_argument("--export_point_cloud", action="store_true", default=False)
    parser.add_argument(
        "--voxel_size",
        "--voxel-size",
        dest="voxel_size",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--image_writer_processes",
        "--image-writer-processes",
        dest="image_writer_processes",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--image_writer_threads",
        "--image-writer-threads",
        dest="image_writer_threads",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    if args.export_point_cloud and args.microstep_dir is None:
        parser.error("--microstep_dir is required when --export_point_cloud is set")
    if args.voxel_size <= 0:
        parser.error("--voxel_size must be > 0")
    if args.image_writer_processes < 0:
        parser.error("--image_writer_processes must be >= 0")
    if args.image_writer_threads < 0:
        parser.error("--image_writer_threads must be >= 0")
    if args.image_writer_processes > 0 and args.image_writer_threads == 0:
        parser.error(
            "--image_writer_threads must be > 0 when "
            "--image_writer_processes > 0"
        )

    return args


if __name__ == "__main__":
    main(parse_args())
