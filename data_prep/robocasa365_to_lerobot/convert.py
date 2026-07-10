from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Any

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm.auto import tqdm

msgpack_numpy.patch()


CAMERA_NAMES = (
    "left",
    "right",
    "wrist",
)

STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qx",
    "eef_qy",
    "eef_qz",
    "eef_qw",
    "gripper_0",
    "gripper_1",
    "base_x",
    "base_y",
    "base_z",
    "base_qx",
    "base_qy",
    "base_qz",
    "base_qw",
]
ACTION_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qx",
    "eef_qy",
    "eef_qz",
    "eef_qw",
    "gripper_close",
    "base_x",
    "base_y",
    "base_yaw",
    "torso",
    "control_mode",
]


def build_features(image_resolution: int) -> dict[str, dict[str, Any]]:
    features = {
        f"observation.images.{camera_name}_image": {
            "dtype": "video",
            "shape": (image_resolution, image_resolution, 3),
            "names": ["height", "width", "channel"],
        }
        for camera_name in CAMERA_NAMES
    }
    features.update(
        {
            "observation.state": {
                "dtype": "float32",
                "shape": (16,),
                "names": {"motors": STATE_NAMES},
            },
            "action": {
                "dtype": "float32",
                "shape": (13,),
                "names": {"motors": ACTION_NAMES},
            },
            "next.reward": {
                "dtype": "float32",
                "shape": (1,),
            },
            "next.done": {
                "dtype": "bool",
                "shape": (1,),
            },
        }
    )
    return features


def read_cache_meta(cache_dir: Path) -> dict[str, Any]:
    meta_path = cache_dir / "cache_meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def cache_episode_files(cache_dir: Path, episodes: list[int] | None) -> list[Path]:
    episode_dir = cache_dir / "episodes"
    if episodes:
        files = [episode_dir / f"episode_{int(episode):06d}.npz" for episode in episodes]
    else:
        files = sorted(episode_dir.glob("episode_*.npz"))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing replay cache files: {missing[:5]}")
    return files


def cache_episode_points_path(episode_file: Path) -> Path:
    return episode_file.with_name(f"{episode_file.stem}_points.npy")


def infer_episode_index_from_cache_file(episode_file: Path) -> int:
    return int(episode_file.stem.split("_")[-1])


def validate_cached_episode_lengths(
    episode_file: Path,
    image_arrays: dict[str, np.ndarray],
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    point_counts: np.ndarray | None,
    point_cloud_offsets: np.ndarray,
    point_cloud_points: np.ndarray,
) -> None:
    lengths = {
        "observation_state": len(states),
        "action": len(actions),
        "next_reward": len(rewards),
        "next_done": len(dones),
        **{f"image_{camera_name}": len(image_arrays[camera_name]) for camera_name in CAMERA_NAMES},
    }
    if point_counts is not None:
        lengths["point_counts"] = len(point_counts)
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{episode_file}: cached timestep length mismatch {lengths}")
    if point_counts is None:
        raise ValueError(f"{episode_file}: missing point_counts")
    if len(point_cloud_offsets) != len(states) + 1:
        raise ValueError(
            f"{episode_file}: point_cloud_offsets length mismatch "
            f"{len(point_cloud_offsets)} != {len(states) + 1}"
        )
    if int(point_cloud_offsets[0]) != 0 or int(point_cloud_offsets[-1]) != len(point_cloud_points):
        raise ValueError(
            f"{episode_file}: point cloud offset bounds mismatch "
            f"offsets[0]={point_cloud_offsets[0]} offsets[-1]={point_cloud_offsets[-1]} "
            f"points={len(point_cloud_points)}"
        )
    if not np.array_equal(np.diff(point_cloud_offsets), point_counts):
        raise ValueError(f"{episode_file}: point_counts do not match point_cloud_offsets")
    if point_cloud_points.ndim != 2 or point_cloud_points.shape[1] != 6:
        raise ValueError(f"{episode_file}: expected point cloud shape (N, 6), got {point_cloud_points.shape}")


def copy_cache_metadata(cache_dir: Path, repo_dir: Path) -> None:
    for name in ("cache_meta.json", "replay_summary.jsonl"):
        src = cache_dir / name
        if src.exists():
            shutil.copy2(src, repo_dir / name)


def create_lerobot_dataset(args: argparse.Namespace, repo_dir: Path, features: dict[str, dict[str, Any]], fps: int):
    supported = set(inspect.signature(LeRobotDataset.create).parameters)
    create_kwargs = {
        "repo_id": args.repo_id,
        "root": repo_dir,
        "fps": fps,
        "robot_type": args.robot_type,
        "features": features,
        "image_writer_processes": args.image_writer_processes,
        "image_writer_threads": args.image_writer_threads,
    }

    return LeRobotDataset.create(**{key: value for key, value in create_kwargs.items() if key in supported})


def save_episode_or_raise(dataset: LeRobotDataset, episode_length: int, source_episode_index: int) -> None:
    before_episodes = int(dataset.meta.total_episodes)
    before_frames = int(dataset.meta.total_frames)
    dataset.save_episode()

    after_episodes = int(dataset.meta.total_episodes)
    after_frames = int(dataset.meta.total_frames)
    if after_episodes != before_episodes + 1 or after_frames != before_frames + episode_length:
        raise RuntimeError(
            f"dataset.save_episode() did not advance metadata as expected for source episode "
            f"{source_episode_index:06d}: episodes {before_episodes}->{after_episodes}, "
            f"frames {before_frames}->{after_frames}, expected +1 episode and +{episode_length} frames."
        )


def write_episode_points_to_lmdb(
    point_out_env,
    point_cloud_points: np.ndarray,
    pending_points: list[tuple[str, int, int]],
    lmdb_commit_every: int,
) -> None:
    out_txn = None
    frames_since_commit = 0
    written_keys = []
    try:
        out_txn = point_out_env.begin(write=True)
        for frame_index, (output_point_key, point_start, point_end) in enumerate(pending_points):
            point_cloud = np.asarray(point_cloud_points[point_start:point_end], dtype=np.float32)
            out_txn.put(output_point_key.encode("ascii"), msgpack.packb(point_cloud))
            written_keys.append(output_point_key)
            frames_since_commit += 1

            if lmdb_commit_every > 0 and frames_since_commit >= lmdb_commit_every:
                out_txn.commit()
                out_txn = None
                if frame_index + 1 < len(pending_points):
                    out_txn = point_out_env.begin(write=True)
                frames_since_commit = 0

        if out_txn is not None:
            out_txn.commit()
            out_txn = None
    except Exception:
        if out_txn is not None:
            out_txn.abort()
        if written_keys:
            with point_out_env.begin(write=True) as cleanup_txn:
                for output_point_key in written_keys:
                    cleanup_txn.delete(output_point_key.encode("ascii"))
        raise


def convert_episode_cache_to_lerobot(
    dataset: LeRobotDataset,
    point_out_env,
    episode_file: Path,
    lmdb_commit_every: int,
) -> dict[str, Any]:
    points_file = cache_episode_points_path(episode_file)
    if not points_file.exists():
        raise FileNotFoundError(f"Missing replay point cloud file: {points_file}")

    point_cloud_points = np.load(points_file, mmap_mode="r")
    with np.load(episode_file, allow_pickle=True) as data:
        source_episode_index = int(data["episode_index"].reshape(-1)[0])
        task = str(data["task"].item())
        states = data["observation_state"].astype(np.float32)
        actions = data["action"].astype(np.float32)
        rewards = data["next_reward"].astype(np.float32)
        dones = data["next_done"].astype(bool)
        point_counts = data["point_counts"].astype(np.int64) if "point_counts" in data else None
        point_cloud_offsets = data["point_cloud_offsets"].astype(np.int64)
        image_arrays = {
            camera_name: data[f"image_{camera_name}"]
            for camera_name in CAMERA_NAMES
        }

        validate_cached_episode_lengths(
            episode_file=episode_file,
            image_arrays=image_arrays,
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            point_counts=point_counts,
            point_cloud_offsets=point_cloud_offsets,
            point_cloud_points=point_cloud_points,
        )

        pending_points = []
        for frame_index in range(len(states)):
            frame_data = {
                "observation.state": states[frame_index],
                "action": actions[frame_index],
                "next.reward": np.asarray([rewards[frame_index]], dtype=np.float32),
                "next.done": np.asarray([dones[frame_index]], dtype=bool),
            }
            for camera_name in CAMERA_NAMES:
                frame_data[f"observation.images.{camera_name}_image"] = image_arrays[camera_name][frame_index]

            dataset.add_frame(frame_data, task=task)

            episode_buffer = dataset.episode_buffer
            output_point_key = "{episode_id}-{step_id}".format(
                episode_id=episode_buffer["episode_index"],
                step_id=episode_buffer["frame_index"][-1],
            )
            pending_points.append(
                (
                    output_point_key,
                    int(point_cloud_offsets[frame_index]),
                    int(point_cloud_offsets[frame_index + 1]),
                )
            )

        save_episode_or_raise(dataset, episode_length=len(states), source_episode_index=source_episode_index)
        write_episode_points_to_lmdb(
            point_out_env=point_out_env,
            point_cloud_points=point_cloud_points,
            pending_points=pending_points,
            lmdb_commit_every=lmdb_commit_every,
        )

        return {
            "source_episode_index": source_episode_index,
            "length": len(states),
            "task": task,
            "point_counts": point_counts,
        }


def convert_cache_to_lerobot(args: argparse.Namespace) -> None:
    cache_dir = args.cache_dir.expanduser().resolve()
    repo_id = args.repo_id
    repo_dir = args.output_dir.expanduser().resolve()
    if repo_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{repo_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    cache_meta = read_cache_meta(cache_dir)
    image_resolution = int(args.image_resolution or cache_meta.get("image_resolution", 256))
    point_cloud_dirname = args.point_cloud_dirname or cache_meta.get("point_cloud_dirname", "points_3views")

    dataset = create_lerobot_dataset(
        args=args,
        repo_dir=repo_dir,
        fps=int(args.fps or cache_meta.get("fps", 20)),
        features=build_features(image_resolution),
    )

    point_out_env = lmdb.open(str(repo_dir / point_cloud_dirname), map_size=int(args.lmdb_map_size))

    npoints = []
    failed_episode_ids = []
    try:
        episode_files = cache_episode_files(cache_dir, args.episodes)
        episode_iter = tqdm(
            enumerate(episode_files),
            total=len(episode_files),
            desc="episodes",
            unit="episode",
            dynamic_ncols=True,
        )
        for output_episode_index, episode_file in episode_iter:
            source_episode_index = infer_episode_index_from_cache_file(episode_file)
            try:
                summary = convert_episode_cache_to_lerobot(
                    dataset=dataset,
                    point_out_env=point_out_env,
                    episode_file=episode_file,
                    lmdb_commit_every=args.lmdb_commit_every,
                )
            except Exception:
                if source_episode_index not in failed_episode_ids:
                    failed_episode_ids.append(source_episode_index)
                raise

            source_episode_index = int(summary["source_episode_index"])
            point_counts = summary["point_counts"]
            if point_counts is not None:
                npoints.extend(int(count) for count in point_counts)

            episode_iter.set_postfix(source=f"{source_episode_index:06d}", length=summary["length"])
            tqdm.write(
                f"[{output_episode_index + 1}/{len(episode_files)}] saved episode "
                f"source={source_episode_index:06d}, len={summary['length']}, task={summary['task']!r}"
            )
    finally:
        point_out_env.close()
        if failed_episode_ids:
            print("failed episode ids:", " ".join(f"{episode_id:06d}" for episode_id in failed_episode_ids))

    copy_cache_metadata(cache_dir, repo_dir)
    if npoints:
        print(
            "point counts min/max/mean:",
            int(np.min(npoints)),
            int(np.max(npoints)),
            float(np.mean(npoints)),
        )
    print(f"wrote LeRobot dataset to {repo_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert RoboCasa365 replay cache to LeRobot + point LMDB.")
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--episodes", nargs="*", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--point-cloud-dirname", default=None)
    parser.add_argument("--image-resolution", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--robot-type", default="PandaOmron")
    parser.add_argument("--lmdb-map-size", type=int, default=int(1024**4))
    parser.add_argument(
        "--lmdb-commit-every",
        type=int,
        default=50,
        help="Commit point-cloud LMDB every N frames. 0 means one transaction per episode.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Async LeRobot image writer processes. 0 uses threads only.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=12,
        help="Async LeRobot image writer threads. 0 disables async image writing when processes is also 0.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.lmdb_commit_every < 0:
        parser.error("--lmdb-commit-every must be >= 0.")
    if args.image_writer_processes < 0:
        parser.error("--image-writer-processes must be >= 0.")
    if args.image_writer_threads < 0:
        parser.error("--image-writer-threads must be >= 0.")
    if args.image_writer_processes > 0 and args.image_writer_threads == 0:
        parser.error("--image-writer-threads must be > 0 when --image-writer-processes > 0.")
    convert_cache_to_lerobot(args)


if __name__ == "__main__":
    main()
