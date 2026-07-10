from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
from tqdm.auto import tqdm

from pointact.robot_envs.robocasa365_utils.environments import RoboCasa365Env


CAMERA_NAMES = (
    "left",
    "right",
    "wrist",
)
DEFAULT_WORKSPACE = {
    "X_BBOX": [-0.8, 0.8],
    "Y_BBOX": [-0.8, 0.8],
    "Z_BBOX": [0.0, 1.0],
}


def dump_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def episode_cache_path(cache_dir: Path, episode_index: int) -> Path:
    return cache_dir / "episodes" / f"episode_{episode_index:06d}.npz"


def episode_points_path(cache_dir: Path, episode_index: int) -> Path:
    return cache_dir / "episodes" / f"episode_{episode_index:06d}_points.npy"


def get_episode_parquet_path(input_dir: Path, episode_index: int) -> Path:
    info = json.loads((input_dir / "meta" / "info.json").read_text())
    chunk_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunk_size
    rel_path = info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    return input_dir / rel_path


def read_episode_column_from_parquet(input_dir: Path, episode_index: int, column: str) -> list[Any]:
    parquet_path = get_episode_parquet_path(input_dir, episode_index)
    table = pq.read_table(parquet_path, columns=[column])
    return table.column(column).to_pylist()


def load_source_actions(
    input_dir: Path,
    episode_index: int,
) -> dict[str, Any]:
    actions = read_episode_column_from_parquet(input_dir, episode_index, "action")
    actions = np.stack([np.asarray(action, dtype=np.float32).reshape(-1) for action in actions], axis=0)
    return {
        "actions_lerobot": actions,
        "source_length": len(actions),
    }


def get_empty_quat_action() -> np.ndarray:
    """Return a no-op 13D action; quaternion uses xyzw identity, not all zeros."""
    action = np.zeros(13, dtype=np.float32)
    action[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return action


def reorder_source_action_to_env(action: np.ndarray) -> np.ndarray:
    """Convert source 12D axis-angle action to public 13D xyzw quaternion action.

    Source order: base_motion(4), control_mode(1), eef_position(3),
    eef_rotation_axisangle(3), gripper_close(1).

    Output/env/dataset order: eef_position(3), eef_quat_xyzw(4),
    gripper_close(1), base_motion(4), control_mode(1).
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape[0] != 12:
        raise ValueError(f"Expected source action shape (12,), got {action.shape}")
    base_motion = action[0:4]
    control_mode = action[4:5]
    eef_position = action[5:8]
    eef_quat = R.from_rotvec(action[8:11]).as_quat().astype(np.float32)
    gripper_close = action[11:12]
    return np.concatenate(
        [eef_position, eef_quat, gripper_close, base_motion, control_mode],
        axis=0,
    ).astype(np.float32)


def convert_env_action_to_dataset_action(action_env: np.ndarray) -> np.ndarray:
    """Env action and saved dataset action share the public 13D xyzw format."""
    action_env = np.asarray(action_env, dtype=np.float32).reshape(-1)
    if action_env.shape[0] != 13:
        raise ValueError(f"Expected env action shape (13,), got {action_env.shape}")
    return action_env.astype(np.float32, copy=True)


def infer_env_name(input_dir: Path) -> str:
    dataset_meta_path = input_dir / "extras" / "dataset_meta.json"
    assert dataset_meta_path.exists(), f"File not found: {dataset_meta_path}"
    dataset_meta = json.loads(dataset_meta_path.read_text())
    env_name = (
        dataset_meta.get("env")
        or dataset_meta.get("env_info", {}).get("env_name")
        or dataset_meta.get("env_args", {}).get("env_name")
    )
    return str(env_name)


def infer_fps(input_dir: Path) -> int:
    info_path = input_dir / "meta" / "info.json"
    if not info_path.exists():
        return 20
    info = json.loads(info_path.read_text())
    return int(info.get("fps", 20))


def build_cache_meta(input_dir: Path, env_name: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "input_dir": str(input_dir),
        "env_name": env_name,
        "fps": infer_fps(input_dir),
        "camera_names": list(CAMERA_NAMES),
        "image_resolution": args.image_resolution,
        "point_cache_format": "episode_npz_plus_points_npy",
        "point_cloud_dirname": "points_3views",
        "point_cloud_frame": "robot_base",
        "point_cloud_columns": ["x", "y", "z", "r", "g", "b"],
        "voxel_size": args.voxel_size,
        "workspace": DEFAULT_WORKSPACE,
        "action_source_order": "base_motion,control_mode,eef_position,eef_rotation_axisangle,gripper_close",
        "action_env_order": "eef_position,eef_rotation_quat_xyzw,gripper_close,base_motion,control_mode",
        "action_dataset_order": "eef_position,eef_rotation_quat_xyzw,gripper_close,base_motion,control_mode",
    }


def validate_resume_cache_meta(cache_meta_path: Path, expected_meta: dict[str, Any]) -> None:
    if not cache_meta_path.exists():
        return

    existing_meta = json.loads(cache_meta_path.read_text())
    checked_keys = (
        "input_dir",
        "env_name",
        "camera_names",
        "image_resolution",
        "point_cache_format",
        "point_cloud_dirname",
        "point_cloud_frame",
        "point_cloud_columns",
        "voxel_size",
        "workspace",
        "action_source_order",
        "action_env_order",
        "action_dataset_order",
    )
    mismatches = {
        key: {"existing": existing_meta.get(key), "current": expected_meta.get(key)}
        for key in checked_keys
        if existing_meta.get(key) != expected_meta.get(key)
    }
    if mismatches:
        raise ValueError(
            f"Cannot resume from {cache_meta_path}: cache settings differ from current args: {mismatches}"
        )


def make_point_cloud(obs: dict[str, Any], voxel_size: float) -> np.ndarray:
    clouds = []
    for camera_name in CAMERA_NAMES:
        rgb = obs[f"observation.images.{camera_name}_image"]
        xyz = obs[f"observation.points.{camera_name}"].astype(np.float32)
        cloud = np.concatenate([xyz, rgb.astype(np.float32) / 255.0], axis=-1).reshape(-1, 6)
        clouds.append(cloud)

    point_cloud = np.concatenate(clouds, axis=0)
    point_cloud = crop_point_cloud_by_workspace(point_cloud, DEFAULT_WORKSPACE)

    if len(point_cloud) == 0:
        return np.empty((0, 6), dtype=np.float32)

    return voxel_downsample(point_cloud, voxel_size=voxel_size)


def crop_point_cloud_by_workspace(point_cloud: np.ndarray, workspace: dict[str, list[float]]) -> np.ndarray:
    mask = (
        (point_cloud[:, 0] > workspace["X_BBOX"][0])
        & (point_cloud[:, 0] < workspace["X_BBOX"][1])
        & (point_cloud[:, 1] > workspace["Y_BBOX"][0])
        & (point_cloud[:, 1] < workspace["Y_BBOX"][1])
        & (point_cloud[:, 2] > workspace["Z_BBOX"][0])
        & (point_cloud[:, 2] < workspace["Z_BBOX"][1])
    )
    return point_cloud[mask]


def voxel_downsample(point_cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0 or len(point_cloud) == 0:
        return point_cloud.astype(np.float32, copy=False)

    voxel_indices = np.floor(point_cloud[:, :3] / float(voxel_size)).astype(np.int64)
    _, inverse, counts = np.unique(voxel_indices, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), point_cloud.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, point_cloud.astype(np.float64))
    return (sums / counts[:, None]).astype(np.float32)


def frame_cache_from_obs(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    frame = {
        "observation_state": np.asarray(obs["observation.state"], dtype=np.float32),
    }
    for camera_name in CAMERA_NAMES:
        frame[f"image_{camera_name}"] = np.asarray(
            obs[f"observation.images.{camera_name}_image"],
            dtype=np.uint8,
        )
    return frame


def save_episode_cache(path: Path, episode: dict[str, Any], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_fn = np.savez_compressed if compress else np.savez
    save_fn(path, **episode)


def load_completed_episode_summary(
    cache_dir: Path,
    episode_index: int,
) -> dict[str, Any] | None:
    cache_path = episode_cache_path(cache_dir, episode_index)
    points_path = episode_points_path(cache_dir, episode_index)
    if not cache_path.exists() or not points_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=True) as data:
            files = set(data.files)
            required_keys = {
                "observation_state",
                "action",
                "next_reward",
                "next_done",
                "point_counts",
                "point_cloud_offsets",
                *{f"image_{camera_name}" for camera_name in CAMERA_NAMES},
            }
            if not required_keys.issubset(files):
                return None

            states = data["observation_state"]
            actions = data["action"]
            rewards = data["next_reward"]
            dones = data["next_done"]
            point_counts = data["point_counts"]
            point_cloud_offsets = data["point_cloud_offsets"]
            lengths = {
                "observation_state": len(states),
                "action": len(actions),
                "next_reward": len(rewards),
                "next_done": len(dones),
                "point_counts": len(point_counts),
                **{f"image_{camera_name}": len(data[f"image_{camera_name}"]) for camera_name in CAMERA_NAMES},
            }
            if len(set(lengths.values())) != 1:
                return None

            replay_length = len(states)
            point_cloud_points = np.load(points_path, mmap_mode="r")
            if point_cloud_points.ndim != 2 or point_cloud_points.shape[1] != 6:
                return None
            if len(point_cloud_offsets) != replay_length + 1:
                return None
            if int(point_cloud_offsets[0]) != 0 or int(point_cloud_offsets[-1]) != len(point_cloud_points):
                return None
            if not np.array_equal(np.diff(point_cloud_offsets), point_counts):
                return None

            return {
                "episode_index": episode_index,
                "source_length": int(data["source_length"].reshape(-1)[0]) if "source_length" in files else replay_length,
                "replay_length": replay_length,
                "task": str(data["task"].item()) if "task" in files else "",
                "success": bool(data["success"].reshape(-1)[0]) if "success" in files else bool(np.any(rewards > 0)),
                "success_frame": int(data["success_frame"].reshape(-1)[0]) if "success_frame" in files else -1,
                "last_reward": float(rewards[-1]) if len(rewards) else 0.0,
                "point_count_min": int(np.min(point_counts)) if len(point_counts) else 0,
                "point_count_max": int(np.max(point_counts)) if len(point_counts) else 0,
                "point_count_mean": float(np.mean(point_counts)) if len(point_counts) else 0.0,
            }
    except Exception:
        return None


def replay_episode(
    robocasa_env: Any,
    input_dir: Path,
    cache_dir: Path,
    episode_index: int,
    action_info: dict[str, Any],
    voxel_size: float,
    compress: bool,
) -> dict[str, Any]:
    episode_dir = input_dir / "extras" / f"episode_{episode_index:06d}"
    obs, _ = robocasa_env.reset(initial_state_dir=episode_dir, step_after_reset=False)
    task = str(obs.get("task", ""))

    source_actions = action_info["actions_lerobot"]
    action_envs = np.stack([reorder_source_action_to_env(action) for action in source_actions], axis=0)

    images: dict[str, list[np.ndarray]] = {f"image_{camera_name}": [] for camera_name in CAMERA_NAMES}
    states = []
    saved_actions = []
    rewards = []
    dones = []
    point_clouds = []
    point_counts = []
    success = False
    success_frame = -1

    step_iter = tqdm(
        enumerate(action_envs),
        total=len(action_envs),
        desc=f"episode {episode_index:06d}",
        unit="step",
        leave=False,
        dynamic_ncols=True,
    )
    for frame_index, action_env in step_iter:
        action_env = convert_env_action_to_dataset_action(action_env)
        frame = frame_cache_from_obs(obs)
        states.append(frame["observation_state"])
        saved_actions.append(action_env)
        for camera_name in CAMERA_NAMES:
            images[f"image_{camera_name}"].append(frame[f"image_{camera_name}"])

        point_cloud = make_point_cloud(obs, voxel_size=voxel_size)
        point_clouds.append(point_cloud)
        point_counts.append(int(len(point_cloud)))

        obs, reward, done, info = robocasa_env.step(action_env)
        reward = float(reward)
        done = bool(done or info.get("success", False))
        if info.get("success", False) and not success:
            success = True
            success_frame = frame_index
        rewards.append(reward)
        dones.append(done)
        step_iter.set_postfix(
            reward=f"{reward:.1f}",
            success=success,
            points=point_counts[-1],
        )

        if done:
            stop_action = get_empty_quat_action()
            terminal_frame = frame_cache_from_obs(obs)
            states.append(terminal_frame["observation_state"])
            saved_actions.append(stop_action)
            for camera_name in CAMERA_NAMES:
                images[f"image_{camera_name}"].append(terminal_frame[f"image_{camera_name}"])

            point_cloud = make_point_cloud(obs, voxel_size=voxel_size)
            point_clouds.append(point_cloud)
            point_counts.append(int(len(point_cloud)))

            # The terminal observation has no source action; repeat terminal flags and
            # use an identity-quaternion no-op action so all frame arrays stay aligned.
            rewards.append(reward)
            dones.append(True)
            break

    states_arr = np.stack(states, axis=0).astype(np.float32)
    actions_arr = np.stack(saved_actions, axis=0).astype(np.float32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=bool)
    point_counts_arr = np.asarray(point_counts, dtype=np.int64)
    if len(dones_arr) > 0 and success:
        dones_arr[-1] = True

    lengths = {
        "observation_state": len(states_arr),
        "action": len(actions_arr),
        "next_reward": len(rewards_arr),
        "next_done": len(dones_arr),
        "point_counts": len(point_counts_arr),
        **{key: len(value) for key, value in images.items()},
    }
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Replay cache length mismatch for episode {episode_index}: {lengths}")

    point_cloud_offsets = np.empty(len(point_counts_arr) + 1, dtype=np.int64)
    point_cloud_offsets[0] = 0
    np.cumsum(point_counts_arr, out=point_cloud_offsets[1:])
    if point_clouds:
        point_cloud_points = np.concatenate(point_clouds, axis=0).astype(np.float32, copy=False)
    else:
        point_cloud_points = np.empty((0, 6), dtype=np.float32)
    if int(point_cloud_offsets[-1]) != len(point_cloud_points):
        raise RuntimeError(
            f"Point cloud offset mismatch for episode {episode_index}: "
            f"offsets[-1]={point_cloud_offsets[-1]} points={len(point_cloud_points)}"
        )

    episode_cache = {
        "episode_index": np.asarray([episode_index], dtype=np.int64),
        "source_length": np.asarray([int(action_info["source_length"])], dtype=np.int64),
        "task": np.asarray(task),
        "observation_state": states_arr,
        "action": actions_arr,
        "action_env": actions_arr,
        "next_reward": rewards_arr,
        "next_done": dones_arr,
        "success": np.asarray([success], dtype=bool),
        "success_frame": np.asarray([success_frame], dtype=np.int64),
        "point_cloud_offsets": point_cloud_offsets,
        "point_counts": point_counts_arr,
    }
    for key, value in images.items():
        episode_cache[key] = np.stack(value, axis=0).astype(np.uint8)

    save_episode_cache(episode_cache_path(cache_dir, episode_index), episode_cache, compress)
    np.save(episode_points_path(cache_dir, episode_index), point_cloud_points)

    return {
        "episode_index": episode_index,
        "source_length": int(action_info["source_length"]),
        "replay_length": int(len(states_arr)),
        "task": task,
        "success": bool(success),
        "success_frame": int(success_frame),
        "last_reward": float(rewards_arr[-1]) if len(rewards_arr) else 0.0,
        "point_count_min": int(np.min(point_counts_arr)) if len(point_counts_arr) else 0,
        "point_count_max": int(np.max(point_counts_arr)) if len(point_counts_arr) else 0,
        "point_count_mean": float(np.mean(point_counts_arr)) if len(point_counts_arr) else 0.0,
    }


def parse_episode_ids(raw_episodes: list[int] | None, input_dir: Path) -> list[int]:
    if raw_episodes:
        return sorted(int(x) for x in raw_episodes)
    info = json.loads((input_dir / "meta" / "info.json").read_text())
    return list(range(int(info["total_episodes"])))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay RoboCasa365 demonstrations with action rollout.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--episodes", nargs="*", type=int, default=None)
    parser.add_argument("--split", type=str, default="target", choices=["pretrain", "target"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-resolution", type=int, default=256)
    parser.add_argument("--voxel-size", type=float, default=0.01)
    parser.add_argument("--compress", action="store_true")
    args = parser.parse_args()
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together.")

    input_dir = args.input_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    if cache_dir.exists() and args.overwrite:
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "episodes").mkdir(parents=True, exist_ok=True)
    summary_path = cache_dir / "replay_summary.jsonl"
    if summary_path.exists() and args.overwrite:
        summary_path.unlink()

    episode_ids = parse_episode_ids(args.episodes, input_dir)
    env_name = infer_env_name(input_dir)

    cache_meta_path = cache_dir / "cache_meta.json"
    cache_meta = build_cache_meta(input_dir=input_dir, env_name=env_name, args=args)
    if args.resume and cache_meta_path.exists():
        validate_resume_cache_meta(cache_meta_path, cache_meta)
    else:
        dump_json(cache_meta_path, cache_meta)

    robocasa_env = None
    try:
        episode_iter = tqdm(
            enumerate(episode_ids, start=1),
            total=len(episode_ids),
            desc="episodes",
            unit="episode",
            dynamic_ncols=True,
        )
        for offset, episode_index in episode_iter:
            if args.resume:
                cached_summary = load_completed_episode_summary(
                    cache_dir=cache_dir,
                    episode_index=episode_index,
                )
                if cached_summary is not None:
                    episode_iter.set_postfix(
                        skipped=True,
                        success=cached_summary["success"],
                        length=cached_summary["replay_length"],
                    )
                    tqdm.write(
                        f"[{offset}/{len(episode_ids)}] skip episode {episode_index:06d} "
                        f"len={cached_summary['replay_length']} success={cached_summary['success']}"
                    )
                    continue
                if episode_cache_path(cache_dir, episode_index).exists():
                    tqdm.write(f"[{offset}/{len(episode_ids)}] rerun incomplete episode {episode_index:06d}")

            if robocasa_env is None:
                robocasa_env = RoboCasa365Env(
                    env_name=env_name,
                    split=args.split,
                    seed=args.seed,
                    image_resolution=args.image_resolution,
                    use_depth=True,
                    use_point_cloud=True,
                    enable_render=True,
                    terminate_on_success=False,
                )

            action_info = load_source_actions(
                input_dir=input_dir,
                episode_index=episode_index,
            )
            summary = replay_episode(
                robocasa_env=robocasa_env,
                input_dir=input_dir,
                cache_dir=cache_dir,
                episode_index=episode_index,
                action_info=action_info,
                voxel_size=args.voxel_size,
                compress=args.compress,
            )
            append_jsonl(summary_path, summary)
            episode_iter.set_postfix(
                success=summary["success"],
                length=summary["replay_length"],
                points=f"{summary['point_count_mean']:.1f}",
            )
            tqdm.write(
                f"[{offset}/{len(episode_ids)}] episode {episode_index:06d} "
                f"len={summary['replay_length']} success={summary['success']} "
                f"points={summary['point_count_min']}/{summary['point_count_mean']:.1f}/{summary['point_count_max']}"
            )
    finally:
        if robocasa_env is not None:
            robocasa_env.close()


if __name__ == "__main__":
    main()
