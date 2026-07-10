"""Compute robot state/action statistics from LeRobot parquet files.

Example:
python data_prep/prepare_robot_state_action_stats.py \
    --dataset_dirs robot_data/rlbench/lerobot_point_lmdb/task_a robot_data/rlbench/lerobot_point_lmdb/task_b \
    --output_file robot_data/rlbench/lerobot_point_lmdb/robot_state_action_stats/eef_rot6d.json \
    --point_cloud_dir points_frontview \
    --state_xyz_slice 0 3 \
    --action_xyz_slice 0 3 \
    --state_rotation_slice 3 7 \
    --action_rotation_slice 3 7 \
    --rotation_type quat \
    --target_rotation_type rot6d

By default this reads:
  - observation.state
  - action
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
from tqdm import tqdm

from pointact.utils.rotation import convert_rotation

msgpack_numpy.patch()


DEFAULT_STATE_KEYS = ("observation.state",)
DEFAULT_ACTION_KEYS = ("action",)
QUANTILES = {
    "q01": 0.01,
    "q10": 0.10,
    "q50": 0.50,
    "q90": 0.90,
    "q99": 0.99,
}
ROTATION_DIMS = {
    "quat": 4,
    "euler": 3,
    "rot6d": 6,
    "axisangle": 3,
    "matrix": 9,
}
ROTATION_TYPES = tuple(ROTATION_DIMS)
DEFAULT_NON_NORMALIZED_ROTATIONS = {"quat", "rot6d", "matrix"}


class VectorStatsAccumulator:
    """Accumulate vector stats with optional exact quantiles."""

    def __init__(self, name: str, keep_values: bool):
        self.name = name
        self.keep_values = keep_values
        self.count = 0
        self.dim: int | None = None
        self.mean: np.ndarray | None = None
        self.m2: np.ndarray | None = None
        self.minimum: np.ndarray | None = None
        self.maximum: np.ndarray | None = None
        self.arrays: list[np.ndarray] = []

    def update(self, values: np.ndarray, source: str):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError(f"{source}: expected {self.name} to be 2D after stacking, got {values.shape}")
        if len(values) == 0:
            return

        if self.dim is None:
            self.dim = values.shape[1]
        elif values.shape[1] != self.dim:
            raise ValueError(
                f"{source}: {self.name} dim {values.shape[1]} does not match previous dim {self.dim}"
            )

        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        batch_m2 = values.var(axis=0) * batch_count
        batch_min = values.min(axis=0)
        batch_max = values.max(axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.minimum = batch_min
            self.maximum = batch_max
        else:
            assert self.mean is not None and self.m2 is not None
            assert self.minimum is not None and self.maximum is not None
            total_count = self.count + batch_count
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * batch_count / total_count
            self.m2 = self.m2 + batch_m2 + delta * delta * self.count * batch_count / total_count
            self.minimum = np.minimum(self.minimum, batch_min)
            self.maximum = np.maximum(self.maximum, batch_max)
            self.count = total_count

        if self.keep_values:
            self.arrays.append(values)

    def result(self, include_quantiles: bool, replace_zero_std: bool) -> dict[str, list[float]]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise ValueError(f"No values were accumulated for {self.name}")
        if self.minimum is None or self.maximum is None:
            raise ValueError(f"No min/max values were accumulated for {self.name}")

        std = np.sqrt(np.maximum(self.m2 / self.count, 0.0))
        if replace_zero_std:
            std[std == 0] = 1

        stats = {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": self.mean.tolist(),
            "std": std.tolist(),
        }

        if include_quantiles:
            if len(self.arrays) == 0:
                raise ValueError(f"Quantiles requested, but no values were kept for {self.name}")
            values = np.concatenate(self.arrays, axis=0)
            quantile_values = np.quantile(values, list(QUANTILES.values()), axis=0)
            for idx, key in enumerate(QUANTILES):
                stats[key] = quantile_values[idx].tolist()

        return stats


def _find_parquet_files(dataset_dir: Path) -> list[Path]:
    if dataset_dir.is_file():
        if dataset_dir.suffix != ".parquet":
            raise ValueError(f"{dataset_dir}: expected a .parquet file")
        return [dataset_dir]

    search_root = dataset_dir / "data" if (dataset_dir / "data").is_dir() else dataset_dir
    pattern = str(search_root / "**" / "*.parquet")
    parquet_files = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
    if len(parquet_files) == 0:
        raise ValueError(f"{dataset_dir}: no parquet files found under {search_root}")
    return parquet_files


def _read_vector_column(df: pd.DataFrame, key: str, source: str) -> np.ndarray:
    if key not in df.columns:
        raise KeyError(f"{source}: missing required column {key!r}")

    values = df[key].to_numpy()
    if len(values) == 0:
        return np.empty((0, 0), dtype=np.float64)
    array = np.stack([np.asarray(value) for value in values], axis=0).astype(np.float64)
    return array.reshape(len(values), -1)


def _read_vector_columns(df: pd.DataFrame, keys: tuple[str, ...], source: str, name: str) -> np.ndarray:
    if len(keys) == 0:
        raise ValueError(f"{source}: {name}_keys must contain at least one key")

    arrays = [_read_vector_column(df, key, source) for key in keys]
    length_by_key = {key: len(array) for key, array in zip(keys, arrays)}
    if len(set(length_by_key.values())) != 1:
        raise ValueError(f"{source}: {name}_keys have mismatched lengths: {length_by_key}")
    return np.concatenate(arrays, axis=1)


def _resolve_point_cloud_dir(dataset_dir: Path, point_cloud_dir: Path | None) -> Path | None:
    if point_cloud_dir is None:
        return None
    if point_cloud_dir.is_absolute():
        return point_cloud_dir
    if dataset_dir.is_file():
        return dataset_dir.parent / point_cloud_dir
    return dataset_dir / point_cloud_dir


def _read_point_cloud_centers(txn, df: pd.DataFrame, source: str) -> np.ndarray:
    missing_columns = [key for key in ("episode_index", "frame_index") if key not in df.columns]
    if missing_columns:
        raise KeyError(f"{source}: missing required point cloud index columns: {missing_columns}")

    centers = []
    for ep_idx, frame_idx in zip(df["episode_index"].to_numpy(), df["frame_index"].to_numpy()):
        point_key = f"{int(ep_idx)}-{int(frame_idx)}"
        point_cloud_bytes = txn.get(point_key.encode("ascii"))
        if point_cloud_bytes is None:
            raise KeyError(f"{source}: point cloud {point_key!r} not found")

        point_cloud = np.asarray(msgpack.unpackb(point_cloud_bytes), dtype=np.float64)
        if point_cloud.ndim != 2 or point_cloud.shape[1] < 3 or len(point_cloud) == 0:
            raise ValueError(f"{source}: point cloud {point_key!r} has invalid shape {point_cloud.shape}")
        centers.append(point_cloud[:, :3].mean(axis=0))

    return np.stack(centers, axis=0)


def _validate_xyz_slice(xyz_slice: tuple[int, int] | None, name: str):
    if xyz_slice is None:
        return

    start, end = xyz_slice
    if start < 0 or end <= start:
        raise ValueError(f"{name}_xyz_slice must be [start, end), got {xyz_slice}")
    if end - start != 3:
        raise ValueError(f"{name}_xyz_slice must contain exactly 3 dims, got {xyz_slice}")


def _validate_point_cloud_args(
    point_cloud_dir: Path | None,
    state_xyz_slice: tuple[int, int] | None,
    action_xyz_slice: tuple[int, int] | None,
):
    _validate_xyz_slice(state_xyz_slice, "state")
    _validate_xyz_slice(action_xyz_slice, "action")

    has_xyz_slice = state_xyz_slice is not None or action_xyz_slice is not None
    if has_xyz_slice and point_cloud_dir is None:
        raise ValueError("--state_xyz_slice/--action_xyz_slice require --point_cloud_dir")
    if point_cloud_dir is not None and not has_xyz_slice:
        raise ValueError("--point_cloud_dir requires --state_xyz_slice or --action_xyz_slice")


def _center_xyz_by_point_cloud(
    values: np.ndarray,
    xyz_slice: tuple[int, int] | None,
    point_centers: np.ndarray,
    source: str,
    name: str,
) -> np.ndarray:
    if xyz_slice is None:
        return values

    if len(point_centers) != len(values):
        raise ValueError(
            f"{source}: point cloud center length {len(point_centers)} does not match {name} length {len(values)}"
        )

    start, end = xyz_slice
    if end > values.shape[1]:
        raise ValueError(f"{source}: {name}_xyz_slice {xyz_slice} exceeds {name} dim {values.shape[1]}")

    values = values.copy()
    values[:, start:end] = values[:, start:end] - point_centers
    return values


def _validate_rotation_slice(
    rotation_slice: tuple[int, int] | None,
    rotation_type: str | None,
    name: str,
):
    if rotation_slice is None:
        return
    if rotation_type is None:
        raise ValueError(f"{name}_rotation_slice requires --rotation_type")

    start, end = rotation_slice
    if start < 0 or end <= start:
        raise ValueError(f"{name}_rotation_slice must be [start, end), got {rotation_slice}")

    expected_dim = ROTATION_DIMS[rotation_type]
    if end - start != expected_dim:
        raise ValueError(
            f"{name}_rotation_slice length {end - start} does not match "
            f"{rotation_type} dim {expected_dim}"
        )


def _validate_rotation_args(
    state_rotation_slice: tuple[int, int] | None,
    action_rotation_slice: tuple[int, int] | None,
    rotation_type: str | None,
    target_rotation_type: str | None,
):
    has_rotation_slice = state_rotation_slice is not None or action_rotation_slice is not None
    if not has_rotation_slice:
        if rotation_type is not None or target_rotation_type is not None:
            raise ValueError(
                "--rotation_type and --target_rotation_type require "
                "--state_rotation_slice or --action_rotation_slice"
            )
        return

    if rotation_type is None or target_rotation_type is None:
        raise ValueError(
            "Rotation conversion requires --rotation_type, --target_rotation_type, "
            "and at least one rotation slice"
        )

    _validate_rotation_slice(state_rotation_slice, rotation_type, "state")
    _validate_rotation_slice(action_rotation_slice, rotation_type, "action")


def _as_rotation_matrix_input(rotation: np.ndarray, rotation_type: str) -> np.ndarray:
    if rotation_type == "matrix":
        return rotation.reshape(len(rotation), 3, 3)
    return rotation


def _flatten_rotation_output(rotation: np.ndarray) -> np.ndarray:
    return np.asarray(rotation, dtype=np.float64).reshape(len(rotation), -1)


def _convert_rotation_slice(
    values: np.ndarray,
    rotation_slice: tuple[int, int] | None,
    rotation_type: str | None,
    target_rotation_type: str | None,
    quat_order_src: str,
    quat_order_dst: str,
    euler_order_src: str,
    euler_order_dst: str,
    source: str,
    name: str,
) -> np.ndarray:
    if rotation_slice is None:
        return values
    assert rotation_type is not None and target_rotation_type is not None

    start, end = rotation_slice
    if end > values.shape[1]:
        raise ValueError(
            f"{source}: {name}_rotation_slice {rotation_slice} exceeds {name} dim {values.shape[1]}"
        )
    if rotation_type == target_rotation_type:
        return values

    rotation = values[:, start:end]
    rotation = _as_rotation_matrix_input(rotation, rotation_type)
    converted = convert_rotation(
        rotation,
        rotation_type,
        target_rotation_type,
        quat_order_src=quat_order_src,
        quat_order_dst=quat_order_dst,
        euler_order_src=euler_order_src,
        euler_order_dst=euler_order_dst,
    )
    converted = _flatten_rotation_output(converted)
    return np.concatenate([values[:, :start], converted, values[:, end:]], axis=1)


def _target_rotation_slice(
    rotation_slice: tuple[int, int] | None,
    target_rotation_type: str | None,
) -> tuple[int, int] | None:
    if rotation_slice is None or target_rotation_type is None:
        return None
    start, _ = rotation_slice
    return start, start + ROTATION_DIMS[target_rotation_type]


def _resolve_normalize_target_rotation(
    normalize_target_rotation: bool | None,
    target_rotation_type: str | None,
) -> bool:
    if normalize_target_rotation is not None:
        return normalize_target_rotation
    if target_rotation_type in DEFAULT_NON_NORMALIZED_ROTATIONS:
        return False
    return True


def _disable_rotation_mean_std(
    stats: dict[str, list[float]],
    rotation_slice: tuple[int, int] | None,
):
    if rotation_slice is None:
        return

    start, end = rotation_slice
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    if end > len(mean):
        raise ValueError(f"rotation_slice {rotation_slice} exceeds stats dim {len(mean)}")

    mean[start:end] = 0
    std[start:end] = 1
    stats["mean"] = mean.tolist()
    stats["std"] = std.tolist()


def _print_stats(name: str, stats: dict[str, list[float]], include_quantiles: bool):
    print(name)
    if include_quantiles:
        keys = ("min", "max", "mean", "std", *QUANTILES.keys())
    else:
        keys = ("min", "max", "mean", "std")
    for key in keys:
        print(f"\t{key}:", np.asarray(stats[key]))


def main(
    dataset_dirs: list[Path],
    output_file: Path,
    state_keys: tuple[str, ...] = DEFAULT_STATE_KEYS,
    action_keys: tuple[str, ...] = DEFAULT_ACTION_KEYS,
    point_cloud_dir: Path | None = None,
    state_xyz_slice: tuple[int, int] | None = None,
    action_xyz_slice: tuple[int, int] | None = None,
    state_rotation_slice: tuple[int, int] | None = None,
    action_rotation_slice: tuple[int, int] | None = None,
    rotation_type: str | None = None,
    target_rotation_type: str | None = None,
    normalize_target_rotation: bool | None = None,
    quat_order_src: str = "xyzw",
    quat_order_dst: str = "xyzw",
    euler_order_src: str = "xyz",
    euler_order_dst: str = "xyz",
    include_quantiles: bool = True,
    replace_zero_std: bool = False,
):
    """Compute global robot state/action stats over raw parquet columns.

    Args:
        dataset_dirs: LeRobot dataset roots, data directories, or parquet files.
        output_file: Destination JSON path.
        state_keys: State columns to read and concatenate.
        action_keys: Action columns to read and concatenate.
        point_cloud_dir: Optional point cloud LMDB path. If relative, resolves
            under each dataset_dir, for example points_frontview.
        state_xyz_slice: [start, end) xyz dims in concatenated state to center
            by subtracting each frame's point cloud mean.
        action_xyz_slice: [start, end) xyz dims in concatenated action to center
            by subtracting each frame's point cloud mean. Leave unset for delta action.
        state_rotation_slice: [start, end) rotation dims in concatenated state.
        action_rotation_slice: [start, end) rotation dims in concatenated action.
        rotation_type: Source rotation type: quat, euler, rot6d, axisangle, matrix.
        target_rotation_type: Target rotation type: quat, euler, rot6d, axisangle, matrix.
        normalize_target_rotation: Whether mean/std should normalize target rotation dims.
            Defaults to false for quat, rot6d, matrix and true for euler, axisangle.
        quat_order_src: Source quaternion order.
        quat_order_dst: Target quaternion order.
        euler_order_src: Source Euler order.
        euler_order_dst: Target Euler order.
        include_quantiles: Whether to compute exact q01/q10/q50/q90/q99 values.
            This keeps the selected vectors in memory.
        replace_zero_std: Replace zero std entries with 1 in the output. Useful if the
            file will also be used as a mean/std normalization file.
    """

    _validate_point_cloud_args(
        point_cloud_dir=point_cloud_dir,
        state_xyz_slice=state_xyz_slice,
        action_xyz_slice=action_xyz_slice,
    )
    _validate_rotation_args(
        state_rotation_slice=state_rotation_slice,
        action_rotation_slice=action_rotation_slice,
        rotation_type=rotation_type,
        target_rotation_type=target_rotation_type,
    )
    normalize_rotation = _resolve_normalize_target_rotation(
        normalize_target_rotation,
        target_rotation_type,
    )
    state_target_rotation_slice = _target_rotation_slice(state_rotation_slice, target_rotation_type)
    action_target_rotation_slice = _target_rotation_slice(action_rotation_slice, target_rotation_type)

    state_acc = VectorStatsAccumulator("state", keep_values=include_quantiles)
    action_acc = VectorStatsAccumulator("action", keep_values=include_quantiles)

    total_frames = 0
    for dataset_dir in dataset_dirs:
        parquet_files = _find_parquet_files(dataset_dir)
        dataset_frames = 0
        point_cloud_env = None
        point_cloud_txn = None
        resolved_point_cloud_dir = _resolve_point_cloud_dir(dataset_dir, point_cloud_dir)
        if resolved_point_cloud_dir is not None:
            if not resolved_point_cloud_dir.exists():
                raise FileNotFoundError(f"Point cloud LMDB not found: {resolved_point_cloud_dir}")
            point_cloud_env = lmdb.open(str(resolved_point_cloud_dir), readonly=True, lock=False)
            point_cloud_txn = point_cloud_env.begin(buffers=True)

        try:
            progress = tqdm(parquet_files, desc=f"Reading {dataset_dir.name}", unit="file")
            for parquet_file in progress:
                parquet_source = str(parquet_file)
                columns = sorted(set(state_keys + action_keys))
                if point_cloud_txn is not None:
                    columns = sorted(set(columns + ["episode_index", "frame_index"]))
                df = pd.read_parquet(parquet_file, columns=columns)
                states = _read_vector_columns(df, state_keys, parquet_source, "state")
                actions = _read_vector_columns(df, action_keys, parquet_source, "action")
                if len(states) != len(actions):
                    raise ValueError(f"{parquet_file}: state length {len(states)} != action length {len(actions)}")

                if point_cloud_txn is not None:
                    point_centers = _read_point_cloud_centers(point_cloud_txn, df, parquet_source)
                    states = _center_xyz_by_point_cloud(
                        states,
                        state_xyz_slice,
                        point_centers,
                        parquet_source,
                        "state",
                    )
                    actions = _center_xyz_by_point_cloud(
                        actions,
                        action_xyz_slice,
                        point_centers,
                        parquet_source,
                        "action",
                    )

                states = _convert_rotation_slice(
                    states,
                    state_rotation_slice,
                    rotation_type,
                    target_rotation_type,
                    quat_order_src,
                    quat_order_dst,
                    euler_order_src,
                    euler_order_dst,
                    parquet_source,
                    "state",
                )
                actions = _convert_rotation_slice(
                    actions,
                    action_rotation_slice,
                    rotation_type,
                    target_rotation_type,
                    quat_order_src,
                    quat_order_dst,
                    euler_order_src,
                    euler_order_dst,
                    parquet_source,
                    "action",
                )

                state_acc.update(states, parquet_source)
                action_acc.update(actions, parquet_source)
                dataset_frames += len(states)
                total_frames += len(states)
                progress.set_postfix(frames=dataset_frames)
        finally:
            if point_cloud_txn is not None:
                point_cloud_txn.abort()
            if point_cloud_env is not None:
                point_cloud_env.close()

        print(f"{dataset_dir}: {dataset_frames} frames from {len(parquet_files)} parquet files")

    state_stats = state_acc.result(include_quantiles=include_quantiles, replace_zero_std=replace_zero_std)
    action_stats = action_acc.result(include_quantiles=include_quantiles, replace_zero_std=replace_zero_std)
    if not normalize_rotation:
        _disable_rotation_mean_std(state_stats, state_target_rotation_slice)
        _disable_rotation_mean_std(action_stats, action_target_rotation_slice)

    _print_stats("robot state", state_stats, include_quantiles)
    _print_stats("robot action", action_stats, include_quantiles)

    output = {
        **{f"state_{key}": value for key, value in state_stats.items()},
        **{f"action_{key}": value for key, value in action_stats.items()},
        "num_data": total_frames,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as outf:
        json.dump(output, outf, indent=2)
    print(f"Saved the data to {output_file}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute robot state/action statistics from LeRobot parquet files."
    )
    parser.add_argument(
        "--dataset_dirs",
        "--dataset-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="LeRobot dataset roots, data directories, or parquet files.",
    )
    parser.add_argument(
        "--output_file",
        "--output-file",
        type=Path,
        required=True,
        help="Destination JSON path.",
    )
    parser.add_argument(
        "--state_keys",
        "--state-keys",
        nargs="+",
        default=DEFAULT_STATE_KEYS,
        help="State columns to read and concatenate.",
    )
    parser.add_argument(
        "--action_keys",
        "--action-keys",
        nargs="+",
        default=DEFAULT_ACTION_KEYS,
        help="Action columns to read and concatenate.",
    )
    parser.add_argument(
        "--point_cloud_dir",
        "--point-cloud-dir",
        type=Path,
        default=None,
        help=(
            "Optional point cloud LMDB path. If relative, resolves under each dataset dir, "
            "for example points_frontview."
        ),
    )
    parser.add_argument(
        "--state_xyz_slice",
        "--state-xyz-slice",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="State xyz dims [start, end) to subtract point cloud mean from.",
    )
    parser.add_argument(
        "--action_xyz_slice",
        "--action-xyz-slice",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Action xyz dims [start, end) to subtract point cloud mean from. Leave unset for delta action.",
    )
    parser.add_argument(
        "--state_rotation_slice",
        "--state-rotation-slice",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Rotation dims [start, end) in concatenated state.",
    )
    parser.add_argument(
        "--action_rotation_slice",
        "--action-rotation-slice",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Rotation dims [start, end) in concatenated action.",
    )
    parser.add_argument(
        "--rotation_type",
        "--rotation-type",
        choices=ROTATION_TYPES,
        default=None,
        help="Source rotation type.",
    )
    parser.add_argument(
        "--target_rotation_type",
        "--target-rotation-type",
        choices=ROTATION_TYPES,
        default=None,
        help="Target rotation type.",
    )
    parser.add_argument(
        "--normalize_target_rotation",
        "--normalize-target-rotation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether mean/std should normalize target rotation dims.",
    )
    parser.add_argument(
        "--quat_order_src",
        "--quat-order-src",
        choices=("wxyz", "xyzw"),
        default="xyzw",
        help="Source quaternion order.",
    )
    parser.add_argument(
        "--quat_order_dst",
        "--quat-order-dst",
        choices=("wxyz", "xyzw"),
        default="xyzw",
        help="Target quaternion order.",
    )
    parser.add_argument(
        "--euler_order_src",
        "--euler-order-src",
        choices=("xyz", "zyx"),
        default="xyz",
        help="Source Euler order.",
    )
    parser.add_argument(
        "--euler_order_dst",
        "--euler-order-dst",
        choices=("xyz", "zyx"),
        default="xyz",
        help="Target Euler order.",
    )
    parser.add_argument(
        "--include_quantiles",
        "--include-quantiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute exact q01/q10/q50/q90/q99 values.",
    )
    parser.add_argument(
        "--replace_zero_std",
        "--replace-zero-std",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace zero std entries with 1 in the output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        dataset_dirs=args.dataset_dirs,
        output_file=args.output_file,
        state_keys=tuple(args.state_keys),
        action_keys=tuple(args.action_keys),
        point_cloud_dir=args.point_cloud_dir,
        state_xyz_slice=None if args.state_xyz_slice is None else tuple(args.state_xyz_slice),
        action_xyz_slice=None if args.action_xyz_slice is None else tuple(args.action_xyz_slice),
        state_rotation_slice=None if args.state_rotation_slice is None else tuple(args.state_rotation_slice),
        action_rotation_slice=None if args.action_rotation_slice is None else tuple(args.action_rotation_slice),
        rotation_type=args.rotation_type,
        target_rotation_type=args.target_rotation_type,
        normalize_target_rotation=args.normalize_target_rotation,
        quat_order_src=args.quat_order_src,
        quat_order_dst=args.quat_order_dst,
        euler_order_src=args.euler_order_src,
        euler_order_dst=args.euler_order_dst,
        include_quantiles=args.include_quantiles,
        replace_zero_std=args.replace_zero_std,
    )
