import os
import random
from collections.abc import Callable
from pathlib import Path

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
from lerobot.constants import ACTION, OBS_STATE
from pointact.constants import OBS_POINTS

from pointact.data.robot.base import LeRobotDatasetMixin
from pointact.data.robot.registry import register_robot_dataset
from pointact.data.transforms.pointcloud import (
    augment_point_cloud_color,
    random_rotate_point_around_z,
    random_rotate_quat_around_z,
    random_rotate_delta_quat_around_z,
)

msgpack_numpy.patch()


@register_robot_dataset("LeRobotPointCloudDataset")
class LeRobotPointCloudDataset(LeRobotDatasetMixin):
    """LeRobot dataset variant backed by precomputed point clouds in LMDB.

    Point cloud LMDB entries are xyzrgb with RGB in [0, 1]. 
    State/action tensors are treated as world-frame values before optional rotation augmentation and point-cloud centering.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        # custom features
        select_video_keys: list[str] | None = None,
        select_state_keys: list[str] | None = None,
        select_action_keys: list[str] | None = None,
        train_subtask: str | None = None, # cumulate, mixture:0.5
        is_delta_action: bool = False,
        is_action_eef: bool = True,
        weight: float | None = None,
        image_size: int | None = None,
        converted_rot_type: str | None = None, # quat (default),euler, rot6d
        state_action_norm_file: str | None = None,
        # point related
        video_key_ids_for_vlm: list[int] | None = None,
        points_workspace: dict | None = None,
        max_npoints: int = 4096,
        augment_pc_rot: int = 0,
        point_cloud_dirname: str | None = None,
        **kwargs,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            select_action_keys=select_action_keys,
            select_state_keys=select_state_keys,
            select_video_keys=select_video_keys,
            video_key_ids_for_vlm=video_key_ids_for_vlm,
            train_subtask=train_subtask,
            is_delta_action=is_delta_action,
            is_action_eef=is_action_eef,
            weight=weight,
            image_size=image_size,
            converted_rot_type=converted_rot_type,
            state_action_norm_file=state_action_norm_file,
        )

        self.points_workspace = points_workspace
        self.max_npoints = max_npoints
        self.augment_pc_rot = augment_pc_rot

        assert point_cloud_dirname is not None
        self.point_cloud_dir = os.path.join(self.root, point_cloud_dirname)
        self._point_cloud_lmdb_env = None
        self._point_cloud_lmdb_txn = None
        self._point_cloud_lmdb_pid = None    

    def __del__(self):
        if getattr(self, "_point_cloud_lmdb_txn", None) is not None:
            self._point_cloud_lmdb_txn.abort()
            self._point_cloud_lmdb_txn = None
        if getattr(self, "_point_cloud_lmdb_env", None) is not None:
            self._point_cloud_lmdb_env.close()
            self._point_cloud_lmdb_env = None
        self._point_cloud_lmdb_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_point_cloud_lmdb_env"] = None
        state["_point_cloud_lmdb_txn"] = None
        state["_point_cloud_lmdb_pid"] = None
        return state

    def set_feature_keys(
        self, video_keys=None, state_keys=None, action_keys=None,
        video_key_ids_for_vlm=None, **kwargs
    ):
        # the point cloud can be constructed using all video keys, while the VLM only utilizes a subset of the video keys
        self.select_video_keys = self.meta.video_keys if video_keys is None else video_keys
        if video_key_ids_for_vlm is None:
            self.select_video_keys_for_vlm = self.select_video_keys
        else:
            self.select_video_keys_for_vlm = [self.select_video_keys[i] for i in video_key_ids_for_vlm]

        self.select_state_keys = (
            [key for key in self.meta.features if key.startswith(OBS_STATE)]
            if state_keys is None
            else state_keys
        )
        self.select_action_keys = (
            [key for key in self.meta.features if key.startswith(ACTION)]
            if action_keys is None
            else action_keys
        )

        self.select_feature_keys = self.select_video_keys_for_vlm + self.select_state_keys + self.select_action_keys
        self.select_action_is_pad_keys = [f"{key}_is_pad" for key in self.select_action_keys]

    def __getitem__(self, idx, delta_indices: dict = None) -> dict:
        delta_indices = delta_indices or self.delta_indices

        if self.weight is not None:
            idx = random.randint(0, self.num_frames - 1)

        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        frame_idx = item["frame_index"].item()

        item, query_indices = self.query_action_chunk(item, idx, ep_idx, delta_indices)
        item = self.add_video_frames(item, ep_idx, query_indices)
        self.apply_image_transforms(item, self.select_video_keys_for_vlm)

        point_cloud = self.load_point_cloud(ep_idx, frame_idx)
        point_cloud = self.filter_point_cloud_by_workspace(point_cloud)
        point_cloud = self.augment_point_cloud(point_cloud, item)
        point_cloud = self.center_point_cloud(point_cloud, item)
        item[OBS_POINTS] = torch.from_numpy(point_cloud)

        self.convert_eef_rotation(item)
        self.normalize_state_action(item)
        self.select_task_text(item, ep_idx, idx)

        return self.post_process(item)

    def load_point_cloud(self, ep_idx: int, frame_idx: int):
        txn = self.get_point_cloud_lmdb_txn()
        point_key = f"{ep_idx}-{frame_idx}"
        point_cloud = txn.get(point_key.encode("ascii"))
        if point_cloud is None:
            raise KeyError(f"Point cloud '{point_key}' not found in {self.point_cloud_dir}")
        return msgpack.unpackb(point_cloud).copy().astype(np.float32)

    def filter_point_cloud_by_workspace(self, point_cloud: np.ndarray):
        if self.points_workspace is None:
            return point_cloud

        workspace = self.points_workspace
        point_mask = (
            (point_cloud[:, 0] > workspace["X_BBOX"][0])
            & (point_cloud[:, 0] < workspace["X_BBOX"][1])
            & (point_cloud[:, 1] > workspace["Y_BBOX"][0])
            & (point_cloud[:, 1] < workspace["Y_BBOX"][1])
            & (point_cloud[:, 2] > workspace["Z_BBOX"][0])
            & (point_cloud[:, 2] < workspace["Z_BBOX"][1])
        )
        return point_cloud[point_mask]

    def get_point_cloud_lmdb_txn(self):
        current_pid = os.getpid()
        if self._point_cloud_lmdb_pid != current_pid:
            self._point_cloud_lmdb_env = None
            self._point_cloud_lmdb_txn = None
            self._point_cloud_lmdb_pid = current_pid

        if self._point_cloud_lmdb_env is None:
            self._point_cloud_lmdb_env = lmdb.open(
                self.point_cloud_dir,
                readonly=True,
                lock=False,
                readahead=False,
                max_spare_txns=1,
            )
            self._point_cloud_lmdb_txn = self._point_cloud_lmdb_env.begin(buffers=True)

        return self._point_cloud_lmdb_txn

    def augment_point_cloud(self, point_cloud: np.ndarray, item: dict):
        max_npoints = min(int(len(point_cloud) * np.random.uniform(0.8, 1.0)), self.max_npoints)
        if len(point_cloud) > max_npoints:
            ridxs = np.random.choice(len(point_cloud), max_npoints, replace=False)
            point_cloud = point_cloud[ridxs]

        point_cloud_color = augment_point_cloud_color(
            point_cloud[:, 3:6],
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            jitter_std=0.02,
        )
        point_cloud[:, 3:6] = point_cloud_color * 2 - 1

        if self.augment_pc_rot != 0:
            angle = np.random.uniform(-1, 1) * np.deg2rad(self.augment_pc_rot)
            point_cloud[:, :3] = random_rotate_point_around_z(point_cloud[:, :3], angle=angle)
            if self.is_action_eef:
                item[OBS_STATE][:3] = random_rotate_point_around_z(
                    item[OBS_STATE][:3].unsqueeze(0), angle=angle
                )[0]
                item[OBS_STATE][3:7] = random_rotate_quat_around_z(item[OBS_STATE][3:7], angle)
                item[ACTION][:, :3] = random_rotate_point_around_z(item[ACTION][:, :3], angle=angle)
                if self.is_delta_action:
                    item[ACTION][:, 3:7] = random_rotate_delta_quat_around_z(item[ACTION][:, 3:7], angle)
                else:
                    item[ACTION][:, 3:7] = random_rotate_quat_around_z(item[ACTION][:, 3:7], angle)

        return point_cloud

    def center_point_cloud(self, point_cloud: np.ndarray, item: dict):
        point_center = point_cloud[:, :3].mean(0)
        point_cloud[:, :3] = point_cloud[:, :3] - point_center
        point_center = torch.from_numpy(point_center)
        if self.is_action_eef:
            item[OBS_STATE][:3] = item[OBS_STATE][:3] - point_center
            if not self.is_delta_action:
                item[ACTION][:, :3] = item[ACTION][:, :3] - point_center[None, :]
        item[f"{OBS_POINTS}.center"] = point_center
        return point_cloud

    def post_process(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ordered_keys = self.select_feature_keys + [OBS_POINTS, "task", f"{OBS_POINTS}.center"] + self.select_action_is_pad_keys
        item = {key: item[key] for key in ordered_keys if key in item}
        return item
