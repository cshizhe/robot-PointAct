import bisect
import json
import random
from collections.abc import Callable
from pathlib import Path

import datasets
import numpy as np
import torch
from torchvision.transforms.v2 import functional as tvF
from datasets import load_dataset

from lerobot.constants import ACTION, OBS_STATE
from lerobot.datasets.utils import hf_transform_to_torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset as BaseLeRobotDataset

from pointact.utils.rotation import convert_rotation


# Suppose rotation is quaternion (4D)
EEF_ROT_START = 3
EEF_ROT_END = 7
EEF_ROT_METADATA = {
    "euler": {"shape_delta": -1, "motor_names": ["x", "y", "z"]},   # 3D
    "rot6d": {"shape_delta": 2, "motor_names": ["rot6d"] * 6},      # 6D
}


class LeRobotDatasetMixin(BaseLeRobotDataset):
    """Shared utilities for LeRobot dataset variants.

    Add the following features:
        - Subtask training modes.
        - Selection of specific video, state, and action keys.
        - Dataset weighting for sampling.
        - State and action normalization.
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
        )

        self.image_size = image_size
        self.converted_rot_type = converted_rot_type or "quat"
        if self.converted_rot_type not in ["quat", *EEF_ROT_METADATA]:
            raise ValueError(
                f"Unsupported converted_rot_type={converted_rot_type!r}. "
                f"Expected one of {sorted(EEF_ROT_METADATA)}, quat or None."
            )

        self.is_action_eef = is_action_eef
        self.is_delta_action = is_delta_action

        self.state_action_norm = self.load_state_action_norm(state_action_norm_file)

        if self.is_action_eef:
            self.configure_eef_feature_metadata()

        # set weight for the dataset
        self.set_weight(weight)

        # calculate substake indices
        self.set_train_subtask(train_subtask)

        # set episode from index
        self.get_episode_from_index(episodes)

        # remove unused features for efficiency
        self.set_feature_keys(select_video_keys, select_state_keys, select_action_keys, **kwargs)

    def set_feature_keys(self, video_keys=None, state_keys=None, action_keys=None, **kwargs):
        raise NotImplementedError("set_feature_keys must be implemented by subclasses to specify selected features.")

    def load_state_action_norm(self, state_action_norm_file: str | None):
        if state_action_norm_file is None:
            return None
        with open(state_action_norm_file) as f:
            state_action_norm = json.load(f)
        return {
            key: torch.from_numpy(np.array(value).astype(np.float32))
            for key, value in state_action_norm.items()
        }

    def configure_eef_feature_metadata(self):
        if self.converted_rot_type == "quat":
            return

        rot_metadata = EEF_ROT_METADATA[self.converted_rot_type]
        for feature_key in (OBS_STATE, ACTION):
            self.update_eef_rotation_feature_metadata(
                feature_key,
                shape_delta=rot_metadata["shape_delta"],
                rotation_motor_names=rot_metadata["motor_names"],
            )

    def update_eef_rotation_feature_metadata(
        self,
        feature_key: str,
        shape_delta: int,
        rotation_motor_names: list[str],
    ):
        feature = self.meta.features[feature_key]
        feature["shape"] = (feature["shape"][0] + shape_delta,)

        motor_names = feature.get("names", {}).get("motors")
        if motor_names is None:
            return

        feature["names"]["motors"] = (
            motor_names[:EEF_ROT_START]
            + rotation_motor_names
            + motor_names[EEF_ROT_END:]
        )

    def set_train_subtask(self, train_subtask: str | None = None):
        self.train_subtask = train_subtask
        self.task_sizes = {}
        if not train_subtask:
            return

        # TODO: copied from EO1 codes, not used for now, need to verify and refactor if we want to use it
        try:
            for _, ep in self.meta.episodes.items():
                self.task_sizes[ep["episode_index"]] = [
                    item["end_frame"] for item in ep["action_config"]
                ]
        except Exception as e:
            print(f"[warn] {self.repo_id} failed to calculate episode subtask cumulate: {e}")
            self.train_subtask = None

        print(f"* set train_subtask {self.train_subtask} for {self.repo_id}")

    def set_weight(self, weight: float | None):
        self.weight = weight
        if weight is not None:
            self._num_frames_weight = int(self.num_frames * weight)
            print(
                f"* set weight {weight} for {self.repo_id}, "
                f"num_frames: {self.num_frames} -> {self._num_frames_weight}"
            )
        else:
            self._num_frames_weight = self.num_frames

    def get_episode_from_index(self, episodes: list[int] | None = None):
        if episodes is None:
            self.episode_from_index = None
        else:
            self.episode_from_index = {ep_idx: i for i, ep_idx in enumerate(episodes)}

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train", keep_in_memory=False)
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train", keep_in_memory=False)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        # TODO: modify if we need to use the video keys for point cloud construction in 3D dataset
        select_video_keys = getattr(self, "select_video_keys_for_vlm", self.select_video_keys)
        for key in select_video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset[q_idx][key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def __len__(self):
        return self._num_frames_weight

    def _get_query_indices(
        self, idx: int, ep_idx: int, delta_indices: dict = None
    ) -> tuple[dict[str, list[int | bool]]]:
        if self.episode_from_index is not None:
            ep_idx = self.episode_from_index[ep_idx]
        # episode_data_index is constructed according to existing episodes, while ep_idx is the original episode index
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in delta_indices.items()
        }
        return query_indices, padding

    @property
    def _stats(self) -> datasets.Features:
        return {k: self.meta.stats[k] for k in (self.select_state_keys + self.select_action_keys)}

    @property
    def _features(self) -> dict[str, dict]:
        return {k: self.meta.features[k] for k in self.select_feature_keys}

    def query_action_chunk(self, item: dict, idx: int, ep_idx: int, delta_indices: dict = None):
        query_indices = None
        if delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx, delta_indices)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
        return item, query_indices

    def add_video_frames(self, item: dict, ep_idx: int, query_indices: dict | None):
        if len(self.meta.video_keys) == 0:
            return item

        current_ts = item["timestamp"].item()
        query_timestamps = self._get_query_timestamps(current_ts, query_indices)
        video_frames = self._query_videos(query_timestamps, ep_idx)
        item = {**video_frames, **item}

        if self.image_size is not None:
            for cam in self.select_video_keys:
                # torch.Tensor: shape=(C, H, W), value_range=[0, 1]
                item[cam] = tvF.resize(item[cam], size=(self.image_size, self.image_size))
        return item

    def apply_image_transforms(self, item: dict, select_video_keys=None):
        # Return the crop bounding boxes for each image if the image_transforms has cropping for later use (e.g., cropping depth or point clouds accordingly)
        if self.image_transforms is None:
            return []

        if select_video_keys is None:
            select_video_keys = self.select_video_keys

        crop_bboxes = []
        for cam in select_video_keys:
            item[cam] = self.image_transforms(item[cam])
            if getattr(self.image_transforms, "crop_idx", None) is not None:
                crop_bboxes.append(self.image_transforms.crop_bbox)
        return crop_bboxes

    def select_task_text(self, item: dict, ep_idx: int, idx: int | None = None):
        task_idx = item["task_index"].item()
        task_text = self.meta.tasks[task_idx]

        # TODO: copied from EO1 codes, not used for now, need to verify and refactor if we want to use it
        if self.train_subtask and len(self.task_sizes.get(ep_idx, [])) > 0:
            try:
                if self.train_subtask == "cumulate":
                    task_text = " ".join(
                        [item["action_text"] for item in self.meta.episodes[ep_idx]["action_config"]]
                    )
                else:
                    sub_idx = bisect.bisect_right(self.task_sizes[ep_idx], item["frame_index"])
                    sub_idx = min(sub_idx, len(self.task_sizes[ep_idx]) - 1)
                    task_text = self.meta.episodes[ep_idx]["action_config"][sub_idx]["action_text"]

                    if self.train_subtask.startswith("mix"):
                        global_text = self.meta.tasks[task_idx]
                        w = float(self.train_subtask.split(":")[-1])
                        task_text = random.choices([task_text, global_text], weights=[w, 1 - w])[0]
            except Exception as e:
                sample_text = "" if idx is None else f" {idx} / {len(self.hf_dataset)}"
                print(f"[warn] {self.repo_id} failed to get subtask{sample_text}: {e}")
                task_text = self.meta.tasks[task_idx]

        # used a special token <br> to separate the instructions
        task_text = [x.strip() for x in task_text.split("<br>")]
        task_text = [x for x in task_text if len(x) > 0]
        item["task"] = random.choice(task_text)

    def convert_eef_rotation(self, item: dict):
        if not self.is_action_eef or self.converted_rot_type == "quat":
            return

        item[OBS_STATE] = self.convert_eef_rotation_tensor(item[OBS_STATE])
        item[ACTION] = self.convert_eef_rotation_tensor(item[ACTION])

    def convert_eef_rotation_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        converted_rotation = self.convert_quat_rotation(tensor[..., EEF_ROT_START:EEF_ROT_END])
        return torch.cat(
            [
                tensor[..., :EEF_ROT_START],
                converted_rotation,
                tensor[..., EEF_ROT_END:],
            ],
            dim=-1,
        )

    def convert_quat_rotation(self, quat: torch.Tensor) -> torch.Tensor:
        convert_kwargs = {"quat_order_src": "xyzw"}
        if self.converted_rot_type == "euler":
            convert_kwargs["euler_order_dst"] = "xyz"

        converted = convert_rotation(
            quat.numpy(),
            "quat",
            self.converted_rot_type,
            **convert_kwargs,
        )
        return torch.as_tensor(converted, dtype=quat.dtype, device=quat.device)

    def normalize_state_action(self, item: dict):
        if self.state_action_norm is None:
            return

        self.normalize_feature(item, ACTION, "action")
        self.normalize_feature(item, OBS_STATE, "state")

    def normalize_feature(self, item: dict, feature_key: str, norm_prefix: str):
        mean_key = f"{norm_prefix}_mean"
        std_key = f"{norm_prefix}_std"
        if feature_key not in item or mean_key not in self.state_action_norm or std_key not in self.state_action_norm:
            return

        feature = item[feature_key]
        mean = self.state_action_norm[mean_key]
        std = self.state_action_norm[std_key]
        while mean.dim() < feature.dim():
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        item[feature_key] = (feature - mean) / std
