import bisect
import json
import multiprocessing
import os
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

import datasets
import torch
from lerobot.constants import ACTION, HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import (
    LeRobotDatasetMetadata,
    MultiLeRobotDataset as BaseMultiLeRobotDataset,
)
from lerobot.datasets.utils import serialize_dict
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from pointact.data.robot.registry import get_robot_dataset_class
from pointact.data.schema import LerobotConfig


class MultiLeRobotDataset(BaseMultiLeRobotDataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`.
    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        data_configs: list[LerobotConfig],
        image_transforms: Callable | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        chunk_size: int = 32,
        max_state_dim: int = 64,
        **kwargs: Any,
    ):
        self.data_configs = data_configs
        self.chunk_size = chunk_size
        self.max_state_dim = max_state_dim
        self.disabled_features = set()

        datasets = self.load_datasets(
            image_transforms=image_transforms,
            download_videos=download_videos,
            video_backend=video_backend,
            chunk_size=chunk_size,
        )
        self._datasets = datasets #[ds for ds in datasets if ds is not None]
        self.repo_ids = [ds.repo_id for ds in self._datasets]
        print(f"successfully load dataset {len(self.repo_ids)}/{len(data_configs)}:\n{self.repo_ids} ")

        self.image_transforms = image_transforms
        self.build_repo_index()
        self.build_feature_key_maps()

    def load_datasets(
        self,
        image_transforms: Callable | None,
        download_videos: bool,
        video_backend: str | None,
        chunk_size: int,
    ):
        num_processes = int(os.environ.get("DATASET_NUM_PROCESSES", 8))
        print(f"* load {len(self.data_configs)} lerobot datasets with {num_processes} processes ...")
        fn = partial(
            load_single_lerobot_dataset,
            data_configs=self.data_configs,
            image_transforms=image_transforms,
            download_videos=download_videos,
            video_backend=video_backend,
            chunk_size=chunk_size,
        )

        indices = range(len(self.data_configs))
        if num_processes <= 1:
            return list(
                tqdm(
                    (fn(idx) for idx in indices),
                    total=len(self.data_configs),
                    desc="Loading lerobot datasets",
                )
            )

        with multiprocessing.Pool(processes=num_processes) as pool:
            return list(
                tqdm(
                    pool.imap(fn, indices),
                    total=len(self.data_configs),
                    desc="Loading lerobot datasets",
                )
            )

    def build_repo_index(self):
        self._repo_ids_index = {repo_id: i for i, repo_id in enumerate(self.repo_ids)}
        self.cumulative_sizes = ConcatDataset.cumsum(self._datasets)

    def build_feature_key_maps(self):
        self._select_video_keys = {
            ds.repo_id.replace("/", "."): ds.select_video_keys for ds in self._datasets
        }
        self._select_video_keys_for_vlm = {
            ds.repo_id.replace("/", "."): getattr(ds, "select_video_keys_for_vlm", ds.select_video_keys) for ds in self._datasets
        }
        self._select_state_keys = {
            ds.repo_id.replace("/", "."): ds.select_state_keys for ds in self._datasets
        }
        self._select_action_keys = {
            ds.repo_id.replace("/", "."): ds.select_action_keys for ds in self._datasets
        }

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        max_retries = 10
        last_error = None
        for attempt in range(max_retries + 1):
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            try:
                item = self._datasets[dataset_idx][sample_idx]
                return self.post_process(item)
            except Exception as e:
                last_error = e
                print(
                    f"cannot load dataset_idx {dataset_idx} sample_idx {sample_idx} "
                    f"(attempt {attempt + 1}/{max_retries + 1})",
                    type(e).__name__,
                    e,
                )
                if attempt == max_retries:
                    break
                idx = np.random.randint(len(self))

        raise RuntimeError(f"Failed to load a sample after {max_retries + 1} attempts.") from last_error
    
    def getitem_by_id(self, repo_id: str, idx: int, chunk_size: int = None) -> dict[str, torch.Tensor]:
        """get an item by repo_id and index."""
        dataset_idx = self._repo_ids_index.get(repo_id)
        if dataset_idx is None:
            raise ValueError(f"invalid dataset: {repo_id}. available dataset: {self.repo_ids}")
        lerobot_dataset = self._datasets[dataset_idx]

        delta_indices = None
        if chunk_size is not None:
            delta_indices = {k: list(range(0, chunk_size)) for k in lerobot_dataset.select_action_keys}

        item = lerobot_dataset.__getitem__(idx, delta_indices=delta_indices)
        return self.post_process(item)

    def post_process(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """unify video keys across datasets."""
        return item

    @property
    def _features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            repo_id_suffix = dataset.repo_id.replace("/", ".")
            features[repo_id_suffix] = dataset._features
        return features

    @property
    def _stats(self) -> datasets.Features:
        stats = {}
        for dataset in self._datasets:
            repo_id_suffix = dataset.repo_id.replace("/", ".")
            stats[repo_id_suffix] = dataset._stats
        return stats

    @property
    def configuration(self) -> dict:
        robot_config = {
            "features": self._features,
            "stats": serialize_dict(self._stats),
            "select_video_keys": self._select_video_keys,
            "select_video_keys_for_vlm": self._select_video_keys_for_vlm,
            "select_state_keys": self._select_state_keys,
            "select_action_keys": self._select_action_keys,
            "max_state_dim": self.max_state_dim,
            "is_delta_action": {},
            "is_action_eef": {},
            "state_action_norm": {},
            "points_workspace": {},
            "max_npoints": {},
        }
        for data_config in self.data_configs:
            repo_id = data_config.repo_id.replace("/", ".")
            robot_config["is_delta_action"][repo_id] = data_config.is_delta_action
            robot_config["is_action_eef"][repo_id] = data_config.is_action_eef
            robot_config["points_workspace"][repo_id] = data_config.points_workspace
            robot_config["max_npoints"][repo_id] = data_config.max_npoints
            if data_config.state_action_norm_file is not None:
                with open(data_config.state_action_norm_file) as f:
                    robot_config["state_action_norm"][repo_id] = json.load(f)
            else:
                robot_config["state_action_norm"][repo_id] = None
        
        return robot_config


def load_single_lerobot_dataset(
    idx,
    data_configs: list[LerobotConfig],
    image_transforms: Callable | None = None,
    download_videos: bool = True,
    video_backend: str | None = None,
    chunk_size: int = 32,
):
    """load a single lerobot dataset"""
    # try:
    data_config = data_configs[idx]
    data_path = Path(data_config.root or HF_LEROBOT_HOME) / data_config.repo_id
    meta = LeRobotDatasetMetadata(data_config.repo_id, data_path)
    select_action_keys = data_config.select_action_keys or [
        k for k in meta.features if k.startswith(ACTION)
    ]
    delta_timestamps = {k: [i / meta.fps for i in range(0, chunk_size)] for k in select_action_keys}

    dataset_args = {}
    for arg_key, arg_value in data_config.__dict__.items():
        if arg_key != 'class_name':
            dataset_args[arg_key] = arg_value
    dataset_args.update({
        "root": data_path,
        "image_transforms": image_transforms,
        "delta_timestamps": delta_timestamps,
        "download_videos": download_videos,
        "video_backend": video_backend,
    })

    dataset_cls = get_robot_dataset_class(data_config.class_name)
    dataset = dataset_cls(**dataset_args)
    # except Exception as e:
    #     print(f"[warn] read dataset {data_config.repo_id} failed, skipped!")
    #     print(e)
    #     return None
    return dataset
