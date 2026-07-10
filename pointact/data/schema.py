# Copyright 2025 EO-Robotics Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Schema for the dataset configuration."""

from dataclasses import dataclass, field

import yaml


@dataclass
class MMDatasetConfig:
    json_path: str
    sampling_strategy: str = "all"
    vision_base_path: str | None = None


@dataclass
class LerobotConfig:
    repo_id: str
    root: str | None = None
    episodes: list[int] | None = None
    is_delta_action: bool = False
    is_action_eef: bool = True

    state_action_norm_file: str | None = None

    converted_rot_type: str | None = "euler"  # None, euler, rot6d

    train_subtask: str | bool | None = False  # Optional[true, false, mix:0.5, cumulate]
    select_video_keys: list[str] = None
    video_key_ids_for_vlm: list[int] | None = None
    select_depth_keys: list[str] = None
    select_camera_keys: list[str] = None
    select_action_keys: list[str] = None
    select_state_keys: list[str] = None
    advantage_key: str | None = None
    effector_indices: list[int] = None
    weight: float | None = None
    
    points_workspace: dict | None = None
    max_npoints: int = 4096
    augment_pc_rot: int = 0 # 0: no rotation augmentation on z-axis, [-rot, rot], unit: degrees
    point_cloud_dirname: str | None = None

    image_size: int | None = None

    class_name: str = 'LeRobotDataset'


@dataclass
class DataConfig:
    mm_datasets: list[MMDatasetConfig] = field(default_factory=list)
    lerobot_datasets: list[LerobotConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "DataConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            mm_datasets=[MMDatasetConfig(**d) for d in raw.get("mm_datasets") or []],
            lerobot_datasets=[LerobotConfig(**d) for d in raw.get("lerobot_datasets") or []],
        )
