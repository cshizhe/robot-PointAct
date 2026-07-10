# Copyright 2025 EO-Robotics Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import copy
import math
import random

import torch
import ujson as json
from torch.utils.data import Dataset

from pointact.data.multimodal.prompts import build_interleaved_prompt, llava_to_openai
from pointact.data.robot.multi_data import MultiLeRobotDataset
from pointact.data.schema import MMDatasetConfig


class MultimodaDataset(Dataset):
    """Visual-language dataset, optionally interleaved with LeRobot samples."""

    def __init__(
        self,
        data_configs: list[MMDatasetConfig],
        max_seq_length: int = 16384,
        meta_dataset: MultiLeRobotDataset = None,
        max_action_dim: int = 32,
        max_state_dim: int = 64,
        chunk_size: int = 50,
        sample_actions: bool = True,
    ):
        super().__init__()
        self.data_configs = data_configs
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.chunk_size = chunk_size
        self.sample_actions = sample_actions
        self.meta_dataset = meta_dataset

        list_data_dict, dataset_lens = [], []
        for i, dataset in enumerate(data_configs):
            cur_data_dict = self.load_json_dataset(dataset.json_path)
            cur_data_dict = [
                line for line in cur_data_dict if line.get("seq_length", 0) <= max_seq_length
            ]
            cur_data_dict = self.sample_dataset(cur_data_dict, dataset.sampling_strategy)

            print(f"Loaded {len(cur_data_dict)} samples from {dataset.json_path}")
            dataset_lens.append(len(cur_data_dict))
            for data in cur_data_dict:
                data["vision_base_idx"] = i
            list_data_dict.extend(cur_data_dict)

        self.data = list_data_dict
        self.dataset_lens = dataset_lens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        raw_source = self.data[i]

        if "lerobot" in raw_source:
            items = []
            for lerobot_ref in raw_source["lerobot"]:
                repo_id, idx, _chunk_size = lerobot_ref.split(" ")
                items.append(self.get_metadata(repo_id, int(idx)))
            transformed_source = build_interleaved_prompt(
                items,
                raw_source,
                max_action_dim=self.max_action_dim,
                max_state_dim=self.max_state_dim,
                chunk_size=self.chunk_size,
                sample_actions=self.sample_actions,
            )
        else:
            transformed_source = copy.deepcopy(raw_source)

        transformed_source["conversations"] = llava_to_openai(
            transformed_source["conversations"], "video" in transformed_source
        )
        return transformed_source

    @staticmethod
    def load_json_dataset(json_path: str):
        if json_path.endswith(".jsonl"):
            with open(json_path) as f:
                return [json.loads(line.strip()) for line in f]
        if json_path.endswith(".json"):
            with open(json_path) as f:
                return json.load(f)
        raise ValueError(f"Unsupported file type: {json_path}")

    @staticmethod
    def sample_dataset(data: list[dict], sampling_strategy: str):
        sampling_number = None
        if ":" in sampling_strategy:
            sampling_strategy, sampling_number = sampling_strategy.split(":")
            if "%" in sampling_number:
                sampling_number = math.ceil(float(sampling_number.split("%")[0]) * len(data) / 100)
            else:
                sampling_number = int(sampling_number)

        if sampling_strategy == "first" and sampling_number is not None:
            return data[:sampling_number]
        if sampling_strategy == "end" and sampling_number is not None:
            return data[-sampling_number:]
        if sampling_strategy == "random" and sampling_number is not None:
            random.shuffle(data)
            return data[:sampling_number]
        return data

    def get_metadata(self, repo_id, idx, chunk_size=None):
        return self.meta_dataset.getitem_by_id(repo_id, idx, chunk_size=chunk_size)

    @property
    def vision_base_paths(self):
        return [dataset.vision_base_path for dataset in self.data_configs]
