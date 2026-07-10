import random

import torch
from lerobot.constants import ACTION, OBS_STATE

from pointact.data.robot.base import LeRobotDatasetMixin
from pointact.data.robot.registry import register_robot_dataset


@register_robot_dataset("LeRobotDataset")
class LeRobotDataset(LeRobotDatasetMixin):
    """2D LeRobot dataset."""
        
    def set_feature_keys(self, video_keys=None, state_keys=None, action_keys=None, **kwargs):
        self.select_video_keys = self.meta.video_keys if video_keys is None else video_keys
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
        self.advantage_key = kwargs.get("advantage_key", None)
        self.select_feature_keys = self.select_video_keys + self.select_state_keys + self.select_action_keys
        self.select_action_is_pad_keys = [f"{key}_is_pad" for key in self.select_action_keys]

    def __getitem__(self, idx, delta_indices: dict = None) -> dict:
        delta_indices = delta_indices or self.delta_indices

        if self.weight is not None:
            idx = random.randint(0, self.num_frames - 1)

        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()

        item, query_indices = self.query_action_chunk(item, idx, ep_idx, delta_indices)
        item = self.add_video_frames(item, ep_idx, query_indices)
        self.apply_image_transforms(item)
        self.select_task_text(item, ep_idx, idx)
        self.convert_eef_rotation(item)
        self.normalize_state_action(item)

        return self.post_process(item)

    def post_process(self, item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        advantage = item.get(self.advantage_key) if self.advantage_key is not None else None
        item = {key: item[key] for key in (self.select_feature_keys + ["task"] + self.select_action_is_pad_keys)}

        if self.advantage_key is not None:
            if advantage is None:
                advantage = torch.tensor(1, dtype=torch.long)
            if not isinstance(advantage, torch.Tensor):
                advantage = torch.as_tensor(advantage)
            item["advantage"] = advantage.flatten()[0].to(dtype=torch.long)

        return item
