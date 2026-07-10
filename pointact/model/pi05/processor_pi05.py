#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

import json
import numpy as np
import torch
from easydict import EasyDict
from lerobot.constants import ACTION, OBS_IMAGE, OBS_IMAGES, OBS_STATE
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from transformers.processing_utils import ProcessorMixin

from pointact.utils.rotation import convert_rotation
from pointact.utils.torch_utils import pad_vector

from .modeling_pi05 import resize_with_pad_torch


class PI05Processor(ProcessorMixin):
    """PI05 processor."""

    @classmethod
    def get_attributes(cls):
        return ["tokenizer"]

    def __init__(
        self, 
        tokenizer=None,
        image_resolution: tuple[int, int] = (224, 224),
        robot_state_action_norm_file: str | None = None,
        state_action_norm: dict | None = None,
        norm_stats: str = "q1_q99",
        robot_config: dict | None = None,
        tokenizer_name: str = "google/paligemma-3b-pt-224",
        chat_template: str | None = None,
    ):
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        super().__init__(tokenizer=tokenizer, chat_template=chat_template)

        self.image_resolution = tuple(image_resolution)
        self.norm_stats = self._parse_norm_stats(norm_stats)
        self.tokenizer_name = tokenizer_name
        self.robot_config = robot_config

        if state_action_norm is not None:
            self.state_action_norm = self._to_numpy(state_action_norm)
            self.robot_state_action_norm_file = robot_state_action_norm_file
        elif robot_state_action_norm_file is not None:
            self.robot_state_action_norm_file = str(robot_state_action_norm_file)
            with open(robot_state_action_norm_file, "r") as f:
                self.state_action_norm = self._to_numpy(json.load(f))
        else:
            raise ValueError(
                "PI05Processor requires robot_state_action_norm_file or state_action_norm "
                "for state/action normalization."
            )

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, dict):
            return {k: PI05Processor._to_numpy(v) for k, v in value.items()}
        return np.array(value, dtype=np.float32)

    @staticmethod
    def _parse_norm_stats(norm_stats: str | list[str] | tuple[str, str]) -> tuple[str, str]:
        if isinstance(norm_stats, (list, tuple)):
            if len(norm_stats) != 2:
                raise ValueError(f"Invalid norm_stats: {norm_stats}")
            return tuple(norm_stats)
        aliases = {
            "min_max": ("min", "max"),
            "q1_q99": ("q01", "q99"),
            "q01_q99": ("q01", "q99"),
        }
        if norm_stats in aliases:
            return aliases[norm_stats]
        else:
            raise ValueError(f"Invalid norm_stats: {norm_stats}")

    def _get_norm_bounds(self, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        low_name, high_name = self.norm_stats

        low = self.state_action_norm[f"{prefix}_{low_name}"]
        high = self.state_action_norm[f"{prefix}_{high_name}"]

        return low, high

    def _normalize_with_bounds(self, value: np.ndarray, prefix: str, clip_value=True) -> np.ndarray:
        low, high = self._get_norm_bounds(prefix)
        denom = np.maximum(high - low, 1e-6)
        normalized = (value - low) / denom * 2.0 - 1.0
        if clip_value:
            normalized = np.clip(normalized, -1.0, 1.0)
        return normalized

    def _unnormalize_with_bounds(self, value: np.ndarray, prefix: str) -> np.ndarray:
        low, high = self._get_norm_bounds(prefix)
        unnormalized = (value + 1.0) / 2.0 * (high - low) + low
        return unnormalized

    def prepare_images(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [V, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, V, C, H, W] format and normalized to [-1, 1].

        Returns:
            images: (V, C, H, W), [-1, 1]
            img_masks: (V, ), 1 for real images
        """
        channels_first = images.shape[1] in (1, 3)
        if channels_first:
            spatial_shape = images.shape[2:4]
        else:
            spatial_shape = images.shape[1:3]
            images = images.permute(0, 3, 1, 2) # (vhwc -> vchw)
        images = images.to(torch.float32)

        if tuple(spatial_shape) != tuple(self.image_resolution):
            images = resize_with_pad_torch(images, *self.image_resolution)

        images = images * 2.0 - 1.0

        img_masks = torch.ones(
            images.shape[:1],
            dtype=torch.bool,
            device=images.device,
        )

        return images, img_masks

    def _normalize_discretize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Discretize robot state into bins.
        Args:
        - state: Tensor of shape [state_dim, ]
        """
        state_np = state.numpy()
        state_np = self._normalize_with_bounds(state_np, "state")

        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        discretized_state = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return np.clip(discretized_state, 0, 255)
    
    def prepare_text_state(
        self,
        task: str,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.LongTensor, torch.Tensor, None]:
        """Build the language prompt for the model based on the task and robot state."""
        discretized_state = self._normalize_discretize_state(state)
        state_str = " ".join(map(str, discretized_state))

        task = task.strip()
        full_prompt = f"Task: {task}, State: {state_str};\nAction: "

        text_inputs = self.tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt")
        text_ids = text_inputs["input_ids"][0]
        text_masks = text_inputs["attention_mask"][0].bool()
        return text_ids, text_masks, None

    def _extract_states(self, batch: dict, state_keys: list) -> torch.Tensor:
        """Assume batch_size=1"""
        states = []
        for key in state_keys:
            value = batch[key][0]
            states.append(torch.from_numpy(value).float())
        state = torch.cat(states, dim=-1)
        return state

    def normalize_robot_action(self, action: torch.Tensor):
        normed_action = self._normalize_with_bounds(action.data.numpy(), "action", clip_value=False)
        normed_action = torch.from_numpy(normed_action).to(dtype=action.dtype, device=action.device)
        normed_action = pad_vector(normed_action, self.robot_config["max_action_dim"])        
        return normed_action

    def unnormalize_robot_action(self, action: np.ndarray):
        action_low, _ = self._get_norm_bounds("action")
        action = action[..., : action_low.shape[-1]]
        return self._unnormalize_with_bounds(action, "action")

    @torch.no_grad()
    def select_action(self, model, batch: dict, pred_rot_type: str | None = None, **kwargs):
        """
        Assume batch_size=1
        """
        device = next(model.parameters()).device

        repo_id = batch.get("repo_id", None) # list of str
        if repo_id is None or len(repo_id) == 0 or repo_id[0] is None:
            repo_id = list(self.robot_config["select_video_keys"].keys())[0]
        else:
            repo_id = repo_id[0]

        image_keys = self.robot_config["select_video_keys"][repo_id]
        images = [batch[key][0] for key in image_keys]
        # Convert PIL image to tensor
        images = np.stack([np.asarray(image) / 255. for image in images], 0)
        images = torch.from_numpy(images)
        images, img_masks = self.prepare_images(images)
        if images.ndim == 4:
            images = images.unsqueeze(0)
            img_masks = img_masks.unsqueeze(0)
        images = images.to(device)
        img_masks = img_masks.to(device)

        state_keys = self.robot_config["select_state_keys"][repo_id]
        states = self._extract_states(batch, state_keys)

        text_ids, text_masks, _ = self.prepare_text_state(batch["task"][0], states)
        text_ids = text_ids.unsqueeze(0).to(device)
        text_masks = text_masks.unsqueeze(0).to(device)

        actions = model.sample_actions(
            text_ids=text_ids,
            text_masks=text_masks,
            images=images,
            img_masks=img_masks,
            **kwargs,
        ).cpu().numpy()

        actions = self.unnormalize_robot_action(actions)

        if pred_rot_type == "euler":
            quat = convert_rotation(
                actions[..., 3:6], "euler", "quat", euler_order_src="xyz", quat_order_dst="xyzw",
            )
            actions = np.concatenate([actions[..., :3], quat, actions[..., 6:]], axis=-1)
        elif pred_rot_type == "rot6d":
            quat = convert_rotation(actions[..., 3:9], "rot6d", "quat", quat_order_dst="xyzw")
            actions = np.concatenate([actions[..., :3], quat, actions[..., 9:]], axis=-1)

        return EasyDict({ACTION: actions})
