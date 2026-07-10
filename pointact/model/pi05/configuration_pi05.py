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

from transformers.configuration_utils import PretrainedConfig

DEFAULT_IMAGE_SIZE = 224


class PI05Config(PretrainedConfig):
    model_type = "pi05"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.paligemma_variant = kwargs.get("paligemma_variant", "gemma_2b")
        self.action_expert_variant = kwargs.get("action_expert_variant", "gemma_300m")

        # Number of action steps to predict
        self.action_chunk_size = kwargs.get("action_chunk_size", 50)

        # Shorter state and action vectors will be padded to these dimensions
        self.max_state_dim = kwargs.get("max_state_dim", 32)
        self.max_action_dim = kwargs.get("max_action_dim", 32)

        # Flow matching parameters: see openpi `PI0Pytorch`
        self.num_denoise_steps = kwargs.get("num_denoise_steps", 10)
        self.time_sampling_beta_alpha = kwargs.get("time_sampling_beta_alpha", 1.5)
        self.time_sampling_beta_beta = kwargs.get("time_sampling_beta_beta", 1.0)
        self.time_sampling_scale = kwargs.get("time_sampling_scale", 0.999)
        self.time_sampling_offset = kwargs.get("time_sampling_offset", 0.001)
        self.min_period = kwargs.get("min_period", 4e-3)
        self.max_period = kwargs.get("max_period", 4.0)

        image_resolution = kwargs.get("image_resolution", None)
        self.image_resolution = (
            (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
            if image_resolution is None
            else tuple(image_resolution)
        )

        # Training related parameters
        self.dtype = kwargs.get("dtype", "float32") # Options: bfloat16, float32

        self.gradient_checkpointing = kwargs.get("gradient_checkpointing", False) # Enable gradient checkpointing for memory optimization

        self.freeze_vision_encoder = kwargs.get("freeze_vision_encoder", False) # Freeze only the vision encoder
        self.train_expert_only = kwargs.get("train_expert_only", False) # Freeze entire VLM, train only action expert and projections


        self._validate()

    def _validate(self):
        if self.paligemma_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid paligemma_variant: {self.paligemma_variant}")

        if self.action_expert_variant not in ["gemma_300m", "gemma_2b"]:
            raise ValueError(f"Invalid action_expert_variant: {self.action_expert_variant}")

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")
