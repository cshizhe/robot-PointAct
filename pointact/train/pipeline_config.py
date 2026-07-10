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

import warnings
from dataclasses import dataclass, field
from typing import List

from transformers import TrainingArguments


@dataclass
class TrainPipelineConfig(TrainingArguments):
    """model selection"""
    model_class: str | None = None

    """ model parameters """
    model_name_or_path: str | None = field(default=None)
    vlm_name_or_path: str | None = field(default="../pretrained/Qwen2.5-VL-3B-Instruct")
    processor_name_or_path: str | None = field(default=None)
    chat_template: str | None = field(default="scripts/chat_template.json")
    action_act: str | None = field(default="linear")
    chunk_size: int | None = field(default=16)
    max_action_dim: int | None = field(default=32)
    max_state_dim: int | None = field(default=64)
    use_robot_state: bool = field(default=True)

    """pi model"""
    pi_state_action_norm_file: str | None = field(default=None)
    pi_train_expert_only: bool = field(default=False)
    advantage_condition_dropout: float = field(default=0.3)
    advantage_cfg_scale: float = field(default=1.0)

    """encdec3d model parameters"""
    ctx_embed_size: int = field(default=256)
    time_embed_size: int = field(default=256)
    ptv3_patch_size: int = field(default=1024)
    ptv3_enc_mode: bool = field(default=True)
    ptv3_enc_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 384, 512])
    ptv3_enc_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 6, 2])
    ptv3_enc_num_head: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    ptv3_dec_channels: List[int] = field(default_factory=lambda: [128, 128, 256, 384])
    ptv3_dec_depths: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    ptv3_dec_num_head: List[int] = field(default_factory=lambda: [4, 4, 8, 16])
    ptv3_apply_point_ca: bool = field(default=True)
    ptv3_input_channels: int = field(default=6)
    ptv3_backend: str = field(default="concerto")
    # initialize the point cloud transformer v3
    ptv3_init_ckpt_file: str = field(default=None)

    ptv3_clf_head_pos_bins: int = field(default=100)
    regression_head_heatmap_temp: float = field(default=0.1)
    
    action_regression_loss: str = field(default='l2') # l1, l2
    action_head_pos_center: str = field(default='moe') # moe, zero

    # diffusion
    flow_matching_target: str = field(default='x-pred')    # x-pred, v-pred
    flow_matching_loss: str = field(default='x-loss')      # x-loss, v-loss
    
    """qwen2.5-vl vision parameters"""

    image_min_pixels: int | None = field(default=64 * 28 * 28)
    image_max_pixels: int | None = field(default=350 * 28 * 28) # merged patch size = 28
    video_min_pixels: int | None = field(default=64 * 28 * 28)
    video_max_pixels: int | None = field(default=128 * 28 * 28)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = field(default=1.0)

    """dataset parameters"""
    data_path: str = field(default=None, metadata={"help": "Path to training data or yaml config."})
    train_mm_only: bool = False
    train_lerobot_only: bool = True
    lerobot_data_video_backend: str | None = "torchcodec"

    """ training parameters """
    cache_dir: str | None = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_lm_head: bool = field(default=False)
    attn_implementation: str = field(default="sdpa")  # sdpa, flash_attention_2, flash_attention_3

    color_aug: bool = field(default=True) # perform color augmentation to input images
    image_aug: bool = field(default=True) # perform color+crop augmentation to input images

    lora_enable: bool = field(default=False)
    vision_lora: bool = field(default=False)
    use_dora: bool = field(default=False)
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")

    vision_lr: float | None = None
    merger_lr: float | None = None
    llm_lr: float | None = None
    lora_namespan_exclude: str | None = field(
        default=None, metadata={"help": "Comma-separated name spans to exclude for LoRA"}
    )
    num_lora_modules: int = field(default=-1)

    """experiment parameters"""
    # if output_dir is not specified, it will be set to {output_base}/{timestamp}-{run_name}
    output_base: str = field(default="outputs", metadata={"help": "Base directory for output."})

    def __post_init__(self):
        super().__post_init__()

        """check validity"""
        if self.train_lerobot_only and self.train_mm_only:
            self.train_mm_only = False
            warnings.warn("`train_mm_only` is set to False when `train_lerobot_only` is True.", stacklevel=2)

        if self.lora_enable and not self.freeze_llm:
            self.freeze_llm = True
            warnings.warn("`freeze_llm` is set to True when `lora_enable`.", stacklevel=2)

        if not self.lora_enable and self.vision_lora:
            self.vision_lora = False
            warnings.warn("`vision_lora` is set to False when `lora_enable` is False.", stacklevel=2)

        if self.vision_lora and not self.freeze_vision_tower:
            self.freeze_vision_tower = True
            warnings.warn("`freeze_vision_tower` is set to True when `vision_lora` is True.", stacklevel=2)

        if isinstance(self.lora_namespan_exclude, str):
            self.lora_namespan_exclude = [
                namespan.strip() for namespan in self.lora_namespan_exclude.split(",") if namespan.strip()
            ]
        elif self.lora_namespan_exclude is None:
            self.lora_namespan_exclude = []

        if self.processor_name_or_path is None:
            self.processor_name_or_path = self.model_name_or_path or self.vlm_name_or_path

        if self.output_dir == "trainer_output":  # default
            import datetime as dt

            self.output_dir = f"{self.output_base}/{dt.datetime.now():%Y-%m-%d/%H-%M-%S}-{self.run_name}"
