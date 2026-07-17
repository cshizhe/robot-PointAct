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

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_eo1 import EO1VisionFlowMatchingConfig
from pointact.model.backbone.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from pointact.model.utils import create_sinusoidal_pos_embedding, create_mm_token_type_ids



@dataclass
class EO1VisionFlowMatchingOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    action_loss: torch.FloatTensor | None = None
    text_loss: torch.FloatTensor | None = None

    actions: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None

    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None


class EO1VisionActionProjector(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 2,
        activation_layer: str = "linear",
        bias: bool = True,
        device: Any = None,
        dtype: torch.dtype = torch.float32,
    ):
        layers = []
        in_dim = in_channels
        hidden_channels = [in_dim] * (num_layers - 1) + [out_channels]
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias, dtype=dtype, device=device))
            layers.append(ACT2FN[activation_layer])
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias, dtype=dtype, device=device))
        super().__init__(*layers)

    @property
    def dtype(self):
        return self[0].weight.dtype


class EO1VisionFlowMatchingModel(PreTrainedModel, GenerationMixin):
    config_class = EO1VisionFlowMatchingConfig
    supports_gradient_checkpointing = True

    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True
    _skip_keys_device_placement = "past_key_values"

    def __init__(
        self,
        config: EO1VisionFlowMatchingConfig,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration = None,
    ):
        super().__init__(config)

        hidden_size = self.config.text_config.hidden_size
        max_action_dim = self.config.max_action_dim
        max_state_dim = self.config.max_state_dim
        self.vlm_backbone = vlm_backbone or Qwen2_5_VLForConditionalGeneration(self.config)
        if self.config.use_robot_state:
            self.state_proj = nn.Linear(max_state_dim, hidden_size)
        self.action_in_proj = nn.Linear(max_action_dim, hidden_size)
        self.action_out_proj = EO1VisionActionProjector(
            hidden_size,
            max_action_dim,
            self.config.num_action_layers,
            self.config.action_act,
        )
        self.action_time_mlp_in = nn.Linear(hidden_size * 2, hidden_size)
        self.action_time_mlp_out = nn.Linear(hidden_size, hidden_size)

        self.post_init()
        self.to_float32_flow_matching_head()

    def save_pretrained(self, *args, **kwargs):
        # When saving the model, we do not want to save the original format of the model, as it will cause issues when loading the new model.
        kwargs.setdefault("save_original_format", False)
        return super().save_pretrained(*args, **kwargs)

    def get_input_embeddings(self):
        return self.vlm_backbone.get_input_embeddings()

    def to_float32_flow_matching_head(self):
        self.action_out_proj = self.action_out_proj.to(dtype=torch.float32)
        self.action_time_mlp_in = self.action_time_mlp_in.to(dtype=torch.float32)
        self.action_time_mlp_out = self.action_time_mlp_out.to(dtype=torch.float32)
        self.action_in_proj = self.action_in_proj.to(dtype=torch.float32)
        if self.config.use_robot_state:
            self.state_proj = self.state_proj.to(dtype=torch.float32)

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def replace_special_embeddings(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        special_features: torch.FloatTensor = None,
        special_token_ids: torch.LongTensor = None,
    ) -> torch.LongTensor:
        """Replace the special embeddings with the special features."""
        if special_features is not None and special_token_ids is not None:
            mask = input_ids == special_token_ids
            n_special_tokens = mask.sum().item()
            n_special_features = special_features.shape[0]
            assert n_special_tokens == n_special_features, (
                f"Special features and special tokens {special_token_ids} do not match: \
                tokens: {n_special_tokens}, features {n_special_features}"
            )
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            special_mask = mask_expanded.to(inputs_embeds.device)
            special_features = special_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_mask, special_features)
        return inputs_embeds

    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
    ) -> tuple[torch.FloatTensor, torch.Tensor, torch.Tensor]:
        """Embed the prefix"""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.vlm_backbone.get_image_features(pixel_values, image_grid_thw, return_dict=True).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.vlm_backbone.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.vlm_backbone.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.vlm_backbone.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if self.config.use_robot_state and states is not None:
            states = states.type(self.state_proj.weight.dtype)
            state_embs = self.state_proj(states)
            inputs_embeds = self.replace_special_embeddings(
                input_ids, inputs_embeds, state_embs, self.config.state_token_id
            )
        return inputs_embeds

    def embed_suffix(
        self,
        timestep: torch.Tensor,
        noisy_actions: torch.Tensor,
    ) -> torch.FloatTensor:
        """Embed the suffix"""
        time_embs = create_sinusoidal_pos_embedding(
            timestep,
            self.config.text_config.hidden_size,
            device=noisy_actions.device,
        )
        time_embs = time_embs.type(noisy_actions.dtype)
        noisy_actions = noisy_actions.type(self.action_in_proj.weight.dtype)
        action_embs = self.action_in_proj(noisy_actions)
        time_embs = time_embs[:, None, :].expand_as(action_embs)

        action_time_embs = torch.cat([action_embs, time_embs], dim=2)
        action_time_embs = self.action_time_mlp_in(action_time_embs)
        action_time_embs = F.silu(action_time_embs)
        action_time_embs = self.action_time_mlp_out(action_time_embs)
        return action_time_embs
        
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        rope_deltas: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        states: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
        **kwargs,
    ) -> EO1VisionFlowMatchingOutputWithPast:
        """multi-modal forward pass, including image, video, state, action, and language."""

        inputs_embeds = self.embed_prefix(
            input_ids,
            inputs_embeds,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            states,
        )

        if actions is not None:
            noise_mask = input_ids == self.config.action_token_id
            pass_mask = input_ids == self.config.action_pass_id
            mask = noise_mask | pass_mask  # (b s)

            pass_mask_in_action = pass_mask[mask]  # (n, )
            pass_mask_in_action = pass_mask_in_action.reshape(*actions.shape[:2], 1)  # (b, h, 1)

            # t_noise
            time = self.sample_time(actions.shape[0], inputs_embeds.device)  # (n,)
            time_expanded = time[:, None, None].repeat(1, actions.shape[1], 1)  # (b, h, 1)
            time_expanded[pass_mask_in_action] = 0.0

            noise = self.sample_noise(actions.shape, inputs_embeds.device)
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            action_time_embs = self.embed_suffix(time, x_t)
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            action_mask = mask_expanded.to(inputs_embeds.device)

            action_time_embs = action_time_embs.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(action_mask, action_time_embs)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None:
            # position_ids: (3, batch, seq_len): denoting temporal, height, width position ids
            mm_token_type_ids = create_mm_token_type_ids(
                input_ids, self.config.image_token_id, self.config.video_token_id
            )
            position_ids = self.vlm_backbone.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        output_actions = None
        if not (self.training or states is None):
            output_actions, outputs = self.sample_actions(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                states=states,
            )
        else:
            # training or states is None (text generation)
            outputs = self.vlm_backbone.model(
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
            )

        hidden_states = outputs.last_hidden_state

        loss = None
        action_loss = None
        v_t = None
        if actions is not None:
            action_time_embs = hidden_states[action_mask[..., 0]]
            action_time_embs = action_time_embs.type(self.action_out_proj.dtype)

            v_t = self.action_out_proj(action_time_embs)
            u_t = u_t.reshape(v_t.shape)
            v_t = v_t.type(u_t.dtype)

            losses = F.mse_loss(u_t, v_t, reduction="none")
            valid_loss_mask = (~pass_mask_in_action).reshape(-1, 1)
            if action_is_pad is not None:
                valid_loss_mask = valid_loss_mask & (~action_is_pad).reshape(-1, 1)

            valid_loss_mask = valid_loss_mask.to(losses.dtype)
            action_loss = (losses * valid_loss_mask).sum() / valid_loss_mask.sum().clamp_min(1)
            loss = action_loss

        logits = None
        text_loss = None
        if labels is not None:
            # only compute necessary logits, do not upcast to float if not computing loss
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.vlm_backbone.lm_head(hidden_states[:, slice_indices, :])

            text_loss = self.vlm_backbone.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
            loss = loss + text_loss if loss is not None else text_loss

        return EO1VisionFlowMatchingOutputWithPast(
            loss=loss,
            action_loss=action_loss,
            text_loss=text_loss,
            actions=output_actions,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.vlm_backbone.model.rope_deltas,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Sample actions from the model."""

        assert attention_mask.shape[-1] == input_ids.shape[-1], "Attention mask and input ids must have the same sequence length"

        if position_ids is None:
            # position_ids: (3, batch, seq_len): denoting temporal, height, width position ids
            mm_token_type_ids = create_mm_token_type_ids(
                input_ids, self.config.image_token_id, self.config.video_token_id
            )
            position_ids = self.vlm_backbone.model.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                inputs_embeds=None,
                past_key_values=None,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_prefix(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                states=states,
            )

        # pass prefix, update kvcache
        seq_len = input_ids.shape[-1]
        chunk_size = self.config.action_chunk_size
        suffix_len = -1  # exclude <|action_end|>
        prefix_len = seq_len - chunk_size - 1 # start of the action token

        outputs = self.vlm_backbone.model(
            position_ids=position_ids[..., :prefix_len],
            attention_mask=attention_mask[:, :prefix_len],
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds[:, :prefix_len],
            use_cache=True,
            cache_position=cache_position[:prefix_len] if cache_position is not None else None,
        )

        # denoising
        device = states.device
        actions_shape = (states.shape[0], chunk_size, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)

        x_t = noise.type(self.action_in_proj.weight.dtype)
        dt = torch.tensor(-1.0 / self.config.num_denoise_steps, device=device)
        time = torch.ones(inputs_embeds.shape[0], device=device)
        past_key_values, past_hidden_state = outputs.past_key_values, outputs.last_hidden_state

        action_mask = input_ids == self.config.action_token_id
        # while time[0] >= -dt / 2:
        for _ in range(self.config.num_denoise_steps):
            action_time_embs = self.embed_suffix(time, x_t)
            inputs_embeds[action_mask] = action_time_embs.to(inputs_embeds.dtype)

            # crop kvcache to only keep the embed prefix
            past_key_values.crop(prefix_len)

            outputs = self.vlm_backbone.model(
                position_ids=position_ids[..., prefix_len:suffix_len],
                attention_mask=attention_mask[:, :suffix_len],
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds[:, prefix_len:suffix_len],
                use_cache=True,
                cache_position=cache_position[prefix_len:suffix_len] if cache_position is not None else None,
            )
            action_time_embs = outputs.last_hidden_state[:, :chunk_size]
            action_time_embs = action_time_embs.type(self.action_out_proj.dtype)
            v_t = self.action_out_proj(action_time_embs)

            x_t += dt * v_t.reshape(x_t.shape)
            time += dt

        outputs.last_hidden_state = torch.cat([past_hidden_state, outputs.last_hidden_state], dim=1)
        return x_t, outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.vlm_backbone.prepare_inputs_for_generation(*args, **kwargs)

    def _expand_inputs_for_generation(self, *args, **kwargs):
        return self.vlm_backbone._expand_inputs_for_generation(*args, **kwargs)


# EO1VisionFlowMatchingModel.register_for_auto_class()
