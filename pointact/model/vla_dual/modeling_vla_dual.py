from dataclasses import dataclass

import torch
from torch import Tensor
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_vla_dual import VLADualFlowMatchingConfig
from pointact.model.backbone.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from pointact.model.action_head.flow_matching_action_head import FlowmatchingActionHead
from pointact.model.utils import create_mm_token_type_ids


logger = logging.get_logger(__name__)

@dataclass
class VLADualOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    action_loss: torch.FloatTensor | None = None
    text_loss: torch.FloatTensor | None = None

    actions: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None

    past_key_values: list[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    rope_deltas: torch.LongTensor | None = None


class VLADualFlowMatchingModel(PreTrainedModel, GenerationMixin):
    config_class = VLADualFlowMatchingConfig
    supports_gradient_checkpointing = True

    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True
    _skip_keys_device_placement = "past_key_values"

    def __init__(
        self,
        config: VLADualFlowMatchingConfig,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration = None,
    ):
        super().__init__(config)

        self.vlm_backbone = vlm_backbone or Qwen2_5_VLForConditionalGeneration(self.config)
        self.action_head = FlowmatchingActionHead(self.config.action_head_config)

        self.post_init()
        self.action_head = self.action_head.to(dtype=torch.float32)

    def save_pretrained(self, *args, **kwargs):
        # When saving the model, we do not want to save the original format of the model, as it will cause issues when loading the new model.
        kwargs.setdefault("save_original_format", False)
        return super().save_pretrained(*args, **kwargs)
    
    def get_input_embeddings(self):
        return self.vlm_backbone.get_input_embeddings()

    def _sync_action_head_num_inference_timesteps(self):
        num_denoise_steps = getattr(self.config, "num_denoise_steps", None)
        if num_denoise_steps is None:
            return
        self.action_head.num_inference_timesteps = num_denoise_steps
        self.action_head.config.num_inference_timesteps = num_denoise_steps

    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        """Embed text and replace vision placeholders with VLM features."""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_embeds = self.vlm_backbone.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.vlm_backbone.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds = self.vlm_backbone.get_video_features(pixel_values_videos, video_grid_thw).pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.vlm_backbone.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return inputs_embeds

    def compute_position_ids(
        self,
        input_ids: torch.LongTensor | None,
        image_grid_thw: torch.LongTensor | None,
        video_grid_thw: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor | None,
        attention_mask: torch.Tensor | None,
        past_key_values: list[torch.FloatTensor] | None,
        second_per_grid_ts: torch.Tensor | None = None,
    ) -> torch.LongTensor | None:
        mm_token_type_ids = None
        if input_ids is not None:
            mm_token_type_ids = create_mm_token_type_ids(
                input_ids, self.config.image_token_id, self.config.video_token_id
            )

        return self.vlm_backbone.model.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

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
    ) -> VLADualOutputWithPast:
        """multi-modal forward pass, including image, video, state, action, and language."""

        inputs_embeds = self.embed_prefix(
            input_ids,
            inputs_embeds,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None:
            position_ids = self.compute_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        # generation
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
        if actions is not None:
            backbone_outputs = {
                "backbone_features": hidden_states,
                "backbone_attention_mask": attention_mask,
            }
            backbone_outputs = self.action_head.prepare_input(backbone_outputs)
            action_inputs = {
                "embodiment_id": torch.zeros(states.size(0), dtype=torch.long, device=states.device),
                "state": states,    # (B, D)
                "action": actions,  # (B, T, D)
                "action_mask": action_is_pad.logical_not(),   # (B, T)
            }
            action_inputs = self.action_head.prepare_input(action_inputs)
            action_head_outputs = self.action_head(backbone_outputs, action_inputs)
            # for n, p in self.action_head.named_parameters():  # make sure the param is float32
            #     print(n, p.size(), p.dtype)
            
            action_loss = action_head_outputs.loss
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

        return VLADualOutputWithPast(
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
        self._sync_action_head_num_inference_timesteps()

        # embed prefix
        if inputs_embeds is None:
            inputs_embeds = self.embed_prefix(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

        if position_ids is None:
            position_ids = self.compute_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        outputs = self.vlm_backbone.model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            cache_position=cache_position if cache_position is not None else None,
        )
        hidden_state = outputs.last_hidden_state

        input_dtype = next(self.action_head.parameters()).dtype
        backbone_outputs = {
            "backbone_features": hidden_state.to(dtype=input_dtype),
            "backbone_attention_mask": attention_mask,
        }
        backbone_outputs = self.action_head.prepare_input(backbone_outputs)
        action_inputs = {
            "embodiment_id": torch.zeros(states.size(0), dtype=torch.long, device=states.device),
            "state": states.to(dtype=input_dtype),    # (B, D)
        }
        action_inputs = self.action_head.prepare_input(action_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        x_t = action_head_outputs.action_pred
            
        return x_t, outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.vlm_backbone.prepare_inputs_for_generation(*args, **kwargs)

    def _expand_inputs_for_generation(self, *args, **kwargs):
        return self.vlm_backbone._expand_inputs_for_generation(*args, **kwargs)


# VLADualFlowMatchingModel.register_for_auto_class()
