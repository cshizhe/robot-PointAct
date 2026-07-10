from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from transformers.generation import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from pointact.model.utils import create_mm_token_type_ids

from pointact.model.backbone.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from pointact.model.vla_dual.modeling_vla_dual import VLADualOutputWithPast
from pointact.model.vla_pointact.action_head_3d.ptv3_backbone import (
    PointTransformerUnet,
    PointTransformerUnetWithAction,
)
from pointact.model.vla_pointact.action_head_3d.regression_head import (
    PointMoERegressionMLPActionHead,
    PointWithActionRegressionMLPActionHead,
)
from pointact.model.vla_pointact.action_head_3d.classification_head import (
    PointMoEClassificationMLPActionHead,
    PointWithActionCenteredClassificationMLPActionHead,
    PointWithActionMoEClassificationMLPActionHead,
)
from pointact.model.action_head.flow_matching_action_head import (
    CategorySpecificMLP,
)
from .configuration_pointact import VLAEncDec3DModelConfig

logger = logging.get_logger(__name__)



class VLAEncDec3DBaseModel(PreTrainedModel, GenerationMixin, ABC):
    config_class = VLAEncDec3DModelConfig
    supports_gradient_checkpointing = True

    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _can_compile_fullgraph = True
    _skip_keys_device_placement = "past_key_values"

    def __init__(
        self,
        config: VLAEncDec3DModelConfig,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration = None,
    ):
        super().__init__(config)
        self.vlm_backbone = vlm_backbone or Qwen2_5_VLForConditionalGeneration(self.config)

    def save_pretrained(self, *args, **kwargs):
        # When saving the model, we do not want to save the original format of the model, as it will cause issues when loading the new model.
        kwargs.setdefault("save_original_format", False)
        return super().save_pretrained(*args, **kwargs)
    
    def get_input_embeddings(self):
        return self.vlm_backbone.get_input_embeddings()

    @abstractmethod
    def compute_action_loss(
        self,
        points: torch.Tensor,
        npoints_in_batch: torch.Tensor,
        ctx_embeds: torch.Tensor,
        ctx_lens: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_is_pad: torch.Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute the concrete action head loss."""
        raise NotImplementedError

    @abstractmethod
    def compute_action(
        self,
        points: torch.Tensor,
        npoints_in_batch: torch.Tensor,
        ctx_embeds: torch.Tensor,
        ctx_lens: torch.Tensor,
        states: torch.Tensor,
        return_intermediate_value: bool = False,
        **kwargs,
    ) -> Tensor:
        """Predict actions with the concrete action head."""
        raise NotImplementedError

    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        **kawargs,
    ) -> tuple[torch.FloatTensor, torch.Tensor, torch.Tensor]:
        """Embed the suffix"""
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

        return inputs_embeds

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
        points: torch.Tensor | None = None,
        npoints_in_batch: torch.Tensor | None = None,
        input_id_lens: list[int] = None,
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
                points=points,
                npoints_in_batch=npoints_in_batch,
                input_id_lens=input_id_lens,
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
            action_loss = self.compute_action_loss(
                points, npoints_in_batch, 
                hidden_states, input_id_lens,
                states, actions, action_is_pad,
            )
            if isinstance(action_loss, tuple):
                action_loss = action_loss[0]
            loss = action_loss

        text_loss = None
        logits = None
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
        points: torch.Tensor | None = None,
        npoints_in_batch = None,
        input_id_lens: list[int] = None,
        **kwargs,
    ) -> Tensor:
        """Sample actions from the model."""

        # prepare position_ids and kv_cache
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

        # embed prefix
        if inputs_embeds is None:
            inputs_embeds = self.embed_prefix(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        outputs = self.vlm_backbone.model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=True,
            cache_position=cache_position if cache_position is not None else None,
        )
        hidden_states = outputs.last_hidden_state

        pred_actions = self.compute_action(
            points, npoints_in_batch,
            hidden_states, input_id_lens,
            states, 
        )
    
        return pred_actions, outputs
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.vlm_backbone.prepare_inputs_for_generation(*args, **kwargs)

    def _expand_inputs_for_generation(self, *args, **kwargs):
        return self.vlm_backbone._expand_inputs_for_generation(*args, **kwargs)


class VLAEncDec3DClassificationModel(VLAEncDec3DBaseModel):
    """VLAEncDec3DModel with a classification action head per point.

    This variant encodes the point cloud with the standard Point Transformer V3
    backbone and concatenates action timestep embeddings with the encoded point
    embeddings. The action head is a mixture-of-experts MLP that predicts
    discretized actions.

    Position is classified independently for each point using spatial bins.
    Rotation and gripper state are predicted from the max pooled point embedding.
    """
    def __init__(
        self,
        config: VLAEncDec3DModelConfig,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration = None,
    ):
        assert config.action_head_pos_center == "moe"

        super().__init__(config, vlm_backbone)

        hidden_size = self.config.text_config.hidden_size
        max_state_dim = self.config.max_state_dim

        self.ctx_proj = nn.Linear(hidden_size, self.config.ctx_embed_size)
        if self.config.use_robot_state:
            self.state_encoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=max_state_dim,
                hidden_dim=None,
                output_dim=self.config.ctx_embed_size,
                dropout=0.
            )
        self.ptv3_model = PointTransformerUnet(
            input_size=self.config.ptv3_input_channels, 
            ctx_embed_size=self.config.ctx_embed_size, 
            voxel_size=0.01,
            patch_size=self.config.ptv3_patch_size,
            enc_channels=self.config.ptv3_enc_channels,
            enc_depths=self.config.ptv3_enc_depths,
            enc_num_head=self.config.ptv3_enc_num_head,
            dec_channels=self.config.ptv3_dec_channels,
            dec_depths=self.config.ptv3_dec_depths,
            dec_num_head=self.config.ptv3_dec_num_head,
            enc_mode=self.config.ptv3_enc_mode,
            ptv3_backend=self.config.ptv3_backend,
        )
        self.action_head = PointMoEClassificationMLPActionHead(
            self.ptv3_model.output_size, 
            self.config.action_chunk_size,
            dropout=0.2, 
            pos_bin_size=0.01, 
            pos_bins=self.config.ptv3_clf_head_pos_bins,
            pos_heatmap_type="plain",
            euler_resolution=5, 
        )

        self.post_init()
        self.to_float32_action_head()

    def to_float32_action_head(self):
        self.ctx_proj = self.ctx_proj.to(dtype=torch.float32)
        if self.config.use_robot_state:
            self.state_encoder = self.state_encoder.to(dtype=torch.float32)
        self.ptv3_model = self.ptv3_model.to(dtype=torch.float32)
        self.action_head = self.action_head.to(dtype=torch.float32)

    def compute_action_loss(
        self,
        points: torch.Tensor,
        npoints_in_batch: torch.Tensor,
        ctx_embeds: torch.Tensor,
        ctx_lens: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_is_pad: torch.Tensor,
        **kwargs,
    ) -> Tensor:
        
        outs = self.compute_action(
            points, npoints_in_batch, ctx_embeds, ctx_lens, states,
            return_intermediate_value=True,
        )
        
        action_masks = action_is_pad.logical_not()
        
        action_loss, (pos_loss, rot_loss, open_loss) = self.action_head.compute_loss(
            outs["disc_actions"], actions, action_masks,
            outs["npoints_in_batch"], outs["point_coords"]
        )
    
        return action_loss, (pos_loss, rot_loss, open_loss)
    
    def compute_action(
        self,
        points: torch.Tensor,
        npoints_in_batch: torch.Tensor,
        ctx_embeds: torch.Tensor,
        ctx_lens: torch.Tensor,
        states: torch.Tensor,
        return_intermediate_value: bool = False,
        **kwargs,
    ) -> Tensor:
        batch_size = ctx_embeds.size(0)
        device = ctx_embeds.device

        ctx_embeds = ctx_embeds.type(self.ctx_proj.weight.dtype)
        ctx_embeds = self.ctx_proj(ctx_embeds)
        
        embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        if self.config.use_robot_state:
            states = states.type(next(self.state_encoder.parameters()).dtype)
            state_embs = self.state_encoder(states.unsqueeze(1), embodiment_id)
            ctx_embeds = torch.cat([state_embs, ctx_embeds], dim=1)
            ctx_lens = ctx_lens + 1

        point_fts, point_coords, point_offsets = self.ptv3_model(
            points, npoints_in_batch, ctx_embeds, ctx_lens
        )
        out_npoints_in_batch = torch.diff(
            point_offsets, prepend=torch.tensor([0], device=device, dtype=torch.long)
        )

        xt, xr, xo, pred_actions = self.action_head(
            point_fts, point_coords, out_npoints_in_batch,
            return_cont_actions=(not return_intermediate_value)
        )
        
        if return_intermediate_value:
            outs = {
                "disc_actions": (xt, xr, xo),
                "point_coords": point_coords,
                "npoints_in_batch": out_npoints_in_batch,
            }
            return outs
    
        return pred_actions
    

class VLAEncDec3DRegressionModel(VLAEncDec3DBaseModel):
    """VLAEncDec3DModel with a regression action head per point.

    Action is regressed independently for each point and then weighted average using a heatmap.
    """
    def __init__(
        self,
        config: VLAEncDec3DModelConfig,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration = None,
    ):
        assert config.action_head_pos_center == "moe"

        super().__init__(config, vlm_backbone)

        hidden_size = self.config.text_config.hidden_size
        max_action_dim = self.config.max_action_dim
        max_state_dim = self.config.max_state_dim

        self.ctx_proj = nn.Linear(hidden_size, self.config.ctx_embed_size)
        if self.config.use_robot_state:
            self.state_encoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=max_state_dim,
                hidden_dim=None,
                output_dim=self.config.ctx_embed_size,
                dropout=0.
            )

        self.ptv3_model = PointTransformerUnet(
            input_size=self.config.ptv3_input_channels, 
            ctx_embed_size=self.config.ctx_embed_size, 
            voxel_size=0.01,
            patch_size=self.config.ptv3_patch_size,
            enc_channels=self.config.ptv3_enc_channels,
            enc_depths=self.config.ptv3_enc_depths,
            enc_num_head=self.config.ptv3_enc_num_head,
            dec_channels=self.config.ptv3_dec_channels,
            dec_depths=self.config.ptv3_dec_depths,
            dec_num_head=self.config.ptv3_dec_num_head,
            enc_mode=self.config.ptv3_enc_mode,
            ptv3_backend=self.config.ptv3_backend,
        )
        self.action_head = PointMoERegressionMLPActionHead(
            self.ptv3_model.output_size, 
            max_action_dim, 
            self.config.action_chunk_size,
            dropout=0.2, 
            heatmap_temp=self.config.regression_head_heatmap_temp,
        )
        
        self.post_init()
        self.to_float32_action_head()

    def to_float32_action_head(self):
        self.ctx_proj = self.ctx_proj.to(dtype=torch.float32)
        if self.config.use_robot_state:
            self.state_encoder = self.state_encoder.to(dtype=torch.float32)
        self.ptv3_model = self.ptv3_model.to(dtype=torch.float32)
        self.action_head = self.action_head.to(dtype=torch.float32)

    def compute_action_loss(
        self,
        points: torch.Tensor,
        npoints_in_batch: torch.Tensor,
        ctx_embeds: torch.Tensor,
        ctx_lens: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_is_pad: torch.Tensor,
        **kwargs,
    ) -> Tensor:
        # The groundtruth action is substracted by point_centers
        pred_actions = self.compute_action(
            points, npoints_in_batch, ctx_embeds, ctx_lens, states,
        )
        
        if self.config.action_regression_loss == "l2":
            action_losses = F.mse_loss(pred_actions, actions, reduction="none")
        else:
            action_losses = F.l1_loss(pred_actions, actions, reduction="none")
        action_mask = action_is_pad.logical_not()
        action_losses = action_losses * action_mask.unsqueeze(-1)
        valid_action_count = action_mask.sum().clamp_min(1).to(action_losses.dtype)
        action_loss = action_losses.sum() / (valid_action_count * action_losses.size(-1))

        # TODO: here we assume using rot6d rotation
        pos_loss = action_losses[..., :3].sum() / (valid_action_count * 3)
        rot_loss = action_losses[..., 3:9].sum() / (valid_action_count * 6)
        open_loss = action_losses[..., 9].sum() / valid_action_count
        
        return action_loss, (pos_loss, rot_loss, open_loss)
    
    def compute_action(
        self,
        points: torch.Tensor,
        npoints_in_batch: torch.Tensor,
        ctx_embeds: torch.Tensor,
        ctx_lens: torch.Tensor,
        states: torch.Tensor,
        **kwargs,
    ) -> Tensor:
        batch_size = ctx_embeds.size(0)
        device = ctx_embeds.device

        ctx_embeds = ctx_embeds.type(self.ctx_proj.weight.dtype)
        ctx_embeds = self.ctx_proj(ctx_embeds)
        
        embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)

        if self.config.use_robot_state:
            states = states.type(next(self.state_encoder.parameters()).dtype)
            state_embs = self.state_encoder(states.unsqueeze(1), embodiment_id)
            ctx_embeds = torch.cat([state_embs, ctx_embeds], dim=1)
            ctx_lens = ctx_lens + 1

        point_fts, point_coords, point_offsets = self.ptv3_model(
            points, npoints_in_batch, ctx_embeds, ctx_lens
        )
        out_npoints_in_batch = torch.diff(
            point_offsets, prepend=torch.tensor([0], device=device, dtype=torch.long)
        )
        # print(point_fts.size(), npoints_in_batch, out_npoints_in_batch)
        pred_actions = self.action_head(
            point_fts, 
            out_npoints_in_batch,
        )

        return pred_actions
    

class VLAEncDec3DWithActionClassificationModel(VLAEncDec3DBaseModel):
    def __init__(
        self,
        config: VLAEncDec3DModelConfig,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration = None,
    ):
        super().__init__(config, vlm_backbone)

        hidden_size = self.config.text_config.hidden_size

        self.ptv3_model = PointTransformerUnetWithAction(
            input_size=self.config.ptv3_input_channels, 
            ctx_embed_size=self.config.ctx_embed_size, 
            voxel_size=0.01,
            patch_size=self.config.ptv3_patch_size,
            enc_channels=self.config.ptv3_enc_channels,
            enc_depths=self.config.ptv3_enc_depths,
            enc_num_head=self.config.ptv3_enc_num_head,
            dec_channels=self.config.ptv3_dec_channels,
            dec_depths=self.config.ptv3_dec_depths,
            dec_num_head=self.config.ptv3_dec_num_head,
            enc_mode=self.config.ptv3_enc_mode,
            apply_point_ca=self.config.ptv3_apply_point_ca,
            ptv3_backend=self.config.ptv3_backend,
        )
        if self.config.action_head_pos_center != "moe":
            # the mlp layers in the last point layer won"t be used
            if self.config.ptv3_enc_mode:
                self.ptv3_model.ptv3_model.enc[-1][-1].mlp = nn.Identity()
                self.ptv3_model.ptv3_model.enc[-1][-1].norm1 = nn.Identity()
                self.ptv3_model.ptv3_model.enc[-1][-1].norm2 = nn.Identity()
                self.ptv3_model.ptv3_model.enc[-1][-2].mlp = nn.Identity()
                self.ptv3_model.ptv3_model.enc[-1][-2].norm2 = nn.Identity()
            else:
                self.ptv3_model.ptv3_model.dec[-1][-1].mlp = nn.Identity()
                self.ptv3_model.ptv3_model.dec[-1][-1].norm1 = nn.Identity()
                self.ptv3_model.ptv3_model.dec[-1][-1].norm2 = nn.Identity()
                self.ptv3_model.ptv3_model.dec[-1][-2].mlp = nn.Identity()
                self.ptv3_model.ptv3_model.dec[-1][-2].norm2 = nn.Identity()

        self.ctx_proj = nn.Linear(hidden_size, self.config.ctx_embed_size)

        if self.config.use_robot_state:
            self.state_encoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.config.max_state_dim,
                hidden_dim=None,
                output_dim=self.ptv3_model.enc_channels[0],
                dropout=0.
            )

        self.position_embedding = nn.Embedding(
            config.action_chunk_size, self.ptv3_model.enc_channels[0]
        )

        if self.config.action_head_pos_center == "moe":
            self.action_head = PointWithActionMoEClassificationMLPActionHead(
                self.ptv3_model.output_size, 
                self.config.action_chunk_size,
                dropout=0.2, 
                euler_resolution=5, 
                pos_bin_size=0.01, 
                pos_bins=self.config.ptv3_clf_head_pos_bins, 
                pos_heatmap_type="plain"
            )
        elif self.config.action_head_pos_center == "zero":
            self.action_head = PointWithActionCenteredClassificationMLPActionHead(
                self.ptv3_model.output_size, 
                self.config.action_chunk_size,
                dropout=0.2, 
                euler_resolution=5, 
                pos_bin_size=0.01, 
                pos_bins=self.config.ptv3_clf_head_pos_bins, 
                pos_heatmap_type="plain"
            )
        else:
            raise NotImplementedError(f"unsupported pos_center {self.config.action_head_pos_center}")
    
        self.post_init()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.to_float32_action_head()

    def to_float32_action_head(self):
        self.ctx_proj = self.ctx_proj.to(dtype=torch.float32)
        if self.config.use_robot_state:
            self.state_encoder = self.state_encoder.to(dtype=torch.float32)
        self.ptv3_model = self.ptv3_model.to(dtype=torch.float32)
        self.position_embedding = self.position_embedding.to(dtype=torch.float32)
        self.action_head = self.action_head.to(dtype=torch.float32)

    def compute_action_loss(
        self,
        points: torch.Tensor,
        npoints_in_batch: torch.Tensor,
        ctx_embeds: torch.Tensor,
        ctx_lens: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_is_pad: torch.Tensor,
        **kwargs,
    ) -> Tensor:
        outs = self.compute_action(
            points, npoints_in_batch, ctx_embeds, ctx_lens, states,
            return_intermediate_value=True,
        )

        action_masks = action_is_pad.logical_not()

        if self.config.action_head_pos_center == "moe":
            action_loss, (pos_loss, rot_loss, open_loss) = self.action_head.compute_loss(
                outs["disc_actions"], actions, action_masks,
                outs["npoints_in_batch"], outs["point_coords"]
            )
        elif self.config.action_head_pos_center == "zero":
            action_loss, (pos_loss, rot_loss, open_loss) = self.action_head.compute_loss(
                outs["disc_actions"], actions, action_masks,
                center_coords=None
            )
        else:
            raise NotImplementedError(f"unsupported {self.config.action_head_pos_center}")

        return action_loss, (pos_loss, rot_loss, open_loss)
 
    def compute_action(
        self,
        points: torch.Tensor,
        npoints_in_batch: torch.Tensor,
        ctx_embeds: torch.Tensor,
        ctx_lens: torch.Tensor,
        states: torch.Tensor,
        return_intermediate_value: bool = False,
        **kwargs,
    ) -> Tensor:
        device = ctx_embeds.device
        batch_size = ctx_embeds.size(0)
        
        ctx_embeds = ctx_embeds.type(self.ctx_proj.weight.dtype)
        ctx_embeds = self.ctx_proj(ctx_embeds)

        pos_ids = torch.arange(self.config.action_chunk_size, dtype=torch.long, device=device)
        action_features = self.position_embedding(pos_ids).unsqueeze(0).expand(batch_size, -1, -1)

        if self.config.use_robot_state:
            states = states.type(next(self.state_encoder.parameters()).dtype)
            embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)
            state_embs = self.state_encoder(states.unsqueeze(1), embodiment_id)
            action_features = torch.cat([state_embs, action_features], dim=1)

        point_fts, point_coords, point_offsets, action_out_embeds = self.ptv3_model(
            points, npoints_in_batch, ctx_embeds, ctx_lens,
            action_features
        )
        out_npoints_in_batch = torch.diff(
            point_offsets, prepend=torch.tensor([0], device=device, dtype=torch.long)
        )
        if self.config.use_robot_state:
            action_out_embeds = action_out_embeds[:, 1:]
        
        if self.config.action_head_pos_center == "moe":
            xt, xr, xo, pred_actions = self.action_head(
                action_out_embeds, point_fts, point_coords, out_npoints_in_batch,
                return_cont_actions=(not return_intermediate_value)
            )
        elif self.config.action_head_pos_center == "zero":
            xt, xr, xo, pred_actions = self.action_head(
                action_out_embeds,
                center_coords=None,
                return_cont_actions=(not return_intermediate_value)
            )
        else:
            raise NotImplementedError(f"unsupported {self.config.action_head_pos_center}")

        if return_intermediate_value:
            outs = {
                "disc_actions": (xt, xr, xo),
                "point_coords": point_coords,
                "npoints_in_batch": out_npoints_in_batch,
            }
            return outs

        return pred_actions
         

class VLAEncDec3DWithActionRegressionModel(VLAEncDec3DBaseModel):
    def __init__(
        self,
        config: VLAEncDec3DModelConfig,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration = None,
    ):
        super().__init__(config, vlm_backbone)

        hidden_size = self.config.text_config.hidden_size
        max_action_dim = self.config.max_action_dim

        self.ptv3_model = PointTransformerUnetWithAction(
            input_size=self.config.ptv3_input_channels, 
            ctx_embed_size=self.config.ctx_embed_size, 
            voxel_size=0.01,
            patch_size=self.config.ptv3_patch_size,
            enc_channels=self.config.ptv3_enc_channels,
            enc_depths=self.config.ptv3_enc_depths,
            enc_num_head=self.config.ptv3_enc_num_head,
            dec_channels=self.config.ptv3_dec_channels,
            dec_depths=self.config.ptv3_dec_depths,
            dec_num_head=self.config.ptv3_dec_num_head,
            enc_mode=self.config.ptv3_enc_mode,
            apply_point_ca=self.config.ptv3_apply_point_ca,
            ptv3_backend=self.config.ptv3_backend,
        )
        if self.config.action_head_pos_center != "moe":
            # the mlp layers in the last point layer won"t be used
            if self.config.ptv3_enc_mode:
                self.ptv3_model.ptv3_model.enc[-1][-1].mlp = nn.Identity()
                self.ptv3_model.ptv3_model.enc[-1][-1].norm1 = nn.Identity()
                self.ptv3_model.ptv3_model.enc[-1][-1].norm2 = nn.Identity()
                self.ptv3_model.ptv3_model.enc[-1][-2].mlp = nn.Identity()
                self.ptv3_model.ptv3_model.enc[-1][-2].norm2 = nn.Identity()
            else:
                self.ptv3_model.ptv3_model.dec[-1][-1].mlp = nn.Identity()
                self.ptv3_model.ptv3_model.dec[-1][-1].norm1 = nn.Identity()
                self.ptv3_model.ptv3_model.dec[-1][-1].norm2 = nn.Identity()
                self.ptv3_model.ptv3_model.dec[-1][-2].mlp = nn.Identity()
                self.ptv3_model.ptv3_model.dec[-1][-2].norm2 = nn.Identity()

        self.ctx_proj = nn.Linear(hidden_size, self.config.ctx_embed_size)

        if self.config.use_robot_state:
            self.state_encoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.config.max_state_dim,
                hidden_dim=None,
                output_dim=self.ptv3_model.enc_channels[0],
                dropout=0.
            )

        self.position_embedding = nn.Embedding(
            config.action_chunk_size, self.ptv3_model.enc_channels[0]
        )

        self.action_head = PointWithActionRegressionMLPActionHead(
            self.ptv3_model.output_size,
            max_action_dim,
            config.action_chunk_size,
            dropout=0.2, 
            use_heatmap=(config.action_head_pos_center == "moe"),
            heatmap_temp=config.regression_head_heatmap_temp,
        )
    
        self.post_init()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.to_float32_action_head()

    def to_float32_action_head(self):
        self.ctx_proj = self.ctx_proj.to(dtype=torch.float32)
        if self.config.use_robot_state:
            self.state_encoder = self.state_encoder.to(dtype=torch.float32)
        self.ptv3_model = self.ptv3_model.to(dtype=torch.float32)
        self.position_embedding = self.position_embedding.to(dtype=torch.float32)
        self.action_head = self.action_head.to(dtype=torch.float32)

    def compute_action_loss(
        self,
        points: torch.Tensor,
        npoints_in_batch: torch.Tensor,
        ctx_embeds: torch.Tensor,
        ctx_lens: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_is_pad: torch.Tensor,
        **kwargs,
    ) -> Tensor:
        pred_actions = self.compute_action(
            points, npoints_in_batch, ctx_embeds, ctx_lens, states,
        )
        if self.config.action_regression_loss == "l2":
            action_losses = F.mse_loss(pred_actions, actions, reduction="none")
        else:
            action_losses = F.l1_loss(pred_actions, actions, reduction="none")
        action_mask = action_is_pad.logical_not()
        action_losses = action_losses * action_mask.unsqueeze(-1)
        valid_action_count = action_mask.sum().clamp_min(1).to(action_losses.dtype)
        action_loss = action_losses.sum() / (valid_action_count * action_losses.size(-1))
        
        # TODO: here we assume using rot6d rotation
        pos_loss = action_losses[..., :3].sum() / (valid_action_count * 3)
        rot_loss = action_losses[..., 3:9].sum() / (valid_action_count * 6)
        open_loss = action_losses[..., 9].sum() / valid_action_count
        #print(action_loss, pos_loss, rot_loss, open_loss)
        
        return action_loss, (pos_loss, rot_loss, open_loss)
 
    def compute_action(
        self,
        points: torch.Tensor,
        npoints_in_batch: torch.Tensor,
        ctx_embeds: torch.Tensor,
        ctx_lens: torch.Tensor,
        states: torch.Tensor,
        **kwargs,
    ) -> Tensor:
        device = ctx_embeds.device
        batch_size = ctx_embeds.size(0)
        
        ctx_embeds = ctx_embeds.type(self.ctx_proj.weight.dtype)
        ctx_embeds = self.ctx_proj(ctx_embeds)

        pos_ids = torch.arange(self.config.action_chunk_size, dtype=torch.long, device=device)
        action_features = self.position_embedding(pos_ids).unsqueeze(0).expand(batch_size, -1, -1)

        if self.config.use_robot_state:
            states = states.type(next(self.state_encoder.parameters()).dtype)
            embodiment_id = torch.zeros(batch_size, dtype=torch.long, device=device)
            state_embs = self.state_encoder(states.unsqueeze(1), embodiment_id)
            action_features = torch.cat([state_embs, action_features], dim=1)

        point_fts, point_coords, point_offsets, action_out_embeds = self.ptv3_model(
            points, npoints_in_batch, ctx_embeds, ctx_lens,
            action_features
        )
        out_npoints_in_batch = torch.diff(
            point_offsets, prepend=torch.tensor([0], device=device, dtype=torch.long)
        )
        # print(out_npoints_in_batch)
        if self.config.use_robot_state:
            action_out_embeds = action_out_embeds[:, 1:]
        
        pred_actions = self.action_head(
            action_out_embeds,
            point_embeds=point_fts,
            npoints_in_batch=out_npoints_in_batch,
        )
    
        return pred_actions
         

# VLAEncDec3DClassificationModel.register_for_auto_class()
# VLAEncDec3DRegressionModel.register_for_auto_class()
# VLAEncDec3DWithActionClassificationModel.register_for_auto_class()
# VLAEncDec3DWithActionRegressionModel.register_for_auto_class()
