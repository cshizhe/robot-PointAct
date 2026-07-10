import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from transformers.utils import logging

from .configuration_vla_dual import VLADualPointFlowMatchingConfig
from .modeling_vla_dual import VLADualFlowMatchingModel, VLADualOutputWithPast
from pointact.model.backbone.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from pointact.model.ptv3.concerto.model import PointTransformerV3


logger = logging.get_logger(__name__)


class VLADualPointFlowMatchingModel(VLADualFlowMatchingModel):
    config_class = VLADualPointFlowMatchingConfig

    def __init__(
        self,
        config: VLADualPointFlowMatchingConfig,
        vlm_backbone: Qwen2_5_VLForConditionalGeneration = None,
    ):
        super().__init__(config, vlm_backbone)

        self.ptv3_model = PointTransformerV3(
            in_channels=6, 
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_channels=self.config.ptv3_enc_channels,
            enc_depths=self.config.ptv3_enc_depths,
            enc_num_head=self.config.ptv3_enc_num_head,
            enc_patch_size=[self.config.ptv3_patch_size] * len(self.config.ptv3_enc_depths),
            enc_mode=self.config.ptv3_enc_mode,
            mlp_ratio=4,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.1,
            drop_path=0,
            pre_norm=True,
            shuffle_orders=True,
            enable_flash=True,
        )

        self.point_multiscale_feature = False
        # self.point_multiscale_feature = True #False
        if self.point_multiscale_feature:
            point_proj = []
            for enc_dim in self.config.ptv3_enc_channels:
                point_proj.append(
                    nn.Linear(enc_dim, self.config.text_config.hidden_size)
                )
            self.point_proj = nn.ModuleList(point_proj)
        else:
            self.point_proj = nn.Linear(
                self.config.ptv3_enc_channels[-1], 
                self.config.text_config.hidden_size,
            )

        self.ptv3_model = self.ptv3_model.to(dtype=torch.float32)
        self.point_proj = self.point_proj.to(dtype=torch.float32)

    def encode_point_cloud(self, pc_fts, npoints_in_batch):
        device = pc_fts.device

        point_offset = torch.cumsum(npoints_in_batch, dim=0).long()
        point_batch_idxs = torch.arange(
            len(npoints_in_batch), device=device, dtype=torch.long
        ).repeat_interleave(npoints_in_batch)
        ptv3_batch = {
            'coord': pc_fts[:, :3],
            'grid_size': self.config.voxel_size,
            'offset': point_offset,
            'batch': point_batch_idxs,
            'feat': pc_fts,
        }

        point_outs = self.ptv3_model(ptv3_batch, return_layer_outputs=self.point_multiscale_feature)

        if self.point_multiscale_feature:
            max_K = 64
            def random_select_point_feature(point_fts, K=128):
                # [N, D] -> [K, D]
                N = point_fts.size(0)
                if N > K:
                    indices = torch.randperm(N)[:K]
                    return point_fts[indices]
                return point_fts

            point_embeds, new_npoints_in_batch = None, None
            for l, layer_output in enumerate(point_outs):
                layer_point_embeds = self.point_proj[l](layer_output.feat)
                layer_new_npoints_in_batch = torch.diff(
                    torch.cat([torch.zeros(1, dtype=torch.long, device=device), layer_output.offset])
                )
                layer_point_embeds = torch.split(
                    layer_point_embeds, layer_new_npoints_in_batch.data.cpu().numpy().tolist()
                )
                if point_embeds is None:
                    point_embeds = [random_select_point_feature(x, K=max_K) for x in layer_point_embeds]
                    new_npoints_in_batch = layer_new_npoints_in_batch.clamp(max=max_K)
                else:
                    for b in range(len(layer_point_embeds)):
                        point_embeds[b] = torch.cat(
                            [point_embeds[b], 
                             random_select_point_feature(layer_point_embeds[b], K=max_K)
                             ], dim=0)
                        new_npoints_in_batch[b] = new_npoints_in_batch[b] + min(layer_new_npoints_in_batch[b], max_K)
            # print(new_npoints_in_batch)
        else:
            point_embeds = self.point_proj(point_outs.feat)
            new_npoints_in_batch = torch.diff(
                torch.cat([torch.zeros(1, dtype=torch.long, device=device), point_outs.offset])
            )
            point_embeds = torch.split(
                point_embeds, new_npoints_in_batch.data.cpu().numpy().tolist()
            )
            
        return point_embeds, new_npoints_in_batch
    
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
        **kwargs,
    ) -> VLADualOutputWithPast:
        """multi-modal forward pass, including image, video, state, action, and language."""
        device = input_ids.device

        inputs_embeds = self.embed_prefix(
            input_ids,
            inputs_embeds,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

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
                points=points,
                npoints_in_batch=npoints_in_batch,
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
        text_hidden_states = hidden_states

        # Encode point cloud
        point_embeds, new_npoints_in_batch = self.encode_point_cloud(
            points, npoints_in_batch
        )
        # concate point clouds into image embeds
        new_hidden_states, new_attention_masks = [], []
        for i in range(len(hidden_states)):
            new_hidden_states.append(
                torch.cat([point_embeds[i].to(hidden_states.dtype), hidden_states[i]], 0)
            )
            tmp_mask = torch.ones(new_npoints_in_batch[i], dtype=attention_mask.dtype, device=device)
            new_attention_masks.append(
                torch.cat([tmp_mask, attention_mask[i]], 0)
            )
        hidden_states = pad_sequence(
            new_hidden_states, batch_first=True, padding_value=0
        )
        attention_mask = pad_sequence(
            new_attention_masks, batch_first=True, padding_value=0
        ).bool()

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
            logits = self.vlm_backbone.lm_head(text_hidden_states[:, slice_indices, :])

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
        npoints_in_batch: torch.Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Sample actions from the model."""
        self._sync_action_head_num_inference_timesteps()

        device = input_ids.device

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

        # Encode point cloud
        point_embeds, new_npoints_in_batch = self.encode_point_cloud(
            points, npoints_in_batch
        )
        # concate point clouds into image embeds
        new_hidden_states, new_attention_masks = [], []
        for i in range(len(hidden_state)):
            new_hidden_states.append(
                torch.cat([point_embeds[i].to(hidden_state.dtype), hidden_state[i]], 0)
            )
            tmp_mask = torch.ones(new_npoints_in_batch[i], dtype=attention_mask.dtype, device=device)
            new_attention_masks.append(
                torch.cat([tmp_mask, attention_mask[i]], 0)
            )
        hidden_state = pad_sequence(
            new_hidden_states, batch_first=True, padding_value=0
        )
        attention_mask = pad_sequence(
            new_attention_masks, batch_first=True, padding_value=0
        )

        input_dtype = next(self.action_head.parameters()).dtype
        backbone_outputs = {
            "backbone_features": hidden_state.to(dtype=input_dtype),
            "backbone_attention_mask": attention_mask,
        }
        backbone_outputs = self.action_head.prepare_input(backbone_outputs)
        action_inputs = {
            "embodiment_id": torch.zeros(states.size(0), dtype=torch.long, device=states.device),
            "state": states.to(dtype=input_dtype),
        }
        action_inputs = self.action_head.prepare_input(action_inputs)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        x_t = action_head_outputs.action_pred
            
        return x_t, outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.vlm_backbone.prepare_inputs_for_generation(*args, **kwargs)

    def _expand_inputs_for_generation(self, *args, **kwargs):
        return self.vlm_backbone._expand_inputs_for_generation(*args, **kwargs)


# VLADualPointFlowMatchingModel.register_for_auto_class()
