import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from pointact.constants import IGNORE_INDEX
from pointact.model.utils import create_mm_token_type_ids
from pointact.model.backbone.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from .configuration_eo1 import EO1VisionPointFlowMatchingConfig
from .modeling_eo1 import EO1VisionFlowMatchingModel, EO1VisionFlowMatchingOutputWithPast

from pointact.model.ptv3.concerto.model import PointTransformerV3



class EO1VisionPointFlowMatchingModel(EO1VisionFlowMatchingModel):
    config_class = EO1VisionPointFlowMatchingConfig

    def __init__(
        self,
        config: EO1VisionPointFlowMatchingConfig,
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
        
        point_outs = self.ptv3_model(ptv3_batch)

        point_embeds = self.point_proj(point_outs.feat)
        new_npoints_in_batch = torch.diff(
            torch.cat([torch.zeros(1, dtype=torch.long, device=device), point_outs.offset])
        )

        point_embeds = torch.split(
            point_embeds, new_npoints_in_batch.data.cpu().numpy().tolist()
        )
            
        return point_embeds, new_npoints_in_batch

    def embed_prefix(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
        points: torch.Tensor | None = None,
        npoints_in_batch: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
    ) -> tuple[torch.LongTensor, torch.FloatTensor, torch.Tensor, torch.LongTensor | None]:
        """Embed the prefix"""
        inputs_embeds = super().embed_prefix(
            input_ids, 
            inputs_embeds, 
            pixel_values, 
            pixel_values_videos,
            image_grid_thw, 
            video_grid_thw, 
            states,
        )

        # Encode point cloud
        point_embeds, new_npoints_in_batch = self.encode_point_cloud(
            points, npoints_in_batch
        )

        # Insert points
        pad_token_id = self.config.pad_token_id # 151643
        point_token_id = self.config.point_token_id
        batch_size = input_ids.size(0)
        input_lens = (input_ids != pad_token_id).sum(dim=1)
        device = input_ids.device

        new_input_ids, new_inputs_embeds, new_labels = [], [], []
        for i in range(batch_size):
            i_point_len = new_npoints_in_batch[i].item()
            point_pos = (input_ids[i] == point_token_id).nonzero(as_tuple=True)[0]
            if point_pos.numel() != 1:
                raise ValueError(f"Expected exactly one point token, got {point_pos.numel()}")
            insert_idx = point_pos.item()
            i_point_tokens = torch.full((i_point_len,), point_token_id, dtype=input_ids.dtype, device=device)
            i_input_ids = torch.cat(
                [input_ids[i, :insert_idx], i_point_tokens, input_ids[i, insert_idx+1:input_lens[i]]]
            )
            i_inputs_embeds = torch.cat(
                [inputs_embeds[i, :insert_idx], point_embeds[i].to(dtype=inputs_embeds.dtype), inputs_embeds[i, insert_idx+1:input_lens[i]]]
            )
            new_input_ids.append(i_input_ids)
            new_inputs_embeds.append(i_inputs_embeds)
            if labels is not None:
                i_point_labels = labels[i, insert_idx:insert_idx+1].expand(i_point_len)
                i_labels = torch.cat(
                    [labels[i, :insert_idx], i_point_labels, labels[i, insert_idx+1:input_lens[i]]]
                )
                new_labels.append(i_labels)

        new_input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=pad_token_id)
        new_inputs_embeds = pad_sequence(new_inputs_embeds, batch_first=True, padding_value=0)
        labels = pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX) if labels is not None else None

        return new_input_ids, new_inputs_embeds, new_npoints_in_batch, labels

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
    ) -> EO1VisionFlowMatchingOutputWithPast:
        """multi-modal forward pass, including image, video, state, action, and language."""

        input_ids, inputs_embeds, new_npoints_in_batch, labels = self.embed_prefix(
            input_ids,
            inputs_embeds,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            states,
            points,
            npoints_in_batch,
            labels,
        )
        attention_mask = input_ids.ne(self.config.pad_token_id)
        
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
                points=points,
                npoints_in_batch=npoints_in_batch,
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
            action_loss = (losses * valid_loss_mask).sum() / valid_loss_mask.expand_as(losses).sum().clamp_min(1)
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
        points: torch.Tensor | None = None,
        npoints_in_batch: torch.Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Sample actions from the model."""
        if input_ids.shape[0] != 1:
            raise ValueError("EO1VisionPointFlowMatchingModel.sample_actions only supports batch_size=1.")

        # embed prefix
        if inputs_embeds is None:
            input_ids, inputs_embeds, new_npoints_in_batch, _ = self.embed_prefix(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                states=states,
                points=points,
                npoints_in_batch=npoints_in_batch,
            )
            
        attention_mask = input_ids.ne(self.config.pad_token_id)

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

# EO1VisionPointFlowMatchingModel.register_for_auto_class()
