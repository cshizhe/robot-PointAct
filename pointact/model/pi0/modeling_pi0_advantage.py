#!/usr/bin/env python

import torch
from torch import Tensor

from .modeling_pi0 import PI0Model, make_att_2d_masks


class PI0AdvantageModel(PI0Model):
    """PI0 with classifier-free guidance for binary advantage conditioning."""

    def _compute_prefix_cache(
        self,
        text_ids: torch.LongTensor,
        text_masks: torch.Tensor,
        images: torch.Tensor,
        img_masks: torch.Tensor,
    ):
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, text_ids, text_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_pad_masks, past_key_values

    @torch.no_grad()
    def sample_actions(
        self,
        text_ids: torch.LongTensor | None = None,
        text_masks: torch.Tensor | None = None,
        images: torch.Tensor | None = None,
        img_masks: torch.Tensor | None = None,
        states: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        uncond_text_ids: torch.LongTensor | None = None,
        uncond_text_masks: torch.Tensor | None = None,
        advantage_cfg_scale: float | None = None,
        **kwargs,
    ) -> Tensor:
        """Sample actions, optionally applying advantage classifier-free guidance."""
        if advantage_cfg_scale is None:
            advantage_cfg_scale = self.config.advantage_cfg_scale
        advantage_cfg_scale = float(advantage_cfg_scale)

        if advantage_cfg_scale == 1.0:
            return super().sample_actions(
                text_ids=text_ids,
                text_masks=text_masks,
                images=images,
                img_masks=img_masks,
                states=states,
                noise=noise,
                num_steps=num_steps,
                **kwargs,
            )

        if uncond_text_ids is None or uncond_text_masks is None:
            raise ValueError(
                "uncond_text_ids and uncond_text_masks are required when "
                "advantage_cfg_scale is not 1.0"
            )

        if num_steps is None:
            num_steps = self.config.num_denoise_steps

        bsize = states.shape[0]
        device = states.device

        if noise is None:
            actions_shape = (
                bsize,
                self.config.action_chunk_size,
                self.config.max_action_dim,
            )
            noise = self.sample_noise(actions_shape, device)

        prefix_pad_masks, past_key_values = self._compute_prefix_cache(
            text_ids=text_ids,
            text_masks=text_masks,
            images=images,
            img_masks=img_masks,
        )
        uncond_prefix_pad_masks, uncond_past_key_values = self._compute_prefix_cache(
            text_ids=uncond_text_ids,
            text_masks=uncond_text_masks,
            images=images,
            img_masks=img_masks,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                v_cond = self.denoise_step(
                    state=states,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )
                v_uncond = self.denoise_step(
                    state=states,
                    prefix_pad_masks=uncond_prefix_pad_masks,
                    past_key_values=uncond_past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )
                return v_uncond + advantage_cfg_scale * (v_cond - v_uncond)

            v_t = denoise_step_partial_call(x_t)
            x_t = x_t + dt * v_t

        return x_t
