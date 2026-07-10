#!/usr/bin/env python

import numpy as np
import torch
from easydict import EasyDict
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from torch.nn.utils.rnn import pad_sequence

from pointact.utils.rotation import convert_rotation

from .processor_pi0 import PI0Processor


class PI0AdvantageProcessor(PI0Processor):
    """PI0 processor with binary advantage text conditioning."""

    def __init__(
        self,
        tokenizer=None,
        image_resolution: tuple[int, int] = (224, 224),
        robot_state_action_norm_file: str | None = None,
        state_action_norm: dict | None = None,
        robot_config: dict | None = None,
        tokenizer_name: str = "google/paligemma-3b-pt-224",
        chat_template: str | None = None,
        advantage_condition_dropout: float = 0.3,
        advantage_cfg_scale: float = 1.0,
        advantage_positive_text: str = "Advantage: positive",
        advantage_negative_text: str = "Advantage: negative",
    ):
        super().__init__(
            tokenizer=tokenizer,
            image_resolution=image_resolution,
            robot_state_action_norm_file=robot_state_action_norm_file,
            state_action_norm=state_action_norm,
            robot_config=robot_config,
            tokenizer_name=tokenizer_name,
            chat_template=chat_template,
        )
        self.advantage_condition_dropout = advantage_condition_dropout
        self.advantage_cfg_scale = advantage_cfg_scale
        self.advantage_positive_text = advantage_positive_text
        self.advantage_negative_text = advantage_negative_text

    @staticmethod
    def _coerce_advantage_label(advantage) -> int:
        if isinstance(advantage, torch.Tensor):
            advantage = advantage.flatten()[0].item()
        elif isinstance(advantage, np.ndarray):
            advantage = advantage.reshape(-1)[0].item()
        elif isinstance(advantage, (list, tuple)):
            advantage = advantage[0]

        if isinstance(advantage, str):
            normalized = advantage.strip().lower()
            if normalized in {"1", "true", "positive", "pos", "success", "successful"}:
                return 1
            if normalized in {"0", "false", "negative", "neg", "failure", "failed"}:
                return 0
            raise ValueError(f"Invalid advantage label string: {advantage!r}")

        return int(bool(advantage))

    def _advantage_text(self, advantage) -> str:
        label = self._coerce_advantage_label(advantage)
        return self.advantage_positive_text if label == 1 else self.advantage_negative_text

    def prepare_text_state(
        self,
        task: str,
        state: torch.Tensor | None = None,
        advantage=None,
        drop_advantage: bool = False,
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.Tensor | None]:
        """Build the PI0 prompt, optionally appending a binary advantage condition."""
        full_prompt = task.strip()
        if advantage is not None and not drop_advantage:
            full_prompt = f"{full_prompt}\n{self._advantage_text(advantage)}"
        if not full_prompt.endswith("\n"):
            full_prompt = f"{full_prompt}\n"

        text_inputs = self.tokenizer(full_prompt, add_special_tokens=False, return_tensors="pt")
        text_ids = text_inputs["input_ids"][0]
        text_masks = text_inputs["attention_mask"][0].bool()

        if state is not None:
            state = self.normalize_robot_state(state)
        return text_ids, text_masks, state

    @torch.no_grad()
    def select_action(self, model, batch: dict, pred_rot_type: str | None = None, **kwargs):
        """Run PI0 advantage inference.

        By default this samples with positive advantage conditioning. When
        advantage_cfg_scale differs from 1.0, an unconditional prompt is also
        built so the model can apply classifier-free guidance in action space.
        """
        device = next(model.parameters()).device

        advantage_label = kwargs.pop("advantage_label", kwargs.pop("advantage", 1))
        advantage_cfg_scale = kwargs.pop("advantage_cfg_scale", None)
        if advantage_cfg_scale is None:
            advantage_cfg_scale = getattr(model.config, "advantage_cfg_scale", self.advantage_cfg_scale)

        repo_id = batch.get("repo_id", None)
        if repo_id is None or len(repo_id) == 0 or repo_id[0] is None:
            repo_id = list(self.robot_config["select_video_keys"].keys())[0]
        else:
            repo_id = repo_id[0]

        image_keys = self.robot_config["select_video_keys"][repo_id]
        images = [batch[key][0] for key in image_keys]
        images = np.stack([np.asarray(image) / 255.0 for image in images], 0)
        images = torch.from_numpy(images)
        images, img_masks = self.prepare_images(images)
        if images.ndim == 4:
            images = images.unsqueeze(0)
            img_masks = img_masks.unsqueeze(0)
        images = images.to(device)
        img_masks = img_masks.to(device)

        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        state_keys = self.robot_config["select_state_keys"][repo_id]
        states = self._extract_states(batch, state_keys)
        states = self.normalize_robot_state(states)

        text_ids = []
        text_masks = []
        uncond_text_ids = []
        uncond_text_masks = []
        use_cfg = float(advantage_cfg_scale) != 1.0
        for task in tasks:
            ids, masks, _ = self.prepare_text_state(
                task,
                state=None,
                advantage=advantage_label,
            )
            text_ids.append(ids)
            text_masks.append(masks)

            if use_cfg:
                ids, masks, _ = self.prepare_text_state(
                    task,
                    state=None,
                    advantage=None,
                    drop_advantage=True,
                )
                uncond_text_ids.append(ids)
                uncond_text_masks.append(masks)

        text_ids = pad_sequence(text_ids, batch_first=True, padding_value=0).to(device)
        text_masks = pad_sequence(text_masks, batch_first=True, padding_value=0).to(device)

        model_kwargs = dict(kwargs)
        model_kwargs["advantage_cfg_scale"] = float(advantage_cfg_scale)
        if use_cfg:
            model_kwargs["uncond_text_ids"] = pad_sequence(
                uncond_text_ids, batch_first=True, padding_value=0
            ).to(device)
            model_kwargs["uncond_text_masks"] = pad_sequence(
                uncond_text_masks, batch_first=True, padding_value=0
            ).to(device)

        if states.ndim == 1:
            states = states.unsqueeze(0)
        states = states.to(device)

        actions = model.sample_actions(
            text_ids=text_ids,
            text_masks=text_masks,
            images=images,
            img_masks=img_masks,
            states=states,
            **model_kwargs,
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
