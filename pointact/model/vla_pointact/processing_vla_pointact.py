import os
from typing import Union

import numpy as np
import torch
from easydict import EasyDict
from lerobot.constants import OBS_STATE
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

from pointact.constants import DEFAULT_STATE_TOKEN, STATE_END_TOKEN, STATE_START_TOKEN
from pointact.model.backbone.processor_base import RobotPointProcessorBase
from pointact.utils.rotation import convert_rotation
from pointact.utils.torch_utils import pad_vector

RobotInput = Union[np.ndarray, "torch.Tensor", list[np.ndarray], list["torch.Tensor"]]

os.environ["TOKENIZERS_PARALLELISM"] = "0"


class VLAEncDec3DProcessor(RobotPointProcessorBase):
    """Processor for Image, Text, Video, PointCloud and Robotic Action Processing"""

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        states: RobotInput = None,
        actions: RobotInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            return_mm_token_type_ids=False,
            **kwargs,
        )

        text = self._remove_state_tokens(text)
        text_inputs, image_inputs, videos_inputs = self._prepare_image_video_action_inputs(
            images, videos, text, output_kwargs
        )
        robot_inputs = self._prepare_robot_tensor_inputs(states=states, actions=actions)

        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs, **robot_inputs},
        )

    @staticmethod
    def _remove_state_tokens(text):
        if not isinstance(text, list):
            text = [text]
        text = text.copy()
        for i in range(len(text)):
            for state_token in [STATE_START_TOKEN, STATE_END_TOKEN, DEFAULT_STATE_TOKEN]:
                text[i] = text[i].replace(state_token, "")
        return text

    @staticmethod
    def _as_batched_tensor(value):
        if value is None:
            return None
        if isinstance(value, list):
            value = torch.stack(value, dim=0)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        return value

    def _prepare_robot_tensor_inputs(self, states=None, actions=None):
        robot_inputs = {}
        states = self._as_batched_tensor(states)
        actions = self._as_batched_tensor(actions)
        if states is not None:
            robot_inputs["states"] = states
        if actions is not None:
            robot_inputs["actions"] = actions
        return robot_inputs

    @torch.no_grad
    def _prepare_robot_inputs(self, batch: dict, points_workspace: dict=None, remove_arm: bool=False):
        """Prepare model inputs from raw robot batch"""
        batch_messages = []
        batch_states = []

        state_keys = [x for x in batch.keys() if x.startswith(OBS_STATE)]
        batch_size = len(batch[state_keys[0]])
        repo_ids = self._resolve_repo_ids(batch, batch_size)

        batch_points, batch_point_centers = [], []
        for i, repo_id in enumerate(repo_ids):
            mini_batch = {k: v[i] for k, v in batch.items()}

            select_video_keys = self.robot_config["select_video_keys_for_vlm"][repo_id]
            select_state_keys = self.robot_config["select_state_keys"][repo_id]

            messages = [
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "image": mini_batch[k]} for k in select_video_keys),
                    ],
                }
            ]
            messages[0]["content"].append(
                {"type": "text", "text": f"{mini_batch['task']}"},
            )

            state = None
            if len(select_state_keys) > 0:
                state_parts = []
                for key in select_state_keys:
                    value = mini_batch[key]
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    state_parts.append(np.asarray(value))
                state = torch.as_tensor(np.concatenate(state_parts, axis=-1), dtype=torch.float32)

            workspace = self._resolve_points_workspace(repo_id, points_workspace)
            point_cloud = self._prepare_point_cloud_for_sample(
                mini_batch,
                repo_id,
                workspace,
                remove_arm=remove_arm,
            )
            point_cloud = torch.from_numpy(point_cloud).float()
            point_cloud, state, point_center = self._center_point_cloud_and_state(
                point_cloud,
                state,
                center_state=self._repo_config_flag("is_action_eef", repo_id, default=True),
            )
            if state is not None:
                state = self._normalize_robot_state(state.numpy(), repo_id)
                state = torch.as_tensor(state, dtype=torch.float32)
                batch_states.append(pad_vector(state, self.robot_config["max_state_dim"]))
            batch_messages.append(messages)
            batch_point_centers.append(point_center.numpy())
            batch_points.append(point_cloud)
            
        return batch_messages, batch_states or None, batch_points, batch_point_centers, repo_ids

    def _action_dim(self, repo_id: str) -> int:
        select_action_keys = self.robot_config["select_action_keys"][repo_id]
        return sum(self.robot_config["features"][repo_id][key]["shape"][0] for key in select_action_keys)

    def _process_robot_outputs(self, repo_ids: list[str], actions: torch.Tensor):
        """Slice padded model actions back to each robot's configured action dimension."""
        output_actions = []
        for i, repo_id in enumerate(repo_ids):
            output_actions.append(actions[i].detach().cpu().float()[..., : self._action_dim(repo_id)])
        return torch.stack(output_actions, dim=0)

    def _build_action_output(self, repo_ids: list[str], actions: torch.Tensor, pred_rot_type: str):
        output_actions = self._process_robot_outputs(repo_ids, actions).numpy()
        for i, repo_id in enumerate(repo_ids):
            output_actions[i] = self._unnormalize_robot_action(output_actions[i], repo_id)

        if pred_rot_type == "euler":
            quat = convert_rotation(
                output_actions[..., 3:6], "euler", "quat", euler_order_src="xyz", quat_order_dst="xyzw"
            )
            output_actions = np.concatenate([output_actions[..., :3], quat, output_actions[..., 6:]], -1)
        elif pred_rot_type == "rot6d":
            quat = convert_rotation(
                output_actions[..., 3:9], "rot6d", "quat", quat_order_dst="xyzw"
            )
            output_actions = np.concatenate([output_actions[..., :3], quat, output_actions[..., 9:]], -1)

        return EasyDict({"action": output_actions})

    @torch.no_grad
    def select_action(
        self, model, batch: dict, pred_rot_type: str, use_cot=False, 
        points_workspace: dict=None, remove_arm: bool=False, **kwargs
    ):
        batch_messages, batch_states, batch_points, batch_point_centers, repo_ids = self._prepare_robot_inputs(
            batch, points_workspace=points_workspace, remove_arm=remove_arm
        )
        device = model.device

        inputs = self.apply_chat_template(
            batch_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"states": batch_states},
        ).to(device)
        # print(inputs['input_ids'])

        inputs["input_id_lens"] = inputs["attention_mask"].sum(dim=1).long().to(device)
        inputs["points"] = torch.cat(batch_points, 0).to(device)
        inputs["npoints_in_batch"] = torch.LongTensor([len(x) for x in batch_points]).to(device)
        inputs["attention_mask"] = inputs["attention_mask"].bool().to(device)

        actions, _ = model.sample_actions(
            **inputs, 
        )
        outs = self._build_action_output(repo_ids, actions.cpu(), pred_rot_type)
        for i in range(len(outs.action)):
            repo_id = repo_ids[i]
            use_delta_action = self._repo_config_flag("is_delta_action", repo_id, default=False)
            if not use_delta_action:
                outs.action[i, :, :3] += batch_point_centers[i][None, :]
        return outs



# VLAEncDec3DProcessor.register_for_auto_class()
