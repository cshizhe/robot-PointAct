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

import os
from typing import Union
from easydict import EasyDict

import numpy as np
import torch
from lerobot.constants import OBS_STATE
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs

from pointact.constants import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_POINT_TOKEN,
    DEFAULT_STATE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    PASS_ACTION_TOKEN,
    TASK_VLA_TOKEN,
)
from pointact.utils.rotation import convert_rotation
from pointact.utils.torch_utils import pad_vector
from pointact.model.backbone.processor_base import (
    PreparedRobotBatch, RobotProcessorBase, RobotPointProcessorBase
)

os.environ["TOKENIZERS_PARALLELISM"] = "0"


RobotInput = Union[np.ndarray, "torch.Tensor", list[np.ndarray], list["torch.Tensor"]]



class EO1VisionProcessor(RobotProcessorBase):
    """EO1Vision Processor for Image, Text, Video, and Robotic Action Processing"""

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

        text_inputs, image_inputs, videos_inputs = self._prepare_image_video_action_inputs(
            images, videos, text, output_kwargs
        )

        robot_inputs = {}

        if states is not None:
            if isinstance(states, list):
                states = torch.stack(states, dim=0)
            if states.ndim == 1:
                states = states.unsqueeze(0)
            robot_inputs.update({"states": states})

        if actions is not None:
            if isinstance(actions, list):
                actions = torch.stack(actions, dim=0)
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            robot_inputs.update({"actions": actions})

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return BatchFeature(
            data={**text_inputs, **image_inputs, **videos_inputs, **robot_inputs},
            tensor_type=return_tensors,
        )

    def _build_robot_message(
        self,
        mini_batch: dict,
        repo_id: str,
        select_video_keys: list[str],
        select_state_keys: list[str],
        include_point_token: bool = False,
    ) -> tuple[list[dict], torch.Tensor | None]:
        content = [
            *({"type": "image", "image": mini_batch[k]} for k in select_video_keys),
        ]

        if include_point_token:
            content.append({"type": "point", "point": []})

        state = None
        if len(select_state_keys) > 0:
            state = np.concatenate([mini_batch[k] for k in select_state_keys], axis=-1)
            state = self._normalize_robot_state(state, repo_id)
            state = torch.from_numpy(state).float()
            state = pad_vector(state, self.robot_config["max_state_dim"])
            content.append({"type": "state", "state": []})

        content.append({"type": "text", "text": f"{mini_batch['task']}{TASK_VLA_TOKEN}"})
        return [{"role": "user", "content": content}], state

    def _prepare_robot_batch(self, batch: dict, include_point_token: bool = False) -> PreparedRobotBatch:
        state_keys = [x for x in batch.keys() if x.startswith(OBS_STATE)]
        batch_size = len(batch[state_keys[0]])

        repo_ids = self._resolve_repo_ids(batch, batch_size)

        batch_messages = []
        batch_states = []

        for i, repo_id in enumerate(repo_ids):
            mini_batch = {k: v[i] for k, v in batch.items()}
            select_video_keys = self.robot_config["select_video_keys"][repo_id]
            select_state_keys = self.robot_config["select_state_keys"][repo_id]
            messages, state = self._build_robot_message(
                mini_batch,
                repo_id,
                select_video_keys,
                select_state_keys,
                include_point_token=include_point_token,
            )

            if state is not None:
                batch_states.append(state)
            batch_messages.append(messages)

        return PreparedRobotBatch(
            messages=batch_messages,
            states=batch_states if state is not None else None,
            repo_ids=repo_ids,
        )        

    def _build_action_prompt_inputs(
        self,
        model,
        prepared_batch: PreparedRobotBatch,
    ) -> BatchFeature:
        noise_prompt = f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN}{ACTION_END_TOKEN}"
        return self.apply_chat_template(
            prepared_batch.messages,
            add_generation_prompt=True,
            noise_prompt=noise_prompt,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"states": prepared_batch.states}
        ).to(model.device)          

    def _build_action_output(
        self,
        repo_ids: list[str],
        actions: torch.Tensor,
        pred_rot_type: str,
    ) -> EasyDict:
        """Process model outputs back to robot format"""
        output_actions = []
        # unnormalize actions per dataset
        for i, repo_id in enumerate(repo_ids):
            select_action_keys = self.robot_config["select_action_keys"][repo_id]
            action_dim = sum([self.robot_config["features"][repo_id][key]["shape"][0] for key in select_action_keys])
            unnorm_actions = self._unnormalize_robot_action(
                actions[i, :, :action_dim].data.numpy(), repo_id
            )
            output_actions.append(unnorm_actions)
        output_actions = np.stack(output_actions, axis=0)
    
        # convert rotation if needed
        if pred_rot_type == "euler":
            quat = convert_rotation(
                output_actions[..., 3:6], "euler", "quat", euler_order_src="xyz", quat_order_dst="xyzw"
            )
            output_actions = np.concatenate([output_actions[..., :3], quat, output_actions[..., 6:]], -1)
        if pred_rot_type == "rot6d":
            quat = convert_rotation(
                output_actions[..., 3:9], "rot6d", "quat", quat_order_dst="xyzw"
            )
            output_actions = np.concatenate([output_actions[..., :3], quat, output_actions[..., 9:]], -1)

        return EasyDict({"action": output_actions})

    @torch.no_grad()
    def select_action(self, model, batch: dict, pred_rot_type: str, **kwargs):
        prepared_batch = self._prepare_robot_batch(batch)
    
        inputs = self._build_action_prompt_inputs(model, prepared_batch)
        
        actions, _ = model.sample_actions(**inputs)
        actions = actions.cpu()
    
        return self._build_action_output(prepared_batch.repo_ids, actions, pred_rot_type)


class EO1VisionPointProcessor(EO1VisionProcessor, RobotPointProcessorBase):

    def _prepare_point_clouds(
        self,
        batch: dict,
        points_workspace: dict | None = None,
        remove_arm: bool = False,
    ) -> list[torch.Tensor]:
        
        state_keys = [x for x in batch.keys() if x.startswith(OBS_STATE)]
        batch_size = len(batch[state_keys[0]])
        if batch_size != 1:
            raise ValueError("EO1VisionPointProcessor.select_action only supports batch_size=1.")
        repo_ids = self._resolve_repo_ids(batch, batch_size)
        
        batch_points, batch_point_centers = [], []
        for i, repo_id in enumerate(repo_ids):
            mini_batch = {k: v[i] for k, v in batch.items()}
            workspace = self._resolve_points_workspace(repo_id, points_workspace)
            point_cloud = self._prepare_point_cloud_for_sample(
                mini_batch,
                repo_id,
                workspace,
                remove_arm=remove_arm,
            )
            point_cloud = torch.from_numpy(point_cloud).float()
            state = mini_batch[OBS_STATE]
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(np.asarray(state))
            # update the state with the point cloud center
            point_cloud, state, point_center = self._center_point_cloud_and_state(
                point_cloud,
                state.float(),
            )
            batch_points.append(point_cloud)
            batch_point_centers.append(point_center)
            batch[OBS_STATE][i] = state

        return batch_points, batch_point_centers, batch
    
    @torch.no_grad()
    def select_action(self, model, batch: dict, pred_rot_type: str, **kwargs):
        points_workspace = kwargs.get("points_workspace", None)
        remove_arm = kwargs.get("remove_arm", False)
        batch_points, batch_point_centers, batch = self._prepare_point_clouds(
            batch,
            points_workspace=points_workspace,
            remove_arm=remove_arm,
        )

        prepared_batch = self._prepare_robot_batch(batch, include_point_token=True)
    
        inputs = self._build_action_prompt_inputs(model, prepared_batch)
        inputs["points"] = torch.cat(batch_points, 0).to(model.device)
        inputs["npoints_in_batch"] = torch.LongTensor([len(points) for points in batch_points]).to(model.device)
        
        actions, _ = model.sample_actions(**inputs)
        actions = actions.cpu()
    
        output_actions = self._build_action_output(prepared_batch.repo_ids, actions, pred_rot_type)

        for i, repo_id in enumerate(prepared_batch.repo_ids):
            if not self.robot_config["is_delta_action"][repo_id]:
                output_actions.action[i][..., :3] += batch_point_centers[i].data.numpy()

        return output_actions


# EO1VisionProcessor.register_for_auto_class()
# EO1VisionPointProcessor.register_for_auto_class()
