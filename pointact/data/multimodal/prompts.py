import copy
import random
import re

import torch
from lerobot.constants import ACTION, OBS_STATE

from pointact.constants import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_STATE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_ACTION_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_STATE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    LLAVA_VLA_TOKEN,
    PASS_ACTION_TOKEN,
    STATE_END_TOKEN,
    STATE_START_TOKEN,
    TASK_VLA_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from pointact.utils.torch_utils import pad_vector


def build_interleaved_prompt(
    items: list[dict],
    sources: dict = None,
    max_action_dim: int = 32,
    max_state_dim: int = 64,
    chunk_size: int = 50,
    sample_actions: bool = False,
):
    """Construct an OpenAI-style multimodal conversation with LeRobot metadata."""
    view = sources.get("view")
    conversation_len = len(sources["conversations"]) // 2

    truncate_ids = [
        i for i in range(conversation_len) if LLAVA_VLA_TOKEN in sources["conversations"][i * 2]["value"]
    ]
    denoise_idx = random.choice(truncate_ids + [conversation_len])

    sources = {
        "conversations": copy.deepcopy(sources["conversations"]) if sources else [],
        "action": [],
        "state": [],
        "image": [],
        "action_is_pad": [],
    }

    idx = 0
    for i in range(conversation_len):
        human_conversation = sources["conversations"][i * 2]["value"]
        conversation_image_n = human_conversation.count(LLAVA_IMAGE_TOKEN)

        le_image_n = 0
        while le_image_n < conversation_image_n:
            item = items[idx]
            actions, states = [], []
            images = [item[v] for v in view[idx]]

            for key, value in item.items():
                if key.startswith(ACTION) and "is_pad" not in key:
                    actions.append(value.unsqueeze(-1) if value.dim() == 1 else value)
                elif key.startswith(OBS_STATE):
                    states.append(value)
                elif key.startswith(ACTION) and "is_pad" in key:
                    action_is_pad = value

            states = pad_vector(torch.cat(states, dim=-1), max_state_dim)
            actions = pad_vector(torch.cat(actions, dim=-1), max_action_dim)
            action_is_pads = action_is_pad.clone()

            idx += 1
            le_image_n += len(images)
            sources["image"].extend(images)

        gpt_conversation = sources["conversations"][i * 2 + 1]["value"]
        if human_conversation.endswith(LLAVA_VLA_TOKEN) and gpt_conversation.endswith(LLAVA_ACTION_TOKEN):
            sources["action"].append(actions)
            sources["state"].append(states)
            sources["action_is_pad"].append(action_is_pads)
            sources["conversations"][i * 2]["value"] = human_conversation.replace(
                LLAVA_VLA_TOKEN, TASK_VLA_TOKEN
            )

            if sample_actions:
                if i < denoise_idx:
                    replacement = f"{ACTION_START_TOKEN}{PASS_ACTION_TOKEN * chunk_size}{ACTION_END_TOKEN}"
                    sources["conversations"][i * 2 + 1]["value"] = gpt_conversation.replace(
                        LLAVA_ACTION_TOKEN, replacement
                    )
                elif i == denoise_idx:
                    sources["conversations"] = sources["conversations"][: (i + 1) * 2]
                    return sources
    return sources


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r"\s*" + re.escape(LLAVA_VIDEO_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r"\s*" + re.escape(LLAVA_IMAGE_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN
    return re.sub(pattern, replacement, input_string)


def replace_action_tokens(input_string):
    pattern = r"\s*" + re.escape(LLAVA_ACTION_TOKEN) + r"\n?"
    replacement = f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN}{ACTION_END_TOKEN}"
    return re.sub(pattern, replacement, input_string)


def replace_state_tokens(input_string):
    pattern = r"\s*" + re.escape(LLAVA_STATE_TOKEN) + r"\n?"
    replacement = f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}"
    return re.sub(pattern, replacement, input_string)


def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}
    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_content = replace_action_tokens(transformed_content)
        transformed_content = replace_state_tokens(transformed_content)
        transformed_data.append(
            {
                "role": role_mapping.get(conversation["from"], conversation["from"]),
                "content": transformed_content,
            }
        )
    return transformed_data
