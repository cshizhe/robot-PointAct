import os
from dataclasses import dataclass

import torch
import transformers
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from pointact.constants import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_POINT_TOKEN,
    DEFAULT_STATE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    POINT_END_TOKEN,
    POINT_START_TOKEN,
    STATE_END_TOKEN,
    STATE_START_TOKEN,
    SYSTEM_MESSAGE,
    TASK_VLA_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
    OBS_POINTS
)
from pointact.data.collators import DataCollator
from pointact.data.multimodal.vl_data import MultimodaDataset
from pointact.data.robot.multi_data import MultiLeRobotDataset
from pointact.data.schema import DataConfig, LerobotConfig
from pointact.data.transforms.image import ImageTransforms, ImageTransformsConfig
from pointact.utils.torch_utils import pad_vector
from pointact.train.pipeline_config import TrainPipelineConfig


def collect_robot_tensors(item: dict, max_action_dim: int, max_state_dim: int):
    images, actions, states = [], [], []
    action_is_pad = None
    points = item.get(OBS_POINTS)
    camera_names = []

    for key, value in item.items():
        if key.startswith(OBS_IMAGE):
            camera_names.append(key.split(".")[-1][: -len("_image")])
            images.append(value)
        elif key.startswith(ACTION) and "is_pad" not in key:
            actions.append(value.unsqueeze(-1) if value.dim() == 1 else value)
        elif key.startswith(OBS_STATE):
            states.append(value)
        elif key.startswith(ACTION) and "is_pad" in key:
            action_is_pad = value

    padded_states = None
    if len(states) > 0:
        padded_states = pad_vector(torch.cat(states, dim=-1), max_state_dim)

    padded_actions = pad_vector(torch.cat(actions, dim=-1), max_action_dim)
    action_is_pads = action_is_pad.clone()

    return images, padded_actions, padded_states, action_is_pads, points, camera_names


@dataclass
class MonolithicPrompt:
    """Prompt where the assistant response contains the action token."""

    ignore_action_tokens: bool = True

    def build_robot_source(self, item: dict, args: TrainPipelineConfig):
        images, actions, states, action_is_pads, points, camera_names = collect_robot_tensors(
            item, args.max_action_dim, args.max_state_dim
        )

        image_replacement = f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}" * len(images)
        if points is not None:
            image_replacement = f"{image_replacement}{POINT_START_TOKEN}{DEFAULT_POINT_TOKEN}{POINT_END_TOKEN}"
        if states is not None and args.use_robot_state:
            states_replacement = f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}"
        else:
            states_replacement = ""
        action_replacement = f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN}{ACTION_END_TOKEN}"

        sources = {
            "conversations": [
                {
                    "role": "user",
                    "content": f"{image_replacement}{states_replacement}{item['task']}{TASK_VLA_TOKEN}",
                },
                {
                    "role": "assistant",
                    "content": action_replacement,
                },
            ],
            "action": [actions],
            "image": images,
            "action_is_pad": [action_is_pads],
        }
        if states is not None and args.use_robot_state:
            sources["state"] = [states]
        if points is not None:
            sources["points"] = points
            sources["npoints_in_batch"] = points.size(0)
        return sources


@dataclass
class DualPrompt:
    """Prompt where robot samples provide input context and actions separately."""

    ignore_action_tokens: bool = False

    def build_robot_source(self, item: dict, args: TrainPipelineConfig):
        images, actions, states, action_is_pads, points, camera_names = collect_robot_tensors(
            item, args.max_action_dim, args.max_state_dim
        )

        image_replacement = f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}" * len(images)
        sources = {
            "conversations": [
                {
                    "role": "user",
                    "content": f"{image_replacement}{item['task']}",
                },
            ],
            "action": [actions],
            "action_is_pad": [action_is_pads],
        }
        if len(images) > 0:
            sources["image"] = images
        if states is not None and args.use_robot_state:
            sources["state"] = [states]
        if points is not None:
            sources["points"] = points
            sources["npoints_in_batch"] = len(points)
        return sources


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        args: TrainPipelineConfig,
        processor: transformers.ProcessorMixin,
        prompt,
    ):
        super().__init__()
        self.args = args
        self.processor = processor
        self.prompt = prompt

        data_configs = self.load_data_config(args)
        self.lerobot_dataset = self.build_lerobot_dataset(args, data_configs)
        self.mm_dataset = self.build_multimodal_dataset(args, data_configs, self.lerobot_dataset)

        self.fps = args.fps
        self.image_min_pixel = args.image_min_pixels
        self.image_max_pixel = args.image_max_pixels
        self.video_min_pixel = args.video_min_pixels
        self.video_max_pixel = args.video_max_pixels
        self.image_resized_w = args.image_resized_width
        self.image_resized_h = args.image_resized_height
        self.video_resized_w = args.video_resized_width
        self.video_resized_h = args.video_resized_height
        self.vision_base_paths = self.mm_dataset.vision_base_paths if self.mm_dataset else None

    @staticmethod
    def load_data_config(args: TrainPipelineConfig):
        if args.data_path.endswith(".yaml"):
            data_configs = DataConfig.from_yaml(args.data_path)
            if args.train_lerobot_only:
                data_configs.mm_datasets = []
        else:
            data_configs = DataConfig(
                lerobot_datasets=[LerobotConfig(repo_id=args.data_path)],
                mm_datasets=[],
            )
        return data_configs

    @staticmethod
    def build_lerobot_dataset(args: TrainPipelineConfig, data_configs: DataConfig):
        if len(data_configs.lerobot_datasets) == 0:
            return []

        if args.image_aug:
            image_transforms = ImageTransforms(ImageTransformsConfig(color_aug=args.color_aug))
        else:
            image_transforms = None

        return MultiLeRobotDataset(
            data_configs=data_configs.lerobot_datasets,
            image_transforms=image_transforms,
            video_backend=args.lerobot_data_video_backend,
            chunk_size=args.chunk_size,
            max_state_dim=args.max_state_dim,
        )

    @staticmethod
    def build_multimodal_dataset(
        args: TrainPipelineConfig,
        data_configs: DataConfig,
        lerobot_dataset,
    ):
        if len(data_configs.mm_datasets) == 0:
            return []

        # TODO: copied from EO1, not used yet
        return MultimodaDataset(
            data_configs=data_configs.mm_datasets,
            max_action_dim=args.max_action_dim,
            max_state_dim=args.max_state_dim,
            meta_dataset=lerobot_dataset,
            chunk_size=args.chunk_size,
        )

    @property
    def lengths(self):
        """group the lengths of the datasets, we set sample_actions to False \
            to avoid action sampling damaging the length of the dataset
            After the length is calculated, reset it back to True
        """
        if getattr(self, "cached_lengths", None):
            return self.cached_lengths
        return []

    def __len__(self):
        if self.args.train_mm_only:
            return len(self.mm_dataset)
        return len(self.mm_dataset) + len(self.lerobot_dataset)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        if i < len(self.mm_dataset):
            sources = self.mm_dataset[i]
        else:
            item = self.lerobot_dataset[i - len(self.mm_dataset)]
            sources = self.prompt.build_robot_source(item, self.args)

        images, videos, pixel_key, grid_key, video_kwargs = self.load_vision_inputs(sources)
        return self.tensorize_source(sources, images, videos, pixel_key, grid_key, video_kwargs)

    def load_vision_inputs(self, sources: dict):
        video_kwargs = {}
        if "image" in sources:
            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

            image_files = sources["image"]
            if isinstance(image_files, str):
                image_files = [image_files]
            images = []
            for image_file in image_files:
                if isinstance(image_file, str) and not image_file.startswith("http"):
                    image_folder = self.vision_base_paths[sources["vision_base_idx"]]
                    image_file = os.path.join(image_folder, image_file)
                elif isinstance(image_file, torch.Tensor):
                    image_file = Image.fromarray(
                        (image_file * 255).to(torch.uint8).permute(1, 2, 0).numpy()
                    )
                # This will resize the image to be patch_size * some factor
                images.append(
                    get_image_info(
                        image_file,
                        self.image_min_pixel,
                        self.image_max_pixel,
                        self.image_resized_w,
                        self.image_resized_h,
                    )
                )
            return images, videos, pixel_key, grid_key, video_kwargs

        if "video" in sources:
            images = None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"
            video_files = sources["video"]
            video_folder = self.vision_base_paths[sources["vision_base_idx"]]
            if isinstance(video_files, str):
                video_files = [video_files]
            videos = []
            for video_file in video_files:
                if isinstance(video_file, str) and not video_file.startswith("http"):
                    video_file = os.path.join(video_folder, video_file)
                video_input, video_kwargs = get_video_info(
                    video_file,
                    self.video_min_pixel,
                    self.video_max_pixel,
                    self.video_resized_w,
                    self.video_resized_h,
                    self.args.fps,
                )
                videos.append(video_input)
            return images, videos, pixel_key, grid_key, video_kwargs

        return None, None, None, None, video_kwargs

    def tensorize_source(self, sources, images, videos, pixel_key, grid_key, video_kwargs):
        actions = sources.get("action", [])
        states = sources.get("state", None)
        points = sources.get("points", None)
        action_is_pad = sources.get("action_is_pad")
        conversations = sources["conversations"]

        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_grid = []

        if len(SYSTEM_MESSAGE) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = self.processor.tokenizer(
                system_message, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))

        img_start = 0
        for j in range(0, len(conversations), 2):
            user_input = conversations[j]
            user_text = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n"
            if j + 1 < len(conversations):
                gpt_response = conversations[j + 1]
                user_text = f"{user_text}{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"
            else:
                gpt_response = None

            if DEFAULT_IMAGE_TOKEN in user_text:
                img_num = user_text.count(DEFAULT_IMAGE_TOKEN)
                inputs = self.processor(
                    text=[user_text],
                    images=images[img_start : img_start + img_num] if images else None,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                )
                prompt_input_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
                img_start += img_num
            elif DEFAULT_VIDEO_TOKEN in user_text:
                inputs = self.processor(
                    text=[user_text],
                    images=images,
                    videos=videos,
                    padding=False,
                    do_resize=False,
                    return_tensors="pt",
                    **video_kwargs,
                )
                all_second_grid.extend(inputs["second_per_grid_ts"])
                prompt_input_ids = inputs["input_ids"]
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])
            else:
                prompt_input_ids = self.processor.tokenizer(
                    user_text, add_special_tokens=False, padding=False, return_tensors="pt"
                )["input_ids"]

            if gpt_response is not None:
                response_text = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
                response_input_ids = self.processor(
                    text=[response_text], padding=False, return_tensors="pt"
                )["input_ids"]
                input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
                labels = torch.cat(
                    [
                        torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                        response_input_ids.squeeze(0),
                    ],
                    dim=0,
                )
                if self.prompt.ignore_action_tokens:
                    cached_action_ids = torch.tensor(
                        [self.processor.action_token_id, self.processor.action_pass_id]
                    )
                    action_mask = torch.isin(labels, cached_action_ids)
                    labels[action_mask] = IGNORE_INDEX
            else:
                input_ids = prompt_input_ids.squeeze(0)
                labels = torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0]))

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if not self.args.train_lerobot_only:
            data_dict["labels"] = labels

        if pixel_key and grid_key:
            data_dict[pixel_key] = torch.cat(all_pixel_values, dim=0)
            data_dict[grid_key] = torch.cat(all_image_grid_thw, dim=0)

        if len(all_second_grid) > 0:
            data_dict["second_per_grid_ts"] = all_second_grid

        if len(actions) > 0:
            data_dict["actions"] = torch.stack(actions, dim=0)
            data_dict["action_is_pad"] = torch.stack(action_is_pad, dim=0)

        if states is not None:
            data_dict["states"] = torch.stack(states, dim=0)

        if points is not None:
            data_dict["points"] = points
            data_dict["npoints_in_batch"] = sources.get("npoints_in_batch", len(points))

        return data_dict

    def info_qwen_vision_fetch(self):
        from qwen_vl_utils import smart_resize

        if not self.lerobot_dataset:
            return

        print(f"qwen2.5 vl min pixel {self.args.image_min_pixels}, max pixel {self.args.image_max_pixels}")
        for dataset in self.lerobot_dataset._datasets:
            meta_features = dataset.meta.features
            video_keys = getattr(dataset, "select_video_keys_for_vlm", dataset.select_video_keys)
            for key in video_keys:
                h, w = meta_features[key]["shape"][0], meta_features[key]["shape"][1]
                h_bar, w_bar = smart_resize(
                    h,
                    w,
                    14 * 2,
                    min_pixels=self.args.image_min_pixels,
                    max_pixels=self.args.image_max_pixels,
                )
                print(f"{dataset.repo_id:<40} | {key:<40} | resize from {h, w} to {h_bar, w_bar} |")


def get_image_info(image_path, min_pixel, max_pixel, width, height):
    content = {"type": "image", "image": image_path, "min_pixels": min_pixel, "max_pixels": max_pixel}
    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]
    image_input, _ = process_vision_info(messages)
    return image_input[0]


def get_video_info(video_path, min_pixels, max_pixels, width, height, fps):
    content = {
        "type": "video",
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "min_frames": 30,
        "max_frames": 60,
        "fps": fps,
    }
    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]
    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    return video_input[0], video_kwargs


def create_monolithic_prompt_data_module(processor, args: TrainPipelineConfig):
    dataset = SupervisedDataset(args=args, processor=processor, prompt=MonolithicPrompt())
    data_collator = DataCollator(pad_token_id=processor.tokenizer.pad_token_id)
    return {"train_dataset": dataset, "eval_dataset": None, "data_collator": data_collator}


def create_dual_prompt_data_module(processor, args: TrainPipelineConfig):
    dataset = SupervisedDataset(args=args, processor=processor, prompt=DualPrompt())
    data_collator = DataCollator(pad_token_id=processor.tokenizer.pad_token_id)
    return {"train_dataset": dataset, "eval_dataset": None, "data_collator": data_collator}
