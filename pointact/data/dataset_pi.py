import random

import torch
import transformers
from lerobot.constants import ACTION, OBS_IMAGE, OBS_STATE
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from pointact.data.robot.multi_data import MultiLeRobotDataset
from pointact.data.schema import DataConfig, LerobotConfig
from pointact.data.transforms.image import ImageTransforms, ImageTransformsConfig
from pointact.utils.torch_utils import pad_vector
from pointact.train.pipeline_config import TrainPipelineConfig


class SupervisedPiDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        args: TrainPipelineConfig,
        processor: transformers.ProcessorMixin,
    ):
        super().__init__()
        self.args = args
        self.processor = processor

        data_configs = self.load_data_config(args)
        self.lerobot_dataset = self.build_lerobot_dataset(args, data_configs)

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

    def __len__(self):
        return len(self.lerobot_dataset)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        item = self.lerobot_dataset[i]

        images = [value for key, value in item.items() if key.startswith(OBS_IMAGE)]
        resized_images, img_masks = [], []
        for image in images:
            image, img_mask = self.processor.prepare_images(image.unsqueeze(0))
            resized_images.append(image)
            img_masks.append(img_mask)
        images = torch.cat(resized_images, 0)
        img_masks = torch.cat(img_masks, 0)
        # images = torch.stack(images, dim=0) # images of different sizes
        # images, img_masks = self.processor.prepare_images(images)

        states = [value for key, value in item.items() if key.startswith(OBS_STATE)]
        states = torch.cat(states, dim=-1)
        text_ids, text_masks, states = self.processor.prepare_text_state(task=item["task"], state=states)

        action_is_pad = item[f"{ACTION}_is_pad"]
        actions = [value for key, value in item.items() if key.startswith(ACTION) and "is_pad" not in key]
        actions = torch.cat(actions, dim=-1)
        actions = self.processor.normalize_robot_action(actions)

        outputs = {
            "text_ids": text_ids,
            "text_masks": text_masks,
            "images": images,
            "img_masks": img_masks,
            "actions": actions,
            "action_is_pad": action_is_pad,
        }
        if states is not None:
            outputs["states"] = states
        return outputs


class SupervisedAdvantagePiDataset(SupervisedPiDataset):
    """Dataset for PI0 advantage-conditioned supervised fine-tuning."""

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        item = self.lerobot_dataset[i]

        images = [value for key, value in item.items() if key.startswith(OBS_IMAGE)]
        resized_images, img_masks = [], []
        for image in images:
            image, img_mask = self.processor.prepare_images(image.unsqueeze(0))
            resized_images.append(image)
            img_masks.append(img_mask)
        images = torch.cat(resized_images, 0)
        img_masks = torch.cat(img_masks, 0)

        states = [value for key, value in item.items() if key.startswith(OBS_STATE)]
        states = torch.cat(states, dim=-1)

        advantage = item.get("advantage", torch.tensor(1, dtype=torch.long))
        dropout = getattr(self.processor, "advantage_condition_dropout", 0.0) or 0.0
        drop_advantage = random.random() < dropout
        text_ids, text_masks, states = self.processor.prepare_text_state(
            task=item["task"],
            state=states,
            advantage=advantage,
            drop_advantage=drop_advantage,
        )

        action_is_pad = item[f"{ACTION}_is_pad"]
        actions = [value for key, value in item.items() if key.startswith(ACTION) and "is_pad" not in key]
        actions = torch.cat(actions, dim=-1)
        actions = self.processor.normalize_robot_action(actions)

        outputs = {
            "text_ids": text_ids,
            "text_masks": text_masks,
            "images": images,
            "img_masks": img_masks,
            "actions": actions,
            "action_is_pad": action_is_pad,
        }
        if states is not None:
            outputs["states"] = states
        return outputs
    
def pi_data_collator(batch):
    data_dict = {}

    for key in batch[0].keys():
        data_dict[key] = [example[key] for example in batch]

    for key in ["text_ids", "text_masks"]:
        data_dict[key] = pad_sequence(data_dict[key], batch_first=True, padding_value=0)

    for key in ["images", "img_masks", "actions", "action_is_pad", "states"]:
        if key in data_dict:
            data_dict[key] = torch.stack(data_dict[key], 0)

    return data_dict


def create_pi_data_module(processor, args: TrainPipelineConfig):
    model_class = (args.model_class or "").split(".")[-1].lower()
    dataset_cls = (
        SupervisedAdvantagePiDataset
        if "advantage" in model_class
        else SupervisedPiDataset
    )
    dataset = dataset_cls(args=args, processor=processor)
    return {"train_dataset": dataset, "eval_dataset": None, "data_collator": pi_data_collator}
