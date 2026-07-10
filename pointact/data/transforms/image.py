# Copyright 2025 EO-Robotics Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import math
import random
from dataclasses import dataclass, field
from typing import Any

from lerobot.datasets.transforms import ImageTransformConfig, RandomSubsetApply, SharpnessJitter
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from torchvision.transforms.v2._utils import query_size


class RandomScaleCrop:
    """Randomly crop an image while preserving the original aspect ratio.
    Args:
        scale: The range of size (as a fraction of the original size) to sample from.
    """

    def __init__(self, scale: tuple[float, float]):
        self.scale = scale
        self.crop_bbox = None

    def __call__(self, img: Any) -> Any:
        h, w = query_size(img)

        area = h * w
        target_area = random.uniform(self.scale[0], self.scale[1]) * area

        aspect_ratio = w / h
        crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
        crop_w = int(round(target_area / crop_h))

        crop_w = min(crop_w, w)
        crop_h = min(crop_h, h)

        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)

        self.crop_bbox = (i, j, i + crop_h, j + crop_w)
        return v2.functional.crop(img, i, j, crop_h, crop_w)


@dataclass
class ImageTransformsConfig:
    """Image and wrist transforms for the LeRobot dataset."""

    enable: bool = True
    max_num_transforms: int = 3
    random_order: bool = True
    color_aug: bool = False

    tfs: dict[str, ImageTransformConfig] = field(
        default_factory=lambda: {
            "brightness": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"brightness": (0.8, 1.2)},
            ),
            "contrast": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"contrast": (0.8, 1.2)},
            ),
            "saturation": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"saturation": (0.5, 1.5)},
            ),
            "hue": ImageTransformConfig(
                type="ColorJitter",
                kwargs={"hue": (-0.05, 0.05)},
            ),
            "sharpness": ImageTransformConfig(
                type="SharpnessJitter",
                kwargs={"sharpness": (0.5, 1.5)},
            ),
            "crop": ImageTransformConfig(
                type="RandomScaleCrop",
                kwargs={"scale": (0.8, 1.0)},
            ),
            # "rotate": ImageTransformConfig(
            #     type="RadomRotate",
            #     kwargs={"degrees": (-3, 3)},
            # ),
        }
    )


class ImageTransforms(Transform):
    """Compose image transforms from configuration."""

    def __init__(self, cfg: ImageTransformsConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self.weights = []
        self.transforms = []
        self.crop_idx = None

        for tf_cfg in cfg.tfs.values():
            if tf_cfg.weight <= 0.0:
                continue

            match tf_cfg.type:
                case "Identity":
                    if self._cfg.color_aug:
                        self.transforms.append(v2.Identity(**tf_cfg.kwargs))
                    else:
                        continue
                case "ColorJitter":
                    if self._cfg.color_aug:
                        self.transforms.append(v2.ColorJitter(**tf_cfg.kwargs))
                    else:
                        continue
                case "SharpnessJitter":
                    if self._cfg.color_aug:
                        self.transforms.append(SharpnessJitter(**tf_cfg.kwargs))
                    else:
                        continue
                case "RandomScaleCrop":
                    self.transforms.append(RandomScaleCrop(**tf_cfg.kwargs))
                    self.crop_idx = len(self.transforms) - 1
                case _:
                    self.transforms.append(v2.Identity(**tf_cfg.kwargs))
            self.weights.append(tf_cfg.weight)

        n_subset = min(len(self.transforms), cfg.max_num_transforms)
        if n_subset == 0 or not cfg.enable:
            self.tf = v2.Identity()
        else:
            self.tf = RandomSubsetApply(
                transforms=self.transforms,
                p=self.weights,
                n_subset=n_subset,
                random_order=cfg.random_order,
            )

    def forward(self, *inputs: Any) -> Any:
        if self.crop_idx is not None:
            self.tf.transforms[self.crop_idx].crop_bbox = None
        return self.tf(*inputs)

    @property
    def crop_bbox(self):
        if self.crop_idx is not None:
            return self.tf.transforms[self.crop_idx].crop_bbox
        return None
