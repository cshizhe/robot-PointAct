'''https://github.com/rjgpinel/RLBench/blob/ebdc3392c1a11c4cdcc9a440cd61ec345bef42ec/rlbench/backend/utils.py#L220
'''
import numpy as np
from PIL import Image

def _resize_if_needed(image, size):
    if image.size[0] != size[0] or image.size[1] != size[1]:
        image = image.resize(size)
    return image

def rgb_handles_to_mask(rgb_coded_handles):
  # rgb_coded_handles should be (w, h, c)
  # Handle encoded as : handle = R + G * 256 + B * 256 * 256
  rgb_coded_handles *= 255  # takes rgb range to 0 -> 255
  rgb_coded_handles.astype(int)
  return (rgb_coded_handles[:, :, 0] +
          rgb_coded_handles[:, :, 1] * 256 +
          rgb_coded_handles[:, :, 2] * 256 * 256)


def load_mask_image(image_file):
    # image_size: (width, height)
    sem_mask = rgb_handles_to_mask(
        np.array(
            Image.open(image_file)
        ) / 255.0
    ).astype(np.uint16)

    return sem_mask