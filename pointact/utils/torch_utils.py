import os
import datetime
import numpy as np
import random

import einops

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


def get_random_seed():
    seed = (
        os.getpid()
        + int(datetime.now().strftime("%S%f"))
        + int.from_bytes(os.urandom(2), "big")
    )
    return seed

def set_seed(seed=None):
    if seed is None:
        seed = get_random_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def grid_sample_interpolation(feature_map, coordinate):
    """
    Interpolate feature using torch.nn.functional.grid_sample
    
    Args:
    feature_map (torch.Tensor): Input feature map of shape (H, W, C)
    coordinate (torch.Tensor): Continuous coordinate of shape (N, 2), [0, 1]
    
    Returns:
    torch.Tensor: Interpolated feature vector of shape (C,)
    """
    # Reshape feature map to (1, C, H, W)
    feature_map = einops.rearrange(feature_map, 'h w c -> c h w').unsqueeze(0)
    
    # Convert coordinate from [0, 1] to [-1, 1], (1, 1, N, 2)
    grid_coord = (coordinate * 2 - 1).unsqueeze(0).unsqueeze(0)
    
    # Use grid_sample for interpolation
    interpolated = F.grid_sample(
        feature_map, 
        grid_coord, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=True
    )
    # (1, C, 1, N)
    # print(interpolated.size())
    
    return interpolated[0, :, 0]

def pad_vector(vector, new_dim, value=0):
    """Pad a tensor on its last dimension."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device) + value
    new_vector[..., :current_dim] = vector
    return new_vector