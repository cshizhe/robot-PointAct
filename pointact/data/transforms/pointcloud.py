import numpy as np
import torch
from scipy.spatial.transform import Rotation


def augment_point_cloud_color(
    c,
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    jitter_std=0.02,
    gamma=0.0,
    auto_contrast=False,
):
    """Apply simple color agumentation to point cloud RGB values.
    Args:
        c: np.ndarray (N, 3) RGB in [0, 1] or [0, 255]
    Returns:
        augmented color (same shape)
    """

    # normalize if color is 0-255
    if c.max() > 1.0:
        c = c / 255.0

    if brightness > 0:
        scale = 1 + (np.random.rand() * 2 - 1) * brightness
        c *= scale

    if contrast > 0:
        mean = c.mean(axis=0, keepdims=True)
        scale = 1 + (np.random.rand() * 2 - 1) * contrast
        c = (c - mean) * scale + mean

    if saturation > 0:
        gray = np.dot(c, np.array([0.299, 0.587, 0.114]))
        gray = gray[:, None]
        scale = 1 + (np.random.rand() * 2 - 1) * saturation
        c = (c - gray) * scale + gray

    if jitter_std > 0:
        c += np.random.normal(0, jitter_std, size=c.shape)

    if gamma > 0:
        g = 1 + (np.random.rand() * 2 - 1) * gamma
        c = np.power(np.clip(c, 0, 1), g)

    if auto_contrast:
        lo = c.min(axis=0, keepdims=True)
        hi = c.max(axis=0, keepdims=True)
        c = (c - lo) / (hi - lo + 1e-6)

    return np.clip(c, 0, 1)


def random_rotate_point_around_z(pc, angle=None):
    """
    Args:
        pc: np.array or torch.Tensor with shape (..., 3)
        angle: float in radians
    """
    if angle is None:
        angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(angle), np.sin(angle)
    rot = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]]).astype(np.float32)

    if isinstance(pc, torch.Tensor):
        rotated = np.dot(pc.numpy(), np.transpose(rot))
        return torch.from_numpy(rotated)
    return np.dot(pc, np.transpose(rot))


def random_rotate_quat_around_z(quat, angle):
    is_tensor = isinstance(quat, torch.Tensor)
    if is_tensor:
        quat = quat.numpy()
    quat = Rotation.from_quat(quat, scalar_first=False) # xyzw
    aug = Rotation.from_euler("z", angle, degrees=False)
    new_quat = (aug * quat).as_quat(scalar_first=False)
    if is_tensor:
        return torch.from_numpy(new_quat).float()
    return new_quat

def random_rotate_delta_quat_around_z(delta_quat, angle):
    is_tensor = isinstance(delta_quat, torch.Tensor)
    if is_tensor:
        delta_quat = delta_quat.numpy()
    delta_quat = Rotation.from_quat(delta_quat, scalar_first=False) # xyzw
    aug = Rotation.from_euler("z", angle, degrees=False)
    new_quat = (aug * delta_quat * aug.inv()).as_quat(scalar_first=False)
    if is_tensor:
        return torch.from_numpy(new_quat).float()
    return new_quat
