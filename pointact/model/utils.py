import math
import torch

def create_sinusoidal_pos_embedding(
    time: torch.tensor,
    dimension: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
    device="cpu",
) -> torch.Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    fraction = torch.linspace(0.0, 1.0, dimension // 2, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def create_mm_token_type_ids(input_ids, image_token_id=None, video_token_id=None):
    """Create multi-modal token type ids for the input ids."""
    mm_token_type_ids = torch.zeros_like(input_ids)
    if image_token_id is not None:
        image_mask = input_ids == image_token_id
        mm_token_type_ids[image_mask] = 1
    if video_token_id is not None:
        video_mask = input_ids == video_token_id
        mm_token_type_ids[video_mask] = 2
    return mm_token_type_ids