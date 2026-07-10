import numpy as np
import open3d as o3d
import torch

from .depth import depth_to_point_cloud

def crop_point_cloud_by_workspace(point_cloud, workspace, return_mask=False):
    """Return points within the workspace
    Args:
        point_cloud: (n_points, dim)
        workspace: dict
    """
    point_mask = (point_cloud[..., 0] > workspace['X_BBOX'][0]) & \
                 (point_cloud[..., 0] < workspace['X_BBOX'][1]) & \
                 (point_cloud[..., 1] > workspace['Y_BBOX'][0]) & \
                 (point_cloud[..., 1] < workspace['Y_BBOX'][1]) & \
                 (point_cloud[..., 2] > workspace['Z_BBOX'][0]) & \
                 (point_cloud[..., 2] < workspace['Z_BBOX'][1])
    point_cloud = point_cloud[point_mask]

    if return_mask:
        return point_cloud, point_mask
    return point_cloud

def get_voxelized_point_cloud_from_rgb_depth(
    rgbs, depths, intrinsics, extrinsics, point_workspace=None, 
    voxel_size=0.01
):
    nviews = len(depths)
    point_cloud = []
    for i in range(nviews):
        cur_point_cloud = depth_to_point_cloud(
            depths[i], intrinsics[i], extrinsics[i]
        )
        cur_point_cloud = cur_point_cloud.astype(np.float32)
        # Concatenate rgb values in points: (H, W, C) [0, 1]
        cur_point_cloud = np.concatenate(
            [cur_point_cloud, rgbs[i] / 255.], axis=2
        ).reshape(-1, 6)

        if point_workspace is not None:
            cur_point_cloud = crop_point_cloud_by_workspace(
                cur_point_cloud, point_workspace
            )
        point_cloud.append(cur_point_cloud)

    point_cloud = np.concatenate(point_cloud)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
    # pcd, _, voxel_point_indexes = pcd.voxel_down_sample_and_trace(
    #     voxel_size=0.01,
    #     min_bound=point_cloud[:, :3].min(0),
    #     max_bound=point_cloud[:, :3].max(0)
    # )
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    point_cloud = np.concatenate(
        [np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1
    ).astype(np.float32)

    return point_cloud

def merge_point_cloud(point_cloud_list, voxel_size=0.01):
    """Downsample the point cloud multiple times results in smaller point cloud
    1. Voxel grid alignment is implicit

    Open3D aligns the voxel grid to the axis-aligned bounding box min corner of the input cloud.

    That means:

    After the first downsample, the bounding box changes slightly

    On the second downsample, voxel boundaries shift

    Points fall into different voxels

    Result:
    Different centroids → different output points

    This is the most common reason.
    """
    point_cloud = np.concatenate(point_cloud_list)

    if voxel_size is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
        # pcd, _, voxel_point_indexes = pcd.voxel_down_sample_and_trace(
        #     voxel_size=0.01,
        #     min_bound=point_cloud[:, :3].min(0),
        #     max_bound=point_cloud[:, :3].max(0)
        # )
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        point_cloud = np.concatenate(
            [np.asarray(pcd.points), np.asarray(pcd.colors)], axis=1
        ).astype(np.float32)

    return point_cloud


def farthest_point_sampling(points: np.ndarray, k: int, start_idx: int | None = None):
    """
    Farthest Point Sampling (FPS) in Euclidean space.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, D) with N points in D-dimensional space.
    k : int
        Number of points to sample (k <= N).
    start_idx : int | None
        Optional index of the first point to start FPS from.
        If None, a random point is chosen.

    Returns
    -------
    sampled_points : np.ndarray
        Array of shape (k, D) with the sampled points.
    sampled_indices : np.ndarray
        Array of shape (k,) with indices into the original `points`.
    """
    points = np.ascontiguousarray(points)
    N, D = points.shape

    if k > N:
        raise ValueError(f"k={k} cannot be larger than number of points N={N}.")

    # Array to store selected indices
    sampled_indices = np.empty(k, dtype=np.int64)

    # Choose initial point
    if start_idx is None:
        start_idx = np.random.randint(0, N)
    sampled_indices[0] = start_idx

    # Squared distances to the nearest selected point so far
    # Initialize with +inf so first update always wins
    min_dist_sq = np.full(N, np.inf, dtype=points.dtype)

    # Main FPS loop
    last_selected = start_idx
    for i in range(1, k):
        # Compute squared distances from all points to the last selected point
        diff = points - points[last_selected]           # (N, D)
        dist_sq = np.einsum('nd,nd->n', diff, diff)     # fast batch dot product

        # Update the running minimum distance to the sampled set
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)

        # Pick the farthest point from the set
        last_selected = int(np.argmax(min_dist_sq))
        sampled_indices[i] = last_selected

    sampled_points = points[sampled_indices]
    return sampled_points, sampled_indices


@torch.no_grad()
def farthest_point_sampling_torch(points: torch.Tensor, k: int, start_idx: int | None = None):
    """
    Farthest Point Sampling using PyTorch (supports GPU).

    Parameters
    ----------
    points : torch.Tensor
        Tensor of shape (N, D), on CPU or GPU.
    k : int
        Number of points to sample.
    start_idx : int | None
        Optional starting index. If None, uses a random one.

    Returns
    -------
    sampled_points : torch.Tensor
        Tensor of shape (k, D).
    sampled_indices : torch.Tensor
        Long tensor of shape (k,).
    """
    if points.dim() != 2:
        raise ValueError("points must be of shape (N, D)")

    device = points.device
    N, D = points.shape

    if k > N:
        raise ValueError(f"k={k} cannot be larger than N={N}")

    sampled_indices = torch.empty(k, dtype=torch.long, device=device)

    if start_idx is None:
        start_idx = torch.randint(0, N, (1,), device=device).item()
    sampled_indices[0] = start_idx

    min_dist_sq = torch.full((N,), float('inf'), device=device, dtype=points.dtype)

    last_selected = start_idx
    for i in range(1, k):
        diff = points - points[last_selected]          # (N, D)
        dist_sq = (diff * diff).sum(dim=1)             # (N,)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)

        last_selected = torch.argmax(min_dist_sq).item()
        sampled_indices[i] = last_selected

    sampled_points = points[sampled_indices]
    return sampled_points, sampled_indices
