import numpy as np
import open3d as o3d


def get_point_idxs_inside_robot(robot_obboxes, xyz=None, pcd=None):
    if xyz is None:
        assert pcd is not None
        points = pcd.points
    else:
        points = o3d.utility.Vector3dVector(xyz)

    overlap_point_ids = set()
    for obbox in robot_obboxes:
        tmp = obbox.get_point_indices_within_bounding_box(points)
        overlap_point_ids = overlap_point_ids.union(set(tmp))
    overlap_point_ids = list(overlap_point_ids)

    return overlap_point_ids
    

def remove_points_inside_robot(points, robot_obboxes):
    inbox_point_ids = get_point_idxs_inside_robot(
        robot_obboxes, xyz=points[:, :3]
    )

    mask = np.ones((points.shape[0], ), dtype=bool)
    if len(inbox_point_ids) > 0:
        mask[inbox_point_ids] = False
        points = points[mask]

    return points, mask

