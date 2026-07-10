import open3d as o3d

def get_robot_joint_boxes(joints_pose_size, return_o3d=True):
    # joints_pose_size = obs["robot0_joints_pose_size"]

    obboxes = []
    for vector in joints_pose_size:
        pos = vector[:3]
        rot = vector[3:12].reshape(3, 3)
        size = vector[12:]

        if return_o3d:
            obbox = o3d.geometry.OrientedBoundingBox(pos, rot, size)
            obboxes.append(obbox)
        else:
            obboxes.append((pos, rot, size))
        
    return obboxes

