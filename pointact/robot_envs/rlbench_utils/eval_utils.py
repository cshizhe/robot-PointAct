import copy
import random
import numpy as np
from filelock import FileLock
import jsonlines
import imageio

import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def write_to_file(filepath, data):
    lock = FileLock(filepath+'.lock')
    with lock:
        with jsonlines.open(filepath, 'a', flush=True) as outf:
            outf.write(data)


def get_2d_action_chunks(action_chunk, extrinsics, intrinsics):
    world_to_camera_matrix = np.linalg.inv(extrinsics)
    
    # homogeneous (4D)
    points_3d_homog = np.column_stack((action_chunk[:, :3], np.ones(len(action_chunk))))
    
    # Transform points from world to camera coordinates
    camera_points = (world_to_camera_matrix @ points_3d_homog.T).T
    
    # Perspective projection (divide by Z)
    # Only consider first 3 coordinates after transformation
    camera_points_proj = camera_points[:, :3]
    
    # Handle points behind the camera or at origin
    valid_points_mask = camera_points_proj[:, 2] > 1e-8
    
    # Prepare output array
    image_points = np.zeros((len(action_chunk), 2), dtype=np.float64)
    
    # Project valid points
    if np.any(valid_points_mask):
        # Normalized image coordinates
        x_norm = camera_points_proj[valid_points_mask, 0] / camera_points_proj[valid_points_mask, 2]
        y_norm = camera_points_proj[valid_points_mask, 1] / camera_points_proj[valid_points_mask, 2]
        
        # Apply intrinsic matrix (convert to pixel coordinates)
        image_points[valid_points_mask, 0] = (
            intrinsics[0, 0] * x_norm + 
            intrinsics[0, 2]
        )
        image_points[valid_points_mask, 1] = (
            intrinsics[1, 1] * y_norm + 
            intrinsics[1, 2]
        )

    image_points = np.round(image_points).astype(np.int32)
    return image_points

def project_action_chunks_on_image(image, action_chunk, extrinsics, intrinsics, radius=2):
    
    image_points = get_2d_action_chunks(action_chunk, extrinsics, intrinsics)
    
    image = copy.deepcopy(image)
    for t, (x, y) in enumerate(image_points):
        color = np.floor(t/len(image_points)*255) # red
        image[max(0, y-radius):y+radius, max(0, x-radius):x+radius] = [color, 0, 0]
    
    return image

def save_images_to_video(images, video_path, fps=20):

    imageio.mimwrite(
        video_path, images, fps=fps, codec="libx264"
    )
    
    # height, width = images[0].shape[:2]

    # # OpenCV video writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # # Convert and write images
    # for image in images:
    #     # Convert Image to OpenCV format (RGB to BGR)
    #     cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     video.write(cv2_image)

    # video.release()


def get_rlbench_robot_workspace():
    # rlbench workspace
    # TABLE_HEIGHT = 0.7505 # meters

    # X_BBOX = (-0.5, 1.5)    # 0 is the robot base
    # Y_BBOX = (-1, 1)        # 0 is the robot base 
    # Z_BBOX = (0.2, 2)       # 0 is the floor

    # https://github.com/rjgpinel/RLBench/blob/master/rlbench/backend/scene.py#L67
    # task._scene._workspace

    X_BBOX = (-0.274, 0.774)    # 0 is the robot base
    Y_BBOX = (-0.655, 0.655)        # 0 is the robot base 
    Z_BBOX = (0.752, 1.751)       # 0 is the floor

    return {
        'X_BBOX': X_BBOX, 
        'Y_BBOX': Y_BBOX, 
        'Z_BBOX': Z_BBOX
    }