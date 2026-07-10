import numpy as np


def convert_opengl_to_opencv_camera(camera_params):
    """
    Convert OpenGL camera parameters in RLBench to OpenCV format
    """
    required_keys = ['intrinsics', 'extrinsics']
    for key in required_keys:
        if key not in camera_params:
            raise ValueError(f"Missing required key: {key}")
    
    converted_params = camera_params.copy()
    
    # Convert intrinsics matrix
    intrinsics = np.array(camera_params['intrinsics'])
    if intrinsics.shape != (3, 3):
        raise ValueError("Intrinsics must be 3x3 matrix")
        
    converted_intrinsics = intrinsics.copy()
    converted_intrinsics[0, 0] = abs(intrinsics[0, 0])
    converted_intrinsics[1, 1] = abs(intrinsics[1, 1])
    
    # Convert extrinsics (cam2world)
    extrinsics = np.array(camera_params['extrinsics'])
    
    # OpenGL to OpenCV coordinate transform
    coord_transform = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    
    # Apply transformation to cam2world directly
    cam_to_world = extrinsics @ np.linalg.inv(coord_transform)
    
    converted_params['intrinsics'] = converted_intrinsics.astype(np.float32)
    converted_params['extrinsics'] = cam_to_world.astype(np.float32)
    
    return converted_params

def extract_camera_info(robot_state, convert_camera=True):
    # Camera name mapping
    cameras = ['left_shoulder', 'right_shoulder', 'wrist', 'front']

    camera_params = ['extrinsics', 'intrinsics', 'near', 'far']
    camera_info = {}

    for camera in cameras:
        cam_name = f'{camera}_camera'
        cam_data = {}
        
        for param in camera_params:
            key = f"{cam_name}_{param}"
            cam_data[param] = robot_state[key]
                
        camera_info[camera] = {
            "camera_name": cam_name,
        }
        if convert_camera:
            cam_data = convert_opengl_to_opencv_camera(cam_data)
        camera_info[camera].update(cam_data)

    return camera_info

