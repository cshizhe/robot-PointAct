"""
Extracted from https://github.com/rjgpinel/RLBench/blob/master/rlbench/backend/utils.py
"""

import numpy as np
from PIL import Image


DEFAULT_RGB_SCALE_FACTOR = 256000.0

DEFAULT_GRAY_SCALE_FACTOR = {
    np.uint8: 100.0,
    np.uint16: 1000.0,
    np.int32: DEFAULT_RGB_SCALE_FACTOR
}


def ClipFloatValues(float_array, min_value, max_value):
    """Clips values to the range [min_value, max_value].

    First checks if any values are out of range and prints a message.
    Then clips all values to the given range.

    Args:
    float_array: 2D array of floating point values to be clipped.
    min_value: Minimum value of clip range.
    max_value: Maximum value of clip range.

    Returns:
    The clipped array.

    """
    if float_array.min() < min_value or float_array.max() > max_value:
        float_array = np.clip(float_array, min_value, max_value)
    return float_array

def float_array_to_rgb_image(
    float_array,
    scale_factor=DEFAULT_RGB_SCALE_FACTOR,
    drop_blue=False
):
    """Convert a floating point array of values to an RGB image.

    Convert floating point values to a fixed point representation where
    the RGB bytes represent a 24-bit integer.
    R is the high order byte.
    B is the low order byte.
    The precision of the depth image is 1/256 mm.

    Floating point values are scaled so that the integer values cover
    the representable range of depths.

    This image representation should only use lossless compression.

    Args:
    float_array: Input array of floating point depth values in meters.
    scale_factor: Scale value applied to all float values.
    drop_blue: Zero out the blue channel to improve compression, results in 1mm
        precision depth values.

    Returns:
    24-bit RGB PIL Image object representing depth values.
    """
    # Scale the floating point array.
    scaled_array = np.floor(float_array * scale_factor + 0.5)
    # Convert the array to integer type and clip to representable range.
    min_inttype = 0
    max_inttype = 2**24 - 1
    scaled_array = ClipFloatValues(scaled_array, min_inttype, max_inttype)
    int_array = scaled_array.astype(np.uint32)
    # Calculate:
    #   r = (f / 256) / 256  high byte
    #   g = (f / 256) % 256  middle byte
    #   b = f % 256          low byte
    rg = np.divide(int_array, 256)
    r = np.divide(rg, 256)
    g = np.mod(rg, 256)
    image_shape = int_array.shape
    rgb_array = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    rgb_array[..., 0] = r
    rgb_array[..., 1] = g
    if not drop_blue:
        # Calculate the blue channel and add it to the array.
        b = np.mod(int_array, 256)
        rgb_array[..., 2] = b
    image_mode = 'RGB'
    image = Image.fromarray(rgb_array, mode=image_mode)
    return image

def image_to_float_array(image, scale_factor=None):
    """Recovers the depth values from an image.

    Reverses the depth to image conversion performed by FloatArrayToRgbImage or
    FloatArrayToGrayImage.

    The image is treated as an array of fixed point depth values.  Each
    value is converted to float and scaled by the inverse of the factor
    that was used to generate the Image object from depth values.  If
    scale_factor is specified, it should be the same value that was
    specified in the original conversion.

    The result of this function should be equal to the original input
    within the precision of the conversion.

    Args:
    image: Depth image output of FloatArrayTo[Format]Image.
    scale_factor: Fixed point scale factor.

    Returns:
    A 2D floating point numpy array representing a depth image.

    """
    image_array = np.array(image)
    image_dtype = image_array.dtype
    image_shape = image_array.shape

    channels = image_shape[2] if len(image_shape) > 2 else 1
    assert 2 <= len(image_shape) <= 3
    if channels == 3:
        # RGB image needs to be converted to 24 bit integer.
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
        if scale_factor is None:
            scale_factor = DEFAULT_RGB_SCALE_FACTOR
    else:
        if scale_factor is None:
            scale_factor = DEFAULT_GRAY_SCALE_FACTOR[image_dtype.type]
        float_array = image_array.astype(np.float32)
    scaled_array = float_array / scale_factor
    return scaled_array


def get_real_metric_depth(depth_image, near, far):
    '''
    Args:
        - depth_images: np.array, (H, W)
        - near, far: using near and far to recover depth
    Returns:
        - metric depths: np.array, (H, W)
    '''
    depth_image = np.maximum(depth_image * (far - near) + near, 0)
    return depth_image


def depth_to_point_cloud(depth_image, intrinsics, camera_to_world_matrix):
    """
    Convert depth image to point cloud in world coordinates.
    
    Parameters:
    -----------
    depth_image : numpy.ndarray
        2D array of depth values (in meters)
    intrinsics : numpy.ndarray
        3x3 camera intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    camera_to_world_matrix : numpy.ndarray
        4x4 transformation matrix from camera to world coordinates
    
    Returns:
    --------
    numpy.ndarray
        Point cloud in world coordinates (Nx3 array of [x, y, z])
    """
    # Get image dimensions
    height, width = depth_image.shape
    
    # Create pixel coordinate grids
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Extract intrinsic parameters
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    # Compute camera coordinates
    z = depth_image
    x_cam = (x - cx) * z / fx
    y_cam = (y - cy) * z / fy
    
    # Stack coordinates
    camera_coords = np.stack((x_cam, y_cam, z), axis=-1)
    
    # Reshape to point cloud format
    points = camera_coords.reshape(-1, 3)
    
    # # Filter out zero depth points
    # valid_mask = z.reshape(-1) > 0
    # points = points[valid_mask]
    
    # Homogeneous coordinates
    points_homo = np.column_stack((points, np.ones(len(points))))
    
    # Transform to world coordinates
    world_points = (camera_to_world_matrix @ points_homo.T).T[:, :3]

    world_points = world_points.reshape(height, width, 3)
    
    return world_points
