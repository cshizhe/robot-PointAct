RLBENCH_FEATURES = {
    "observation.images.front_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.left_shoulder_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.right_shoulder_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "gripper"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": {"motors": ["x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "gripper"]},
    },
}
