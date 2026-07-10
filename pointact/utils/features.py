from lerobot.configs.types import FeatureType, PolicyFeature


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:
    """Convert dataset feature metadata to LeRobot policy features."""
    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            feature_type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
            names = ft["names"]
            if names[2] in ["channel", "channels"]:  # (h, w, c) -> (c, h, w)
                shape = (shape[2], shape[0], shape[1])
        elif key == "observation.environment_state":
            feature_type = FeatureType.ENV
        elif key.startswith("observation.depths"):
            feature_type = FeatureType.VISUAL
        elif key.startswith("observation.camera"):
            feature_type = FeatureType.VISUAL
        elif key.startswith("observation"):
            feature_type = FeatureType.STATE
        elif key.startswith("action"):
            feature_type = FeatureType.ACTION
        else:
            continue
        policy_features[key] = PolicyFeature(
            type=feature_type,
            shape=shape,
        )
    return policy_features
