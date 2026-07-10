ROBOT_DATASET_REGISTRY = {}


def register_robot_dataset(*names: str):
    def decorator(dataset_cls):
        for name in names:
            if name in ROBOT_DATASET_REGISTRY and ROBOT_DATASET_REGISTRY[name] is not dataset_cls:
                raise ValueError(f"Robot dataset '{name}' is already registered.")
            ROBOT_DATASET_REGISTRY[name] = dataset_cls
        return dataset_cls

    return decorator


def get_robot_dataset_class(name: str):
    try:
        return ROBOT_DATASET_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(ROBOT_DATASET_REGISTRY))
        raise KeyError(f"Unknown robot dataset class '{name}'. Available: {available}") from exc
