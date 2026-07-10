from safetensors import safe_open

def load_safetensors_weights(checkpoint_files):
    """
    Load weights from multiple SafeTensors checkpoint files
    
    Args:
        checkpoint_files (list): List of paths to SafeTensors checkpoint files
    
    Returns:
        dict: Consolidated model weights
    """
    model_weights = {}
    
    for checkpoint_file in checkpoint_files:
        with safe_open(checkpoint_file, framework="pt") as f:
            # Get all keys in the current checkpoint file
            keys = f.keys()
            
            # Load each tensor
            for key in keys:
                model_weights[key] = f.get_tensor(key)
    
    return model_weights