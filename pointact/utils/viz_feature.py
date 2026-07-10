from sklearn.decomposition import PCA

def visualize_feature_heatmap_pca(feature_heatmap):
    """
    Visualize a feature heatmap (H, W, C) or (N, C) using PCA to convert to RGB
    Parameters:
    -----------
    feature_heatmap : numpy.ndarray
        Input feature heatmap with shape (Height, Width, Channels)
    
    Returns:
    --------
    numpy.ndarray
        RGB visualization of the feature heatmap
    """
    # Reshape the heatmap to 2D: (H*W, C)
    feature_shape = feature_heatmap.shape
    reshaped_features = feature_heatmap.reshape(-1, feature_shape[-1])
    
    # Perform PCA to reduce to 3 components
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(reshaped_features)
    
    # Normalize PCA features to [0, 1] range for RGB visualization
    pca_features_normalized = (pca_features - pca_features.min(axis=0)) / \
                               (pca_features.max(axis=0) - pca_features.min(axis=0))
    
    # Reshape back to original spatial dimensions
    new_shape = list(feature_shape[:-1]) + [3]
    rgb_heatmap = pca_features_normalized.reshape(*new_shape)
    
    return rgb_heatmap