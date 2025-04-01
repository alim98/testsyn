import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import sys
import pickle
from tqdm import tqdm

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.contrastive_model import ContrastiveModel, Conv3DEncoder
from utils.config import config

def extract_features(model_path, point_clouds_path, device='cuda'):
    """
    Extract features from the trained encoder for all synapses using pre-extracted point clouds.
    
    Args:
        model_path: Path to the saved encoder weights
        point_clouds_path: Path to the pre-extracted point clouds
        device: Device to run inference on
        
    Returns:
        features: Extracted features for all synapses
        synapse_ids: Corresponding synapse IDs
        bbox_names: Corresponding bounding box names
    """
    print(f"Loading model from {model_path}...")
    
    # Load the state dict without initializing a model first
    state_dict = torch.load(model_path)
    
    # Check if this is a dual encoder model by examining keys in state_dict
    is_dual_encoder = any('texture_encoder' in key for key in state_dict.keys())
    
    # Dynamically load appropriate model based on state dict
    if is_dual_encoder:
        print("Detected dual encoder model structure")
        
        # We'll create the texture encoder directly from state dict
        texture_encoder = Conv3DEncoder(
            in_channels=config.in_channels,
            feature_dim=config.encoder_dim
        )
        
        # Extract texture encoder state dict
        texture_encoder_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('texture_encoder.'):
                # Remove 'texture_encoder.' prefix
                texture_encoder_state_dict[key.replace('texture_encoder.', '')] = value
        
        # Load the texture encoder weights
        texture_encoder.load_state_dict(texture_encoder_state_dict)
        texture_encoder.eval()
        texture_encoder.to(device)
        
        # This is our feature extractor
        feature_extractor = texture_encoder
    else:
        print("Detected single encoder model structure")
        
        # Check state dict structure to determine if it's a full model or just encoder
        if any('encoder.encoder' in key for key in state_dict.keys()):
            print("Found nested encoder structure, creating new model and extracting encoder weights")
            
            # Initialize a model with the standard architecture
            model = ContrastiveModel(
                in_channels=config.in_channels, 
                encoder_dim=config.encoder_dim,
                projection_dim=config.projection_dim
            )
            
            # Create a new state dict with correctly mapped keys
            encoder_state_dict = {}
            for key, value in state_dict.items():
                # Match keys like "encoder.encoder.0.weight" to "encoder.0.weight"
                if key.startswith('encoder.encoder.'):
                    new_key = key.replace('encoder.encoder.', 'encoder.')
                    encoder_state_dict[new_key] = value
                # Match keys like "encoder.dense.0.weight" to "dense.0.weight"
                elif key.startswith('encoder.dense.'):
                    new_key = key.replace('encoder.dense.', 'dense.')
                    encoder_state_dict[new_key] = value
            
            # Create a standalone encoder
            encoder = Conv3DEncoder(
                in_channels=config.in_channels,
                feature_dim=config.encoder_dim
            )
            
            # Load the encoder weights
            encoder.load_state_dict(encoder_state_dict)
            encoder.eval()
            encoder.to(device)
            
            # Use the standalone encoder
            feature_extractor = encoder
        else:
            # Assume this is a standalone encoder checkpoint
            encoder = Conv3DEncoder(
                in_channels=config.in_channels,
                feature_dim=config.encoder_dim
            )
            
            # Load the encoder weights directly
            encoder.load_state_dict(state_dict)
            encoder.eval()
            encoder.to(device)
            
            # Use the encoder as feature extractor
            feature_extractor = encoder
    
    print(f"Loading pre-extracted point clouds from {point_clouds_path}...")
    # Load pre-extracted point clouds
    with open(point_clouds_path, 'rb') as f:
        point_clouds_data = pickle.load(f)
    
    print(f"Loaded point clouds for {len(point_clouds_data)} synapses")
    
    # Process each synapse individually
    all_features = []
    all_synapse_ids = []
    all_bbox_names = []
    
    with torch.no_grad():
        for synapse_id, data in tqdm(point_clouds_data.items(), desc="Extracting features"):
            # Get the raw cube from the pre-extracted data
            raw_cube = data['raw_cube']
            
            # Convert raw cube to volume tensor
            volume_tensor = preprocess_raw_cube(raw_cube, device)
            
            # Extract features directly using the feature extractor
            features = feature_extractor(volume_tensor)
            
            # Add to lists
            all_features.append(features.cpu().numpy())
            
            # Extract bbox_name and syn_id from synapse_id
            if '_' in synapse_id:
                bbox_name, syn_id = synapse_id.split('_', 1)
            else:
                bbox_name = 'unknown'
                syn_id = synapse_id
                
            all_synapse_ids.append(syn_id)
            all_bbox_names.append(bbox_name)
    
    # Concatenate features
    features = np.vstack(all_features)
    
    return features, all_synapse_ids, all_bbox_names

def preprocess_raw_cube(raw_cube, device='cuda'):
    """
    Preprocess a raw cube for inference.
    
    Args:
        raw_cube: Raw 3D cube
        device: Device to run inference on
        
    Returns:
        Preprocessed tensor ready for the model
    """
    from data.dataloader import Synapse3DProcessor
    
    # Create processor with standard settings
    processor = Synapse3DProcessor(size=(config.subvol_size, config.subvol_size))
    
    # Extract frames from raw cube
    frames = [raw_cube[:, :, z] for z in range(raw_cube.shape[2])]
    
    # Process frames
    processed = processor(frames, return_tensors="pt")
    
    # Add batch dimension and move to device
    volume_tensor = processed["pixel_values"].unsqueeze(0).to(device)
    
    return volume_tensor

def perform_clustering(features, n_clusters=5, algorithm='kmeans'):
    """
    Perform clustering on the extracted features.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        n_clusters: Number of clusters (for KMeans)
        algorithm: Clustering algorithm ('kmeans' or 'dbscan')
        
    Returns:
        labels: Cluster labels for each sample
    """
    if algorithm == 'kmeans':
        print(f"Performing KMeans clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
    elif algorithm == 'dbscan':
        print("Performing DBSCAN clustering...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(features)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    return labels

def visualize_clusters(features, labels, synapse_ids, bbox_names, output_dir='results/clustering'):
    """
    Visualize the clustering results.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Cluster labels
        synapse_ids: Synapse IDs
        bbox_names: Bounding box names
        output_dir: Directory to save visualization plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'synapse_id': synapse_ids,
        'bbox_name': bbox_names,
        'cluster': labels
    })
    results_df.to_csv(os.path.join(output_dir, 'clustering_results.csv'), index=False)
    
    # Determine appropriate number of PCA components
    n_samples, n_features = features.shape
    min_dim = min(n_samples, n_features)
    n_components = min(min_dim - 1, 50)  # Use at most min_dim-1 or 50 components
    
    # Reduce dimensionality for visualization
    print(f"Reducing dimensionality with PCA to {n_components} components...")
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    
    # Calculate explained variance
    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA explained variance: {explained_variance:.2f}%")
    
    print("Reducing dimensionality with t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init='pca')
    features_tsne = tsne.fit_transform(features_pca)
    
    # Visualize the clusters
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization of Synapse Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_clusters.png'), dpi=300)
    
    # Create a more detailed plot with bbox information
    plt.figure(figsize=(12, 10))
    bbox_names_unique = list(set(bbox_names))
    
    # Ensure we don't run out of markers
    available_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    if len(bbox_names_unique) > len(available_markers):
        # Repeat markers if needed
        markers = available_markers * (len(bbox_names_unique) // len(available_markers) + 1)
        bbox_to_marker = {bbox: marker for bbox, marker in zip(bbox_names_unique, markers)}
    else:
        bbox_to_marker = {bbox: marker for bbox, marker in 
                        zip(bbox_names_unique, available_markers[:len(bbox_names_unique)])}
    
    # Plot by bounding box
    for bbox in bbox_names_unique:
        mask = [b == bbox for b in bbox_names]
        plt.scatter(
            features_tsne[mask, 0], 
            features_tsne[mask, 1], 
            c=[labels[i] for i, m in enumerate(mask) if m],
            marker=bbox_to_marker[bbox],
            label=bbox,
            alpha=0.7,
            cmap='viridis'
        )
    
    plt.colorbar(label='Cluster')
    plt.title('t-SNE Visualization of Synapse Clusters by Bounding Box')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_clusters_by_bbox.png'), dpi=300)
    
    # Create heatmap of feature correlation (only if dataset is not too large)
    if n_features <= 100:  # Only create correlation heatmap for manageable feature dimensions
        plt.figure(figsize=(12, 10))
        feature_corr = np.corrcoef(features)
        sns.heatmap(feature_corr, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=300)
    
    # Count synapses per cluster
    cluster_counts = results_df['cluster'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind='bar')
    plt.title('Number of Synapses per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_counts.png'), dpi=300)
    
    print(f"Visualization complete. Results saved to {output_dir}")
    
def main():
    """Main function to run the clustering pipeline."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Cluster synapses using features from a trained encoder")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved encoder weights')
    parser.add_argument('--point_clouds_path', type=str, required=True, help='Path to pre-extracted point clouds file')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for KMeans')
    parser.add_argument('--algorithm', type=str, default='kmeans', choices=['kmeans', 'dbscan'], help='Clustering algorithm')
    parser.add_argument('--output_dir', type=str, default='results/clustering', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Update config with command-line arguments
    config_obj = config.parse_args()
    
    # Extract features from pre-extracted point clouds
    features, synapse_ids, bbox_names = extract_features(args.model_path, args.point_clouds_path)
    
    # Perform clustering
    labels = perform_clustering(features, n_clusters=args.n_clusters, algorithm=args.algorithm)
    
    # Visualize results
    visualize_clusters(features, labels, synapse_ids, bbox_names, output_dir=args.output_dir)
    
if __name__ == "__main__":
    main() 