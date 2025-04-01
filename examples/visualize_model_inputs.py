#!/usr/bin/env python
"""
Script to visualize model inputs (3D volumes and point clouds) for the dual encoder model.
This will display both the raw 3D image data and point clouds extracted from the masks.
Uses the configuration from utils.config for all paths and parameters.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib.colors import ListedColormap
from functools import partial

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import directly from utils instead of utils.utils
from utils.config import config
from data.dataloader import SynapseDataLoader, ContrastiveAugmentationProcessor
from data.dataset import ContrastiveAugmentedLoader
from models.contrastive_model import PointCloudEncoder

# Custom colormaps for each structure type
CLEFT_CMAP = ListedColormap(['red'])
VESICLE_CMAP = ListedColormap(['green'])
PRESYNAPSE_CMAP = ListedColormap(['blue'])

# Add arguments for visualization
parser = argparse.ArgumentParser(description='Visualize model inputs')
parser.add_argument('--show_plots', action='store_true', help='Show plots instead of saving them')
parser.add_argument('--raw_base_dir', type=str, default='/path/to/raw_data', help='Base directory for raw data')
parser.add_argument('--seg_base_dir', type=str, default='/path/to/segmentation_data', help='Base directory for segmentation data')
parser.add_argument('--add_mask_base_dir', type=str, default='/path/to/additional_mask_data', help='Base directory for additional mask data')
parser.add_argument('--output_dir', type=str, default='output/visualizations', help='Output directory for visualizations')
parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to visualize')
parser.add_argument('--subvol_size', type=int, default=80, help='Size of the subvolume cube')

def visualize_volume_slice(volume, title, slice_idx=None, ax=None):
    """Visualize a slice from a 3D volume."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    if slice_idx is None:
        slice_idx = volume.shape[2] // 2  # Middle slice
    
    if slice_idx >= volume.shape[2]:
        slice_idx = volume.shape[2] - 1
        
    # Extract the slice
    slice_data = volume[:, :, slice_idx]
    
    # Display the slice
    ax.imshow(slice_data, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    
    return ax

def visualize_point_cloud(point_cloud, ax=None, color='blue', title=None, label=None, alpha=0.8, s=10):
    """Visualize a point cloud in 3D."""
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
    
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    
    # Plot the points
    ax.scatter(
        point_cloud[:, 0], 
        point_cloud[:, 1], 
        point_cloud[:, 2], 
        c=color, 
        alpha=alpha, 
        s=s,
        label=label
    )
    
    if title:
        ax.set_title(title)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    return ax

def extract_separate_point_clouds(mask_cube, max_points=None):
    """
    Extract separate point clouds for each mask type.
    EXCEPT vesicles - we skip vesicle point clouds entirely.
    
    IMPORTANT: This function extracts point clouds DIRECTLY from the mask data (not from overlaid or raw data).
    It uses the mask values directly, where:
    - 1 = cleft
    - 3 = presynapse/mitochondria
    
    Note: Vesicle point clouds (mask value 2) are intentionally excluded.
    
    Args:
        mask_cube: 3D mask with values 1=cleft, 2=vesicle, 3=mitochondria/presynapse
        max_points: Maximum number of points to sample per structure
        
    Returns:
        Dict of point clouds for each structure type (EXCEPT vesicles)
    """
    # Use the centralized implementation from SynapseDataLoader
    data_loader = SynapseDataLoader("", "", "", max_points=max_points, use_point_cloud=True)
    return data_loader.extract_separate_point_clouds(mask_cube)

def visualize_sample(raw_cube, mask_cube, sample_idx, output_dir=None):
    """Visualize raw 3D volume and point clouds for a sample."""
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 10))
    
    # Add subplots for volume slices (raw data)
    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 2)
    ax3 = fig.add_subplot(2, 4, 3)
    
    # Add subplot for mask overlay
    ax_overlay = fig.add_subplot(2, 4, 4)
    
    # Add subplot for combined point cloud
    ax4 = fig.add_subplot(2, 4, 5, projection='3d')
    
    # Add subplots for individual point clouds
    ax5 = fig.add_subplot(2, 4, 6, projection='3d')
    ax6 = fig.add_subplot(2, 4, 7, projection='3d')
    
    # Placeholder for potential future use
    ax7 = fig.add_subplot(2, 4, 8)
    ax7.axis('off')
    
    # Show volume slices (front, middle, back)
    slice_indices = [
        raw_cube.shape[2] // 4,
        raw_cube.shape[2] // 2,
        3 * raw_cube.shape[2] // 4
    ]
    
    visualize_volume_slice(raw_cube, f"Slice {slice_indices[0]}", slice_indices[0], ax1)
    visualize_volume_slice(raw_cube, f"Slice {slice_indices[1]}", slice_indices[1], ax2)
    visualize_volume_slice(raw_cube, f"Slice {slice_indices[2]}", slice_indices[2], ax3)
    
    # IMPORTANT: Extract point clouds DIRECTLY from the mask_cube
    # Each structure type (cleft, presynapse) gets its own separate mask cube
    # with no overlaps possible between different structures.
    # Values in original mask_cube: 1=cleft, 3=presynapse
    # Note: Vesicles (label 2) are intentionally excluded
    point_clouds = extract_separate_point_clouds(mask_cube, max_points=512)
    
    # Visualize combined point cloud
    ax4.set_title("Combined Point Cloud")
    if 'cleft' in point_clouds and len(point_clouds['cleft']) > 0:
        visualize_point_cloud(point_clouds['cleft'], ax4, color='red', label='Cleft', s=15)
    if 'presynapse' in point_clouds and len(point_clouds['presynapse']) > 0:
        visualize_point_cloud(point_clouds['presynapse'], ax4, color='blue', label='Presynapse', s=15)
    ax4.legend()
    
    # Visualize individual point clouds
    if 'cleft' in point_clouds and len(point_clouds['cleft']) > 0:
        visualize_point_cloud(point_clouds['cleft'], ax5, color='red', title="Cleft Point Cloud", s=20)
    else:
        ax5.set_title("No Cleft Points")
        ax5.set_box_aspect([1, 1, 1])
        
    if 'presynapse' in point_clouds and len(point_clouds['presynapse']) > 0:
        visualize_point_cloud(point_clouds['presynapse'], ax6, color='blue', title="Presynapse Point Cloud", s=20)
    else:
        ax6.set_title("No Presynapse Points")
        ax6.set_box_aspect([1, 1, 1])
    
    # Create color-coded mask overlay
    middle_slice_idx = raw_cube.shape[2] // 2
    raw_slice = raw_cube[:, :, middle_slice_idx]
    mask_slice = mask_cube[:, :, middle_slice_idx]
    
    # Create RGB image from raw slice
    raw_rgb = np.stack([raw_slice] * 3, axis=2)
    
    # Create masks for each structure type
    cleft_mask = mask_slice == 1
    presynapse_mask = mask_slice == 3
    
    # Apply color overlays with transparency
    overlay = raw_rgb.copy()
    alpha = 0.5  # Transparency factor
    
    # Apply red for cleft (label 1)
    if np.any(cleft_mask):
        overlay[cleft_mask] = alpha * np.array([1, 0, 0]) + (1 - alpha) * overlay[cleft_mask]
    
    # Apply blue for presynapse (label 3)
    if np.any(presynapse_mask):
        overlay[presynapse_mask] = alpha * np.array([0, 0, 1]) + (1 - alpha) * overlay[presynapse_mask]
    
    # Add overlay to separate subplot
    ax_overlay.imshow(overlay)
    ax_overlay.set_title(f"Mask Overlay (Slice {middle_slice_idx})")
    ax_overlay.axis('off')
    
    # Add legend for overlay colors
    patch_cleft = plt.Rectangle((0, 0), 1, 1, color='red', alpha=alpha, label='Cleft (1)')
    patch_presynapse = plt.Rectangle((0, 0), 1, 1, color='blue', alpha=alpha, label='Presynapse (3)')
    ax_overlay.legend(handles=[patch_cleft, patch_presynapse], 
              loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    # Set overall title
    plt.suptitle(f"Sample {sample_idx}: 3D Volume and Point Clouds (No Vesicles)", fontsize=16)
    plt.tight_layout()
    
    # Save or show the figure
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"sample_{sample_idx}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_synthetic_sample(size=80):
    """
    Create synthetic data for visualization when real data is not available.
    
    Args:
        size: Size of the cube to generate
        
    Returns:
        (raw_cube, mask_cube) tuple with synthetic data
    """
    # Create a blank volume
    raw_cube = np.zeros((size, size, size), dtype=np.float32)
    mask_cube = np.zeros((size, size, size), dtype=np.int32)
    
    # Create a spherical structure in the center (simulating a synapse)
    center = size // 2
    radius = size // 4
    
    # Create coordinates grid
    x, y, z = np.ogrid[:size, :size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    
    # Create raw volume with gaussian intensity from center
    raw_cube = np.exp(-(dist_from_center**2) / (2 * (radius/2)**2))
    
    # Create mask volume with different structures
    # Cleft (value 1): thin disc in the middle
    cleft_mask = (dist_from_center < radius * 0.5) & (np.abs(z - center) < size * 0.05)
    mask_cube[cleft_mask] = 1
    
    # Vesicles (value 2): small spheres on one side
    for i in range(5):
        vesicle_x = center + np.random.randint(-radius, radius)
        vesicle_y = center + np.random.randint(-radius, radius)
        vesicle_z = center + np.random.randint(radius//2, radius)
        
        vesicle_radius = size // 15
        vesicle_dist = np.sqrt((x - vesicle_x)**2 + (y - vesicle_y)**2 + (z - vesicle_z)**2)
        vesicle_mask = vesicle_dist < vesicle_radius
        mask_cube[vesicle_mask] = 2
    
    # Presynapse (value 3): larger structure on the other side
    presynapse_z = center - radius * 0.7
    presynapse_dist = np.sqrt((x - center)**2 + (y - center)**2 + (z - presynapse_z)**2)
    presynapse_mask = presynapse_dist < radius * 0.8
    mask_cube[presynapse_mask] = 3
    
    return raw_cube, mask_cube

def main():
    # Parse arguments directly
    args = parser.parse_args()
    
    # Set configuration values from arguments
    config.raw_base_dir = args.raw_base_dir
    config.seg_base_dir = args.seg_base_dir
    config.add_mask_base_dir = args.add_mask_base_dir
    config.show_plots = args.show_plots
    config.num_samples_to_visualize = args.num_samples
    config.subvol_size = args.subvol_size
    config.visualization_output_dir = args.output_dir
    config.max_points = 512  # Default value for max points
    
    # Create output directory if saving plots
    if not config.show_plots:
        os.makedirs(config.visualization_output_dir, exist_ok=True)
    
    # Flag to track if we're using real or synthetic data
    using_synthetic_data = False
    
    try:
        # Initialize data loader
        data_loader = SynapseDataLoader(
            config.raw_base_dir,
            config.seg_base_dir,
            config.add_mask_base_dir
        )
        
        # Check if we need to use synthetic data
        if not os.path.exists(config.raw_base_dir) or not os.path.exists(config.seg_base_dir):
            print("Data directories not found. Using synthetic data.")
            using_synthetic_data = True
        else:
            # Try to load data
            try:
                # Initialize loader helper
                loader_helper = ContrastiveAugmentedLoader(
                    config,
                    use_point_cloud=True,
                    max_points=config.max_points
                )
                
                # Load data
                vol_data_dict = loader_helper.load_data()
                synapse_df = loader_helper.load_synapse_metadata()
                
                # Check if we have valid synapse metadata
                if synapse_df is None or len(synapse_df) == 0:
                    print("No valid synapse metadata found. Using synthetic data for visualization.")
                    using_synthetic_data = True
                else:
                    # Filter to only include available bounding boxes
                    available_bboxes = list(vol_data_dict.keys())
                    synapse_df = synapse_df[synapse_df['bbox_name'].isin(available_bboxes)].reset_index(drop=True)
                    
                    # Check if we have any synapses after filtering
                    if len(synapse_df) == 0:
                        print("No synapses found in available bounding boxes. Using synthetic data for visualization.")
                        using_synthetic_data = True
                    else:
                        print(f"Found {len(synapse_df)} synapses in {len(available_bboxes)} bounding boxes")
            except Exception as e:
                print(f"Error loading data: {e}")
                print("Using synthetic data for visualization.")
                using_synthetic_data = True
    except Exception as e:
        print(f"Error initializing data loader: {e}")
        print("Using synthetic data for visualization.")
        using_synthetic_data = True
    
    if using_synthetic_data:
        # Generate synthetic data
        num_to_visualize = min(config.num_samples_to_visualize, 5)  # Limit to 5 for synthetic data
        
        for i in range(num_to_visualize):
            print(f"Generating synthetic sample {i+1}/{num_to_visualize}")
            raw_cube, mask_cube = create_synthetic_sample(size=config.subvol_size)
            
            # Visualize the sample
            output_dir = None if config.show_plots else config.visualization_output_dir
            visualize_sample(raw_cube, mask_cube, f"synthetic_{i+1}", output_dir)
            
        print(f"Visualization of {num_to_visualize} synthetic samples complete!")
        return
    
    # Visualize the specified number of samples
    num_to_visualize = min(config.num_samples_to_visualize, len(synapse_df))
    print(f"Visualizing {num_to_visualize} samples...")
    
    for i in range(num_to_visualize):
        # Get synapse info
        syn_info = synapse_df.iloc[i]
        bbox_name = syn_info['bbox_name']
        print(f"Processing sample {i+1}/{num_to_visualize} from {bbox_name}")
        
        try:
            # Get volumes
            raw_vol, seg_vol, add_mask_vol = vol_data_dict[bbox_name]
            
            # Get coordinates
            central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
            side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
            side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))
            
            # Extract raw and mask cubes
            raw_cube, mask_cube = data_loader.extract_raw_and_mask_cubes(
                raw_vol=raw_vol,
                seg_vol=seg_vol,
                add_mask_vol=add_mask_vol,
                central_coord=central_coord,
                side1_coord=side1_coord,
                side2_coord=side2_coord,
                subvolume_size=config.subvol_size,
                bbox_name=bbox_name
            )
            
            # Debug: Print mask statistics
            print(f"  Mask statistics:")
            unique_values = np.unique(mask_cube)
            print(f"  Unique values in mask: {unique_values}")
            for val in unique_values:
                count = np.sum(mask_cube == val)
                percentage = (count / mask_cube.size) * 100
                print(f"  Label {val}: {count} voxels ({percentage:.2f}% of volume)")
            
            # Visualize the sample
            output_dir = None if config.show_plots else config.visualization_output_dir
            visualize_sample(raw_cube, mask_cube, i, output_dir)
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            print("Generating a synthetic sample instead.")
            
            # Create a synthetic sample as fallback
            raw_cube, mask_cube = create_synthetic_sample(size=config.subvol_size)
            
            # Visualize the sample
            output_dir = None if config.show_plots else config.visualization_output_dir
            visualize_sample(raw_cube, mask_cube, f"synthetic_{i+1}", output_dir)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 