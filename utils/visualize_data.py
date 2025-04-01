import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataset import ContrastiveAugmentedLoader
from utils.config import config


def create_animation(volume_tensor, title="Volume Animation"):
    """
    Create a matplotlib animation from a 3D volume tensor.
    
    Args:
        volume_tensor: Tensor of shape (D, C, H, W) or (D, H, W)
        title: Title for the animation
    
    Returns:
        Animation object
    """
    # Check dimensions and reshape if needed
    if len(volume_tensor.shape) == 4:  # (D, C, H, W)
        depth, channels, height, width = volume_tensor.shape
        if channels == 1:
            # Squeeze the channel dimension for grayscale
            volume = volume_tensor.squeeze(1).numpy()
        else:
            # Take the first channel for multi-channel data
            volume = volume_tensor[:, 0].numpy()
    elif len(volume_tensor.shape) == 3:  # (D, H, W)
        volume = volume_tensor.numpy()
    else:
        raise ValueError(f"Unexpected tensor shape: {volume_tensor.shape}")
    
    # Normalize for visualization
    v_min = volume.min()
    v_max = volume.max()
    
    # Create figure and first frame
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.title(f"{title} (Frame: 1/{depth})")
    img = ax.imshow(volume[0], cmap='gray', vmin=v_min, vmax=v_max)
    
    def update(frame):
        img.set_array(volume[frame])
        plt.title(f"{title} (Frame: {frame+1}/{depth})")
        return [img]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=depth, interval=100, blit=True)
    return anim, fig


def save_sample_frames(volume_tensor, output_path, synapse_id=None, bbox_name=None, num_frames=5):
    """
    Save sample frames from a volume tensor as a grid.
    
    Args:
        volume_tensor: Tensor of shape (D, C, H, W) or (D, H, W)
        output_path: Path to save the visualization
        synapse_id: ID of the synapse (for title)
        bbox_name: Name of the bounding box (for title)
        num_frames: Number of frames to sample
    """
    # Check dimensions and reshape if needed
    if len(volume_tensor.shape) == 4:  # (D, C, H, W)
        depth, channels, height, width = volume_tensor.shape
        if channels == 1:
            # Squeeze the channel dimension for grayscale
            volume = volume_tensor.squeeze(1).numpy()
        else:
            # Take the first channel for multi-channel data
            volume = volume_tensor[:, 0].numpy()
    elif len(volume_tensor.shape) == 3:  # (D, H, W)
        volume = volume_tensor.numpy()
        depth = volume.shape[0]
    else:
        raise ValueError(f"Unexpected tensor shape: {volume_tensor.shape}")
    
    # Sample frames evenly
    if depth <= num_frames:
        frame_indices = range(depth)
    else:
        frame_indices = np.linspace(0, depth-1, num_frames, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(1, len(frame_indices), figsize=(len(frame_indices) * 4, 4))
    
    if len(frame_indices) == 1:
        axes = [axes]  # Make iterable for single frame case
    
    # Plot each frame
    for i, frame_idx in enumerate(frame_indices):
        axes[i].imshow(volume[frame_idx], cmap='gray')
        axes[i].set_title(f"Frame {frame_idx}")
        axes[i].axis('off')
    
    # Add overall title
    if synapse_id and bbox_name:
        fig.suptitle(f"Synapse: {synapse_id}, Bbox: {bbox_name}", fontsize=16)
    elif synapse_id:
        fig.suptitle(f"Synapse: {synapse_id}", fontsize=16)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def visualize_augmented_pairs(data_loader_helper, output_dir, num_samples=5):
    """
    Visualize augmented pairs from the contrastive data loader.
    
    Args:
        data_loader_helper: ContrastiveAugmentedLoader instance
        output_dir: Directory to save visualizations
        num_samples: Number of random samples to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    vol_data_dict = data_loader_helper.load_data()
    synapse_df = data_loader_helper.load_synapse_metadata()
    
    # Create dataset
    print("Creating dataset...")
    dataset, _ = data_loader_helper.create_dataset(
        vol_data_dict=vol_data_dict,
        synapse_df=synapse_df,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )
    
    # Get random samples
    print(f"Visualizing {num_samples} random samples...")
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(indices):
        # Get sample
        sample = dataset[idx]
        
        # Get data
        aug1 = sample['pixel_values_aug1']
        aug2 = sample['pixel_values_aug2']
        syn_id = sample['syn_id']
        bbox_name = sample['bbox_name']
        
        print(f"Sample {i+1}/{len(indices)}: Synapse {syn_id}, Bbox {bbox_name}")
        
        # Save visualization of augmented pairs
        save_sample_frames(
            aug1, 
            os.path.join(output_dir, f"sample_{i+1}_aug1_{syn_id}_{bbox_name}.png"),
            synapse_id=syn_id,
            bbox_name=bbox_name
        )
        
        save_sample_frames(
            aug2, 
            os.path.join(output_dir, f"sample_{i+1}_aug2_{syn_id}_{bbox_name}.png"),
            synapse_id=syn_id,
            bbox_name=bbox_name
        )
        
        # Save side-by-side comparison of a specific frame
        mid_frame = aug1.shape[0] // 2
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        if len(aug1.shape) == 4:  # (D, C, H, W)
            ax1.imshow(aug1[mid_frame, 0].numpy(), cmap='gray')
            ax2.imshow(aug2[mid_frame, 0].numpy(), cmap='gray')
        else:
            ax1.imshow(aug1[mid_frame].numpy(), cmap='gray')
            ax2.imshow(aug2[mid_frame].numpy(), cmap='gray')
            
        ax1.set_title("Augmentation 1")
        ax2.set_title("Augmentation 2")
        ax1.axis('off')
        ax2.axis('off')
        
        fig.suptitle(f"Synapse: {syn_id}, Bbox: {bbox_name}, Frame: {mid_frame}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"comparison_{i+1}_{syn_id}_{bbox_name}.png"), dpi=150)
        plt.close(fig)
        
        # Create animated GIFs for the first few samples
        if i < 3:  # Only do this for the first 3 samples to avoid too many files
            anim1, fig1 = create_animation(aug1, title=f"Augmentation 1 - Synapse {syn_id}")
            anim1.save(os.path.join(output_dir, f"anim_{i+1}_aug1_{syn_id}_{bbox_name}.gif"), 
                      writer='pillow', fps=10, dpi=80)
            plt.close(fig1)
            
            anim2, fig2 = create_animation(aug2, title=f"Augmentation 2 - Synapse {syn_id}")
            anim2.save(os.path.join(output_dir, f"anim_{i+1}_aug2_{syn_id}_{bbox_name}.gif"), 
                      writer='pillow', fps=10, dpi=80)
            plt.close(fig2)
    
    print(f"Visualizations saved to {output_dir}")


def visualize_processing_steps(data_loader_helper, syn_index=0, output_dir="results/processing_steps"):
    """
    Visualize the processing steps for a single synapse.
    
    Args:
        data_loader_helper: ContrastiveAugmentedLoader instance
        syn_index: Index of the synapse to visualize
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    vol_data_dict = data_loader_helper.load_data()
    synapse_df = data_loader_helper.load_synapse_metadata()
    
    if syn_index >= len(synapse_df):
        print(f"Error: Synapse index {syn_index} out of range (0-{len(synapse_df)-1})")
        return
    
    # Get synapse information
    syn_info = synapse_df.iloc[syn_index]
    bbox_name = syn_info['bbox_name']
    syn_id = str(syn_info.get('Var1', syn_index))
    
    print(f"Visualizing processing steps for synapse {syn_id} in {bbox_name}...")
    
    # Get volumes
    raw_vol, seg_vol, add_mask_vol = vol_data_dict.get(bbox_name, (None, None, None))
    
    if raw_vol is None:
        print(f"Error: No data found for bounding box {bbox_name}")
        return
    
    # Get coordinates
    central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
    side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
    side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))
    
    # Ensure data loader is initialized
    if data_loader_helper.data_loader is None:
        from data.dataloader import SynapseDataLoader
        data_loader_helper.data_loader = SynapseDataLoader("", "", "")
    
    # Extract raw and mask cubes
    raw_cube, mask_cube = data_loader_helper.data_loader.extract_raw_and_mask_cubes(
        raw_vol=raw_vol,
        seg_vol=seg_vol,
        add_mask_vol=add_mask_vol,
        central_coord=central_coord,
        side1_coord=side1_coord,
        side2_coord=side2_coord,
        subvolume_size=config.subvol_size,
        bbox_name=bbox_name,
    )
    
    # Visualize raw cube
    mid_slice = raw_cube.shape[2] // 2
    
    # Plot raw cube middle slice
    plt.figure(figsize=(8, 8))
    plt.imshow(raw_cube[:, :, mid_slice], cmap='gray')
    plt.title(f"Raw Cube (Synapse {syn_id}, Slice {mid_slice})")
    plt.colorbar(label="Intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"raw_cube_slice_{syn_id}.png"), dpi=150)
    plt.close()
    
    # Plot mask cube middle slice if available
    if mask_cube is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(mask_cube[:, :, mid_slice], cmap='viridis')
        plt.title(f"Mask Cube (Synapse {syn_id}, Slice {mid_slice})")
        plt.colorbar(label="Segment ID")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mask_cube_slice_{syn_id}.png"), dpi=150)
        plt.close()
        
        # Plot overlay
        plt.figure(figsize=(8, 8))
        plt.imshow(raw_cube[:, :, mid_slice], cmap='gray')
        plt.imshow(mask_cube[:, :, mid_slice] > 0, cmap='RdBu', alpha=0.3)
        plt.title(f"Overlay (Synapse {syn_id}, Slice {mid_slice})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"overlay_slice_{syn_id}.png"), dpi=150)
        plt.close()
    
    # Generate augmented views
    from data.dataloader import ContrastiveAugmentationProcessor
    processor = ContrastiveAugmentationProcessor(
        size=(config.subvol_size, config.subvol_size),
        augmentation_strength='medium'
    )
    
    # Generate multiple augmentations to show variation
    num_augmentations = 5
    
    # Prepare raw cube for processing (ensure correct shapes and dimensions)
    if raw_cube.shape[2] < config.num_frames:
        # Duplicate the last frame to reach the desired number
        raw_cube_padded = np.zeros((raw_cube.shape[0], raw_cube.shape[1], config.num_frames), dtype=np.float32)
        raw_cube_padded[:, :, :raw_cube.shape[2]] = raw_cube
        for z in range(raw_cube.shape[2], config.num_frames):
            raw_cube_padded[:, :, z] = raw_cube[:, :, -1]
        raw_cube = raw_cube_padded
    elif raw_cube.shape[2] > config.num_frames:
        # Sample frames evenly
        indices = np.linspace(0, raw_cube.shape[2]-1, config.num_frames, dtype=int)
        raw_cube_sampled = np.zeros((raw_cube.shape[0], raw_cube.shape[1], config.num_frames), dtype=np.float32)
        for i, idx in enumerate(indices):
            raw_cube_sampled[:, :, i] = raw_cube[:, :, idx]
        raw_cube = raw_cube_sampled
    
    # Extract frames for visualization
    frames = [raw_cube[:, :, z] for z in range(raw_cube.shape[2])]
    
    # Show multiple augmentations
    fig, axes = plt.subplots(1, num_augmentations, figsize=(num_augmentations * 4, 4))
    
    for i in range(num_augmentations):
        # Generate new augmentation
        aug = processor.process_raw_volume(raw_cube, apply_augmentation=True)
        
        # Show middle slice
        mid_aug_slice = aug.shape[0] // 2
        if len(aug.shape) == 4:  # (D, C, H, W)
            axes[i].imshow(aug[mid_aug_slice, 0].numpy(), cmap='gray')
        else:
            axes[i].imshow(aug[mid_aug_slice].numpy(), cmap='gray')
            
        axes[i].set_title(f"Aug {i+1}")
        axes[i].axis('off')
    
    fig.suptitle(f"Multiple Augmentations for Synapse {syn_id}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"multiple_augmentations_{syn_id}.png"), dpi=150)
    plt.close(fig)
    
    print(f"Processing step visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize data from the contrastive data loader")
    parser.add_argument('--mode', type=str, choices=['aug_pairs', 'processing'], default='aug_pairs',
                      help="Visualization mode: 'aug_pairs' for augmented pairs, 'processing' for processing steps")
    parser.add_argument('--output_dir', type=str, default='results/data_visualizations',
                      help="Directory to save visualizations")
    parser.add_argument('--num_samples', type=int, default=5,
                      help="Number of random samples to visualize (for aug_pairs mode)")
    parser.add_argument('--syn_index', type=int, default=0,
                      help="Index of synapse to visualize (for processing mode)")
    
    args = parser.parse_args()
    
    # Get config
    current_config = config.parse_args()
    
    # Initialize data loader helper
    data_loader_helper = ContrastiveAugmentedLoader(current_config)
    
    if args.mode == 'aug_pairs':
        visualize_augmented_pairs(data_loader_helper, args.output_dir, args.num_samples)
    elif args.mode == 'processing':
        visualize_processing_steps(data_loader_helper, args.syn_index, args.output_dir)


if __name__ == "__main__":
    main()