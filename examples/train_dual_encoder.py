#!/usr/bin/env python
"""
Example script to train a dual encoder model with both texture and shape information.
This example shows how to use segmentation masks as point clouds for shape-based learning.
"""

import os
import sys
import argparse

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config import config
from trainers.train_contrastive import train

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a dual encoder model with texture and shape information')
    
    # Add dual encoder specific arguments
    parser.add_argument('--use_dual_encoder', action='store_true', default=True, 
                        help='Use dual encoder with texture and shape paths (default: True)')
    parser.add_argument('--max_points', type=int, default=512,
                        help='Maximum number of points to sample from masks (default: 512)')
    
    # Add data path arguments
    parser.add_argument('--raw_base_dir', type=str, required=True,
                        help='Base directory for raw data')
    parser.add_argument('--seg_base_dir', type=str, required=True,
                        help='Base directory for segmentation data')
    parser.add_argument('--add_mask_base_dir', type=str, required=True,
                        help='Base directory for additional mask data')
    parser.add_argument('--synapse_coordinates_path', type=str, required=True,
                        help='Path to synapse coordinates Excel file')
    parser.add_argument('--synapse_coordinates_sheet', type=str, default='Sheet1',
                        help='Sheet name in the synapse coordinates Excel file')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--encoder_dim', type=int, default=256,
                        help='Dimension of encoder output')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Dimension of projection head output')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature for NT-Xent loss')
    parser.add_argument('--save_every_epochs', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--contrastive_output_dir', type=str, default='results/contrastive_dual',
                        help='Output directory for model checkpoints')
    parser.add_argument('--subvol_size', type=int, default=80,
                        help='Size of subvolume cube')
    parser.add_argument('--num_frames', type=int, default=80,
                        help='Number of frames to use')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Number of input channels')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # Parse arguments and update config
    args = parser.parse_args()
    
    # Print run configuration
    print("=== Training Dual Encoder Model ===")
    print(f"Using texture (CNN) and shape (point cloud) paths")
    print(f"Max points for point cloud: {args.max_points}")
    print(f"Raw data: {args.raw_base_dir}")
    print(f"Segmentation data: {args.seg_base_dir}")
    print(f"Additional mask data: {args.add_mask_base_dir}")
    print(f"Synapse coordinates: {args.synapse_coordinates_path}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}, Weight decay: {args.weight_decay}")
    print(f"Output directory: {args.contrastive_output_dir}")
    print("==============================")
    
    # Set config values from args
    for arg_name, arg_value in vars(args).items():
        setattr(config, arg_name, arg_value)
    
    # Train the model
    train(config) 