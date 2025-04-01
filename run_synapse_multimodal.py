#!/usr/bin/env python
"""
Launcher script for running synapse multimodal model visualization and training.
Uses the configuration from utils.config for all paths and parameters.
"""

import os
import sys
import argparse
import subprocess

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config import config

def visualize_data():
    """Run the visualization script with the current configuration."""
    print("=== Visualizing Multimodal Inputs ===")
    print(f"Raw data: {config.raw_base_dir}")
    print(f"Segmentation data: {config.seg_base_dir}")
    print(f"Additional mask data: {config.add_mask_base_dir}")
    print(f"Synapse coordinates: {config.synapse_coordinates_path}")
    print(f"Number of samples: {config.num_samples_to_visualize}")
    print(f"Output directory: {config.visualization_output_dir}")
    print(f"Show plots: {config.show_plots}")
    print("==============================")
    
    # Run the visualization script as a module
    subprocess.call([sys.executable, "examples/visualize_model_inputs.py"])
    
def train_model():
    """Run the training script with the current configuration."""
    print("=== Training Multimodal Encoder ===")
    print(f"Raw data: {config.raw_base_dir}")
    print(f"Segmentation data: {config.seg_base_dir}")
    print(f"Additional mask data: {config.add_mask_base_dir}")
    print(f"Synapse coordinates: {config.synapse_coordinates_path}")
    print(f"Encoder dimension: {config.encoder_dim}, Projection dimension: {config.projection_dim}")
    print(f"Max points per structure: {config.max_points}")
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}, Weight decay: {config.weight_decay}")
    print(f"Output directory: {os.path.join(config.contrastive_output_dir, 'multimodal')}")
    print(f"Device: {config.device}")
    print("==============================")
    
    # Run the training script as a module
    subprocess.call([sys.executable, "examples/train_multimodal.py"])

def main():
    # Parse command line arguments for this launcher
    parser = argparse.ArgumentParser(description="Synapse multimodal model launcher")
    parser.add_argument('action', choices=['visualize', 'train'], help='Action to perform')
    
    # Add all config parameters as optional arguments to override defaults
    config.parse_args()
    
    # Parse the action argument separately
    action_parser = argparse.ArgumentParser()
    action_parser.add_argument('action', choices=['visualize', 'train'])
    action_args, _ = action_parser.parse_known_args()
    
    # Run the appropriate action
    if action_args.action == 'visualize':
        visualize_data()
    elif action_args.action == 'train':
        train_model()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 