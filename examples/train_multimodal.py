#!/usr/bin/env python
"""
Example script to train a MultiModalEncoder model that uses separate point clouds
for each structure type (cleft, vesicle, presynapse) alongside the raw texture data.
Uses the configuration from utils.config for all paths and parameters.
"""

import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure the project root is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.config import config
from models.multimodal_encoder import MultiModalEncoder
from models.contrastive_model import nt_xent_loss
from data.dataset import ContrastiveAugmentedLoader
from data.dataloader import ContrastiveAugmentationProcessor


def get_dataloader(config):
    """Initialize and return a dataloader for multimodal training."""
    print("Loading data...")
    
    # Initialize the data loader helper with point cloud extraction enabled
    data_loader_helper = ContrastiveAugmentedLoader(
        config,
        augmentation_strength=config.augmentation_strength,
        use_point_cloud=True,
        max_points=config.max_points
    )
    
    # Create dataset and dataloader
    dataset, _ = data_loader_helper.create_dataset(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Use 0 to disable multiprocessing
    )

    # Create custom DataLoader with our desired configuration
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 to disable multiprocessing
        pin_memory=True
    )
    
    print(f"Data loaded. Dataset size: {len(dataset)}")
    return dataloader


def train(config):
    """Train the MultiModalEncoder model."""
    # Determine device
    device = torch.device(config.device if hasattr(config, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare Dataloader
    dataloader = get_dataloader(config)

    # Initialize Model
    model = MultiModalEncoder(
        in_channels=config.in_channels,
        encoder_dim=config.encoder_dim,
        projection_dim=config.projection_dim,
        max_points=config.max_points
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Create output directory
    # Use a subdirectory of the contrastive_output_dir for multimodal models
    output_dir = os.path.join(config.contrastive_output_dir, 'multimodal')
    os.makedirs(output_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch in progress_bar:
            try:
                # Get texture input for both views
                view1 = batch['pixel_values_aug1'].to(device)
                view2 = batch['pixel_values_aug2'].to(device)
                
                # Get separate point clouds if available
                point_clouds = None
                if 'separate_point_clouds' in batch:
                    separate_point_clouds = batch['separate_point_clouds']
                    point_clouds = {}
                    
                    # Move each point cloud to device
                    for key in separate_point_clouds:
                        if separate_point_clouds[key] is not None:
                            point_clouds[key] = separate_point_clouds[key].to(device)
            except Exception as e:
                print(f"Error unpacking batch: {e}")
                if isinstance(batch, dict):
                    print(f"Batch keys: {list(batch.keys())}")
                continue

            optimizer.zero_grad()

            # Get projections
            z1 = model(view1, point_clouds)
            z2 = model(view2, point_clouds)  # Use same point clouds for both views

            # Calculate loss
            loss = nt_xent_loss(z1, z2, temperature=config.temperature)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{config.epochs}, Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        if (epoch + 1) % config.save_every_epochs == 0 or (epoch + 1) == config.epochs:
            # Save whole model
            model_path = os.path.join(output_dir, f'multimodal_model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # Save individual encoders
            texture_path = os.path.join(output_dir, f'texture_encoder_epoch_{epoch+1}.pth')
            torch.save(model.texture_encoder.state_dict(), texture_path)
            
            cleft_path = os.path.join(output_dir, f'cleft_encoder_epoch_{epoch+1}.pth')
            torch.save(model.cleft_encoder.state_dict(), cleft_path)
            
            vesicle_path = os.path.join(output_dir, f'vesicle_encoder_epoch_{epoch+1}.pth')
            torch.save(model.vesicle_encoder.state_dict(), vesicle_path)
            
            presynapse_path = os.path.join(output_dir, f'presynapse_encoder_epoch_{epoch+1}.pth')
            torch.save(model.presynapse_encoder.state_dict(), presynapse_path)

    print("Training finished.")


if __name__ == "__main__":
    # Parse command-line arguments to update config
    config.parse_args()
    
    # Add 'max_points' to config if not already present
    if not hasattr(config, 'max_points'):
        config.max_points = 512  # Default value
    
    # Add 'device' to config if not already present
    if not hasattr(config, 'device'):
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Print run configuration
    print("=== Training MultiModalEncoder with Separate Point Clouds ===")
    print(f"Using raw texture data + 3 separate point clouds (cleft, vesicle, presynapse)")
    print(f"Max points per structure: {config.max_points}")
    print(f"Raw data: {config.raw_base_dir}")
    print(f"Segmentation data: {config.seg_base_dir}")
    print(f"Additional mask data: {config.add_mask_base_dir}")
    print(f"Synapse coordinates: {config.excel_file}")
    print(f"Encoder dimension: {config.encoder_dim}, Projection dimension: {config.projection_dim}")
    print(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}, Weight decay: {config.weight_decay}")
    print(f"Output directory: {os.path.join(config.contrastive_output_dir, 'multimodal')}")
    print("==============================")
    
    # Train the model
    train(config) 