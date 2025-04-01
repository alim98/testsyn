"""
Script to train a contrastive learning model using pre-extracted point clouds.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import pickle
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Add the project root to the path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary modules
from models.contrastive_model import ContrastiveModel, DualEncoderModel, nt_xent_loss
from data.dataloader import Synapse3DProcessor, ContrastiveAugmentationProcessor

class PointCloudDataset(Dataset):
    """Dataset for contrastive learning with pre-extracted point clouds"""
    
    def __init__(self, point_clouds_path, processor=None, use_point_cloud=True):
        """
        Initialize the dataset with pre-extracted point clouds.
        
        Args:
            point_clouds_path: Path to the pickle file with pre-extracted point clouds
            processor: Image processor for the raw cubes
            use_point_cloud: Whether to use point cloud data
        """
        self.use_point_cloud = use_point_cloud
        
        # Load the point clouds
        print(f"Loading point clouds from {point_clouds_path}...")
        with open(point_clouds_path, "rb") as f:
            self.point_clouds = pickle.load(f)
        
        # Get the synapse IDs
        self.synapse_ids = list(self.point_clouds.keys())
        print(f"Loaded {len(self.synapse_ids)} synapses with point clouds")
        
        # Initialize the processor if not provided
        if processor is None:
            self.processor = ContrastiveAugmentationProcessor(
                size=(80, 80),
                augmentation_strength='medium',
            )
        else:
            self.processor = processor
    
    def __len__(self):
        return len(self.synapse_ids)
    
    def __getitem__(self, idx):
        """
        Get a synapse with its augmented views and point cloud.
        
        Returns:
            Dictionary with pixel_values_aug1, pixel_values_aug2, point_cloud, etc.
        """
        syn_id = self.synapse_ids[idx]
        data = self.point_clouds[syn_id]
        
        # Get the raw cube
        raw_cube = data['raw_cube']
        
        # Process the raw cube to get two augmented views
        # Convert to tensor and apply augmentations
        processed_data = self.processor(raw_cube, None, return_tensors="pt", generate_pair=True)
        
        result = {
            "pixel_values_aug1": processed_data["pixel_values_aug1"],
            "pixel_values_aug2": processed_data["pixel_values_aug2"],
            "syn_id": syn_id,
        }
        
        # Add point cloud data if requested
        if self.use_point_cloud:
            # Process point cloud data - combine cleft and presynapse if both exist
            cleft_points = data.get('cleft_points')
            presynapse_points = data.get('presynapse_points')
            
            if cleft_points is not None and presynapse_points is not None:
                # If both exist, randomly select one to prevent overfitting
                if torch.rand(1).item() > 0.5:
                    result["point_cloud"] = cleft_points
                else:
                    result["point_cloud"] = presynapse_points
            elif cleft_points is not None:
                result["point_cloud"] = cleft_points
            elif presynapse_points is not None:
                result["point_cloud"] = presynapse_points
            else:
                # Create an empty point cloud if none exists
                result["point_cloud"] = torch.zeros((1, 3), dtype=torch.float32)
        
        return result

def train(args):
    """
    Train a contrastive learning model with pre-extracted point clouds.
    
    Args:
        args: Command-line arguments
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = PointCloudDataset(
        args.point_clouds_path,
        use_point_cloud=args.use_dual_encoder
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    if args.use_dual_encoder:
        print("Using Dual Encoder Model with texture and shape paths")
        model = DualEncoderModel(
            in_channels=1,  # Gray-scale images
            encoder_dim=args.encoder_dim,
            projection_dim=args.projection_dim,
            max_points=args.max_points
        ).to(device)
    else:
        print("Using standard Contrastive Model (texture only)")
        model = ContrastiveModel(
            in_channels=1,  # Gray-scale images
            encoder_dim=args.encoder_dim,
            projection_dim=args.projection_dim
        ).to(device)
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            try:
                # Get the augmented views
                view1 = batch['pixel_values_aug1'].to(device)
                view2 = batch['pixel_values_aug2'].to(device)
                
                # Get point cloud data if available and using dual encoder
                point_cloud1 = None
                point_cloud2 = None
                if args.use_dual_encoder and 'point_cloud' in batch:
                    point_cloud = batch['point_cloud'].to(device)
                    # Use the same point cloud for both views
                    point_cloud1 = point_cloud
                    point_cloud2 = point_cloud
            except Exception as e:
                print(f"Error unpacking batch: {e}")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            if args.use_dual_encoder:
                z1 = model(view1, point_cloud1)
                z2 = model(view2, point_cloud2)
            else:
                z1 = model(view1)
                z2 = model(view2)
            
            # Calculate loss
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % args.save_every_epochs == 0 or (epoch + 1) == args.epochs:
            # Save the model
            if args.use_dual_encoder:
                # For dual encoder, save texture and shape encoders separately
                texture_save_path = os.path.join(args.output_dir, f'texture_encoder_epoch_{epoch+1}.pth')
                shape_save_path = os.path.join(args.output_dir, f'shape_encoder_epoch_{epoch+1}.pth')
                # Save the texture encoder
                torch.save(model.texture_encoder.state_dict(), texture_save_path)
                # Save the shape encoder
                torch.save(model.shape_encoder.state_dict(), shape_save_path)
                # Save the full model too
                full_model_path = os.path.join(args.output_dir, f'dual_encoder_model_epoch_{epoch+1}.pth')
                torch.save(model.state_dict(), full_model_path)
                print(f"Dual encoder model saved to {full_model_path}")
            else:
                # For the standard model, save just the encoder
                save_path = os.path.join(args.output_dir, f'contrastive_encoder_epoch_{epoch+1}.pth')
                torch.save(model.encoder.state_dict(), save_path)
                print(f"Encoder saved to {save_path}")
    
    print("Training finished.")

def main():
    parser = argparse.ArgumentParser(description="Train contrastive model with pre-extracted point clouds")
    parser.add_argument("--point_clouds_path", type=str, required=True, help="Path to pre-extracted point clouds file")
    parser.add_argument("--output_dir", type=str, default="results/contrastive_models", help="Directory to save model outputs")
    parser.add_argument("--use_dual_encoder", action="store_true", help="Use dual encoder with texture and shape paths")
    parser.add_argument("--max_points", type=int, default=512, help="Maximum number of points per structure type")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for optimizer")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for NT-Xent loss")
    parser.add_argument("--encoder_dim", type=int, default=256, help="Dimension of encoder output")
    parser.add_argument("--projection_dim", type=int, default=128, help="Dimension of projection head output")
    parser.add_argument("--save_every_epochs", type=int, default=5, help="Save model every N epochs")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of worker processes for dataloader")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main() 