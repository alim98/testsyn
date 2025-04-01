"""
Script to train a contrastive learning model using pre-extracted point clouds.
Enhanced for easier use in Google Colab.
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

# Import necessary modules with error handling
try:
    from models.contrastive_model import ContrastiveModel, DualEncoderModel, nt_xent_loss
    from data.dataloader import Synapse3DProcessor, ContrastiveAugmentationProcessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script from the project root directory")
    print("Current working directory:", os.getcwd())
    sys.exit(1)

class PointCloudDataset(Dataset):
    """Dataset for contrastive learning with pre-extracted point clouds"""
    
    def __init__(self, point_clouds_path, processor=None, use_point_cloud=True, verbose=False):
        """
        Initialize the dataset with pre-extracted point clouds.
        
        Args:
            point_clouds_path: Path to the pickle file with pre-extracted point clouds
            processor: Image processor for the raw cubes
            use_point_cloud: Whether to use point cloud data
            verbose: Whether to print verbose output
        """
        self.use_point_cloud = use_point_cloud
        self.verbose = verbose
        
        # Load the point clouds
        print(f"Loading point clouds from {point_clouds_path}...")
        try:
            with open(point_clouds_path, "rb") as f:
                self.point_clouds = pickle.load(f)
            
            # Get the synapse IDs
            self.synapse_ids = list(self.point_clouds.keys())
            print(f"Loaded {len(self.synapse_ids)} synapses with point clouds")
            
            # Print statistics if verbose
            if verbose:
                cloud_stats = {}
                for syn_id, data in self.point_clouds.items():
                    for key, val in data.items():
                        if key.endswith('_points') and val is not None:
                            if key not in cloud_stats:
                                cloud_stats[key] = []
                            cloud_stats[key].append(val.shape[0] if hasattr(val, 'shape') else 0)
                
                print("Point cloud statistics:")
                for key, counts in cloud_stats.items():
                    if counts:
                        print(f"  {key}: mean={np.mean(counts):.1f}, median={np.median(counts)}, min={np.min(counts)}, max={np.max(counts)}")
        except Exception as e:
            print(f"Error loading point clouds: {e}")
            raise RuntimeError(f"Failed to load point clouds from {point_clouds_path}")
        
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
        
        try:
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
        except Exception as e:
            if self.verbose:
                print(f"Error processing synapse {syn_id}: {e}")
            
            # Return a dummy result in case of error
            dummy_shape = (1, 1, 80, 80)  # D, C, H, W
            return {
                "pixel_values_aug1": torch.zeros(dummy_shape, dtype=torch.float32),
                "pixel_values_aug2": torch.zeros(dummy_shape, dtype=torch.float32),
                "syn_id": syn_id,
                "point_cloud": torch.zeros((1, 3), dtype=torch.float32)
            }

def train(args):
    """
    Train a contrastive learning model with pre-extracted point clouds.
    
    Args:
        args: Command-line arguments
    """
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Create dataset and dataloader
        dataset = PointCloudDataset(
            args.point_clouds_path,
            use_point_cloud=args.use_dual_encoder,
            verbose=getattr(args, 'verbose', False)
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True  # Drop last batch if it's smaller than batch_size
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
        
        # Optional learning rate scheduler
        if hasattr(args, 'use_scheduler') and args.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=args.epochs * len(dataloader), 
                eta_min=args.learning_rate * 0.01
            )
        else:
            scheduler = None
        
        # Track losses for visualization
        all_losses = []
        
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
                try:
                    if args.use_dual_encoder:
                        z1 = model(view1, point_cloud1)
                        z2 = model(view2, point_cloud2)
                    else:
                        z1 = model(view1)
                        z2 = model(view2)
                    
                    # Calculate loss
                    loss = nt_xent_loss(z1, z2, temperature=args.temperature)
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Optional gradient clipping
                if hasattr(args, 'clip_grad') and args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                
                optimizer.step()
                
                # Update scheduler if used
                if scheduler is not None:
                    scheduler.step()
                
                # Track loss
                loss_val = loss.item()
                total_loss += loss_val
                all_losses.append(loss_val)
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss_val)
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{args.epochs} - Average loss: {avg_loss:.4f}")
            
            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save final model
        final_model_path = os.path.join(args.output_dir, "model_final.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")
        
        # Save losses for visualization
        losses_path = os.path.join(args.output_dir, "losses.npy")
        np.save(losses_path, np.array(all_losses))
        print(f"Saved loss history to {losses_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a contrastive learning model with point clouds')
    parser.add_argument('--point_clouds_path', type=str, required=True, help='Path to point clouds pickle file')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for models and logs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--encoder_dim', type=int, default=512, help='Dimension of encoder')
    parser.add_argument('--projection_dim', type=int, default=128, help='Dimension of projection head')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')
    parser.add_argument('--use_dual_encoder', action='store_true', help='Use dual encoder model')
    parser.add_argument('--max_points', type=int, default=1024, help='Maximum points for point cloud')
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    
    args = parser.parse_args()
    
    # Call train function
    train(args)

if __name__ == '__main__':
    main() 