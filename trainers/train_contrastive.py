#!/usr/bin/env python3
"""
Train a contrastive learning model using extracted point clouds.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import random

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from data.dataloader import ContrastiveAugmentationProcessor
from utils.config import config


class PointCloudDataset(Dataset):
    """Dataset for contrastive learning with pre-extracted point clouds"""
    
    def __init__(self, point_clouds, processor):
        """
        Initialize the dataset with pre-extracted point clouds.
        
        Args:
            point_clouds: Dictionary mapping synapse IDs to extracted data
            processor: Processor to convert raw cubes to tensors
        """
        self.point_clouds = list(point_clouds.items())
        self.processor = processor
        
    def __len__(self):
        return len(self.point_clouds)
    
    def __getitem__(self, idx):
        synapse_id, data = self.point_clouds[idx]
        
        # Get the raw cube
        raw_cube = data['raw_cube']
        
        # Use process_raw_volume with augmentation instead of __call__
        pixel_values_aug1 = self.processor.process_raw_volume(raw_cube, apply_augmentation=True)
        pixel_values_aug2 = self.processor.process_raw_volume(raw_cube, apply_augmentation=True)
        
        # Get bbox_name and syn_id from synapse_id
        if '_' in synapse_id:
            bbox_name, syn_id = synapse_id.split('_', 1)
        else:
            bbox_name = 'unknown'
            syn_id = synapse_id
        
        return {
            'pixel_values_aug1': pixel_values_aug1,
            'pixel_values_aug2': pixel_values_aug2,
            'bbox_name': bbox_name,
            'syn_id': syn_id
        }


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper.
    """
    def __init__(self, temperature=0.5, epsilon=1e-8):
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def forward(self, z_i, z_j):
        """
        Compute loss between two batches of augmented samples.
        
        Args:
            z_i: First batch of projections, shape (batch_size, proj_dim)
            z_j: Second batch of projections, shape (batch_size, proj_dim)
            
        Returns:
            Loss value
        """
        # Concatenate both batches
        batch_size = z_i.shape[0]
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.exp(torch.mm(representations, representations.t()) / self.temperature)
        
        # Create mask to filter out self-similarities
        mask = torch.eye(2 * batch_size, device=similarity_matrix.device)
        mask = 1 - mask  # Flip to keep non-diagonal elements
        
        # Filter out self-similarities
        similarity_matrix = similarity_matrix * mask
        
        # Compute NT-Xent loss
        z_i_norm = torch.sum(similarity_matrix[:batch_size], dim=1)
        z_j_norm = torch.sum(similarity_matrix[batch_size:], dim=1)
        
        # Labels are all diagonal elements (positive pairs)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        # Similarity between augmented pairs (i and j)
        sim_i_j = torch.diag(similarity_matrix[:batch_size, batch_size:])
        sim_j_i = torch.diag(similarity_matrix[batch_size:, :batch_size])
        
        # Compute loss
        loss_i = -torch.log(sim_i_j / (z_i_norm + self.epsilon))
        loss_j = -torch.log(sim_j_i / (z_j_norm + self.epsilon))
        
        # Total loss
        loss = torch.mean(loss_i) + torch.mean(loss_j)
        
        return loss


def train_contrastive_model(model, point_clouds, batch_size=8, epochs=10, learning_rate=1e-4, 
                          model_save_path='results/models', model_save_prefix='contrastive_model', 
                          device='cuda'):
    """
    Train a contrastive learning model using pre-extracted point clouds.
    
    Args:
        model: The contrastive model to train
        point_clouds: Dictionary mapping synapse IDs to extracted data
        batch_size: Batch size for training
        epochs: Number of training epochs
        learning_rate: Learning rate
        model_save_path: Directory to save models
        model_save_prefix: Prefix for saved model files
        device: Device to run training on
    """
    print(f"Training contrastive model with {len(point_clouds)} point clouds")
    print(f"Batch size: {batch_size}, Epochs: {epochs}, Learning rate: {learning_rate}")
    
    # Create processor for raw data
    processor = ContrastiveAugmentationProcessor(
        size=(config.subvol_size, config.subvol_size),
        augmentation_strength='medium'
    )
    
    # Create dataset and dataloader
    dataset = PointCloudDataset(point_clouds, processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4 if device == 'cuda' else 0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Loss function and optimizer
    criterion = NTXentLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create model save directory
    os.makedirs(model_save_path, exist_ok=True)
    
    # Training loop
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        
        with tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for step, batch in enumerate(pbar):
                # Get the two augmented views
                aug1 = batch['pixel_values_aug1'].to(device)
                aug2 = batch['pixel_values_aug2'].to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass - compute projections
                z_i = model(aug1)
                z_j = model(aug2)
                
                # Compute loss
                loss = criterion(z_i, z_j)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (step + 1)})
        
        # Save model checkpoint
        if epoch % 5 == 0 or epoch == epochs:
            model_save_file = os.path.join(model_save_path, f"{model_save_prefix}_epoch_{epoch}.pth")
            encoder_save_file = os.path.join(model_save_path, f"encoder_epoch_{epoch}.pth")
            
            # Save the complete model
            torch.save(model.state_dict(), model_save_file)
            
            # Also save just the encoder for clustering/inference
            torch.save(model.encoder.state_dict(), encoder_save_file)
            
            print(f"Saved model checkpoint to {model_save_file}")
            print(f"Saved encoder checkpoint to {encoder_save_file}")
        
        # Print epoch statistics
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")
    
    print("Training completed")
    
    return model 