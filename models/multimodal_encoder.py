import torch
import torch.nn as nn
import torch.nn.functional as F

from models.contrastive_model import Conv3DEncoder, PointCloudEncoder, ProjectionHead

class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder that supports separate point clouds for each structure type.
    Combines texture (CNN) and shape (point cloud) information from multiple structure types.
    """
    
    def __init__(self, in_channels=1, encoder_dim=256, projection_dim=128, max_points=512):
        """
        Initialize the multi-modal encoder.
        
        Args:
            in_channels: Number of input channels for texture (default: 1 for grayscale)
            encoder_dim: Dimension of each encoder output (default: 256)
            projection_dim: Dimension of projection head output (default: 128)
            max_points: Maximum number of points for point cloud encoder (default: 512)
        """
        super().__init__()
        
        # Texture encoder (CNN)
        self.texture_encoder = Conv3DEncoder(in_channels=in_channels, feature_dim=encoder_dim)
        
        # Shape encoders for each structure type (point cloud)
        self.cleft_encoder = PointCloudEncoder(feature_dim=encoder_dim, max_points=max_points)
        self.vesicle_encoder = PointCloudEncoder(feature_dim=encoder_dim, max_points=max_points)
        self.presynapse_encoder = PointCloudEncoder(feature_dim=encoder_dim, max_points=max_points)
        
        # Fusion network to combine all features
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim * 4, encoder_dim * 2),  # 4 = texture + 3 structure types
            nn.LayerNorm(encoder_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.LayerNorm(encoder_dim),
            nn.ReLU(inplace=True)
        )
        
        # Projection head for contrastive learning
        self.projection_head = ProjectionHead(input_dim=encoder_dim, output_dim=projection_dim)
        
    def forward(self, texture_input, point_clouds=None, return_features=False):
        """
        Forward pass through the model.
        
        Args:
            texture_input: Texture input tensor of shape (N, C, D, H, W) or (N, D, C, H, W)
            point_clouds: Dictionary of point clouds for each structure type
                          {'cleft': tensor, 'vesicle': tensor, 'presynapse': tensor}
                          If None, zeros will be used
            return_features: Whether to return features before projection
            
        Returns:
            If return_features is True: (fused_features, projections)
            Otherwise: projections
        """
        # Get texture features
        texture_features = self.texture_encoder(texture_input)
        batch_size = texture_features.size(0)
        
        # Initialize point cloud features with zeros if not provided
        if point_clouds is None:
            point_clouds = {}
            
        # Process each point cloud type
        # For any missing point cloud, use zeros
        device = texture_features.device
        
        # Cleft features
        if 'cleft' in point_clouds and point_clouds['cleft'] is not None:
            cleft_features = self.cleft_encoder(point_clouds['cleft'])
        else:
            cleft_features = torch.zeros_like(texture_features)
            
        # Vesicle features
        if 'vesicle' in point_clouds and point_clouds['vesicle'] is not None:
            vesicle_features = self.vesicle_encoder(point_clouds['vesicle'])
        else:
            vesicle_features = torch.zeros_like(texture_features)
            
        # Presynapse features
        if 'presynapse' in point_clouds and point_clouds['presynapse'] is not None:
            presynapse_features = self.presynapse_encoder(point_clouds['presynapse'])
        else:
            presynapse_features = torch.zeros_like(texture_features)
        
        # Concatenate all features
        all_features = torch.cat([
            texture_features,
            cleft_features,
            vesicle_features,
            presynapse_features
        ], dim=1)
        
        # Fuse features
        fused_features = self.fusion(all_features)
        
        # Project features
        projections = self.projection_head(fused_features)
        
        if return_features:
            return fused_features, projections
        else:
            return projections
    
    def get_features(self, texture_input, point_clouds=None):
        """
        Extract fused features without projection.
        
        Args:
            texture_input: Texture input tensor
            point_clouds: Dictionary of point clouds for each structure type
            
        Returns:
            Fused features
        """
        with torch.no_grad():
            # Get texture features
            texture_features = self.texture_encoder(texture_input)
            
            # Initialize point cloud features with zeros if not provided
            if point_clouds is None:
                point_clouds = {}
                
            # Process each point cloud type
            # For any missing point cloud, use zeros
            device = texture_features.device
            
            # Cleft features
            if 'cleft' in point_clouds and point_clouds['cleft'] is not None:
                cleft_features = self.cleft_encoder(point_clouds['cleft'])
            else:
                cleft_features = torch.zeros_like(texture_features)
                
            # Vesicle features
            if 'vesicle' in point_clouds and point_clouds['vesicle'] is not None:
                vesicle_features = self.vesicle_encoder(point_clouds['vesicle'])
            else:
                vesicle_features = torch.zeros_like(texture_features)
                
            # Presynapse features
            if 'presynapse' in point_clouds and point_clouds['presynapse'] is not None:
                presynapse_features = self.presynapse_encoder(point_clouds['presynapse'])
            else:
                presynapse_features = torch.zeros_like(texture_features)
            
            # Concatenate all features
            all_features = torch.cat([
                texture_features,
                cleft_features,
                vesicle_features,
                presynapse_features
            ], dim=1)
            
            # Fuse features
            fused_features = self.fusion(all_features)
        
        return fused_features 