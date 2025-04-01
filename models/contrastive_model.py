import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3DEncoder(nn.Module):
    """3D CNN encoder for volumetric medical data"""
    
    def __init__(self, in_channels=1, feature_dim=256, dropout_rate=0.5):
        """
        Initialize the 3D CNN encoder.
        
        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            feature_dim: Dimension of output feature vector (default: 256)
            dropout_rate: Dropout rate for regularization (default: 0.5)
        """
        super().__init__()
        
        # 3D CNN layers
        self.encoder = nn.Sequential(
            # Layer 1: (N, 1, 80, 80, 80) -> (N, 32, 40, 40, 40)
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Layer 2: (N, 32, 40, 40, 40) -> (N, 64, 20, 20, 20)
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Layer 3: (N, 64, 20, 20, 20) -> (N, 128, 10, 10, 10)
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Layer 4: (N, 128, 10, 10, 10) -> (N, 256, 5, 5, 5)
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        
        # Calculate the flattened size after convolutions
        self.flattened_size = 256 * 5 * 5 * 5
        
        # Dense layers for feature encoding
        self.dense = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (N, C, D, H, W)
            
        Returns:
            Encoded features of shape (N, feature_dim)
        """
        # Check if input x is in the expected format (N, D, C, H, W) and reshape if needed
        # The expected format for Conv3D is (N, C, D, H, W)
        if x.dim() == 5 and x.shape[2] == 1: # Detects (N, D, C, H, W)
             # Permute depth and channel dimensions: (N, D, C, H, W) -> (N, C, D, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            
        # Pass through encoder
        features = self.encoder(x)
        
        # Flatten the features
        flattened = features.view(features.size(0), -1)
        
        # Pass through dense layers
        encoded = self.dense(flattened)
        
        return encoded


class PointCloudEncoder(nn.Module):
    """Point Cloud encoder for segmentation masks"""
    
    def __init__(self, feature_dim=256, max_points=512, dropout_rate=0.5):
        """
        Initialize the point cloud encoder.
        
        Args:
            feature_dim: Dimension of output feature vector (default: 256)
            max_points: Maximum number of points to sample from masks (default: 512)
            dropout_rate: Dropout rate for regularization (default: 0.5)
        """
        super().__init__()
        
        self.max_points = max_points
        
        # PointNet-like architecture
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Dense layers for feature encoding
        self.dense = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the point cloud encoder.
        
        Args:
            x: Input tensor of shape (N, num_points, 3) representing point clouds
            
        Returns:
            Encoded features of shape (N, feature_dim)
        """
        batch_size = x.size(0)
        
        # Transpose to (N, 3, num_points) for 1D convolutions
        x = x.transpose(2, 1)
        
        # Process through convolution layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Max pooling across points (symmetric function)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(batch_size, -1)
        
        # Pass through dense layers
        encoded = self.dense(x)
        
        return encoded
    
    @staticmethod
    def mask_to_point_cloud(mask_cube, num_samples=None):
        """
        Convert a 3D segmentation mask to a point cloud.
        
        Args:
            mask_cube: 3D tensor of shape (H, W, D) or (N, H, W, D) with binary/label values
            num_samples: Number of points to sample (if None, uses all points)
            
        Returns:
            Point cloud tensor of shape (N, num_points, 3)
        """
        # Ensure mask is a torch tensor
        if not isinstance(mask_cube, torch.Tensor):
            mask_cube = torch.tensor(mask_cube, dtype=torch.float32)
        
        # Handle batch dimension if not present
        was_batch = True
        if mask_cube.dim() == 3:
            mask_cube = mask_cube.unsqueeze(0)
            was_batch = False
        
        batch_size = mask_cube.size(0)
        point_clouds = []
        
        for b in range(batch_size):
            # Get coordinates where mask is non-zero
            mask = mask_cube[b] > 0
            if mask.sum() == 0:
                # If mask is empty, create a single zero point
                points = torch.zeros((1, 3), dtype=torch.float32, device=mask_cube.device)
            else:
                coords = torch.nonzero(mask, as_tuple=False).float()
                
                # Sample points if needed
                if num_samples is not None and coords.size(0) > num_samples:
                    # Randomly sample points
                    idx = torch.randperm(coords.size(0))[:num_samples]
                    coords = coords[idx]
                
                # Normalize coordinates to [-1, 1] range
                h, w, d = mask.shape
                coords[:, 0] = 2 * (coords[:, 0] / (h - 1)) - 1
                coords[:, 1] = 2 * (coords[:, 1] / (w - 1)) - 1
                coords[:, 2] = 2 * (coords[:, 2] / (d - 1)) - 1
                
                points = coords
            
            # Ensure we have a consistent number of points
            if num_samples is not None:
                if points.size(0) < num_samples:
                    # Pad with zeros if we have fewer points than requested
                    padding = torch.zeros((num_samples - points.size(0), 3), 
                                         dtype=torch.float32, 
                                         device=mask_cube.device)
                    points = torch.cat([points, padding], dim=0)
            
            point_clouds.append(points)
        
        # Stack point clouds
        point_cloud = torch.stack(point_clouds, dim=0)
        
        # Remove batch dimension if it wasn't originally present
        if not was_batch:
            point_cloud = point_cloud.squeeze(0)
            
        return point_cloud


class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=64):
        """
        Initialize the projection head.
        
        Args:
            input_dim: Dimension of input features (default: 256)
            hidden_dim: Dimension of hidden layer (default: 128)
            output_dim: Dimension of output projections (default: 64)
        """
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the projection head.
        
        Args:
            x: Input features of shape (N, input_dim)
            
        Returns:
            Projected features of shape (N, output_dim)
        """
        return self.projection(x)


class DualEncoderModel(nn.Module):
    """Dual encoder contrastive model with texture and shape paths"""
    
    def __init__(self, in_channels=1, encoder_dim=256, projection_dim=128, max_points=512):
        """
        Initialize the dual encoder contrastive model.
        
        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            encoder_dim: Dimension of encoder output (default: 256)
            projection_dim: Dimension of projection head output (default: 128)
            max_points: Maximum number of points for point cloud encoder (default: 512)
        """
        super().__init__()
        
        # Texture encoder (CNN)
        self.texture_encoder = Conv3DEncoder(in_channels=in_channels, feature_dim=encoder_dim)
        
        # Shape encoder (Point Cloud)
        self.shape_encoder = PointCloudEncoder(feature_dim=encoder_dim, max_points=max_points)
        
        # Fusion layer to combine features
        self.fusion = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.ReLU(inplace=True)
        )
        
        # Projection head
        self.projection_head = ProjectionHead(input_dim=encoder_dim, output_dim=projection_dim)
        
    def forward(self, texture_input, point_cloud=None, return_features=False):
        """
        Forward pass through the model.
        
        Args:
            texture_input: Texture input tensor of shape (N, C, D, H, W) or (N, D, C, H, W)
            point_cloud: Optional point cloud input tensor of shape (N, num_points, 3)
            return_features: Whether to return features before projection
            
        Returns:
            If return_features is True: (features, projections)
            Otherwise: projections
        """
        # Get texture features
        texture_features = self.texture_encoder(texture_input)
        
        # Get shape features
        if point_cloud is not None:
            shape_features = self.shape_encoder(point_cloud)
        else:
            # If no point cloud is provided, use a tensor of zeros
            shape_features = torch.zeros_like(texture_features)
        
        # Concatenate features
        combined_features = torch.cat([texture_features, shape_features], dim=1)
        
        # Fuse features
        fused_features = self.fusion(combined_features)
        
        # Project features
        projections = self.projection_head(fused_features)
        
        if return_features:
            return fused_features, projections
        else:
            return projections
    
    def get_features(self, texture_input, point_cloud=None):
        """
        Extract features without projection.
        
        Args:
            texture_input: Texture input tensor
            point_cloud: Optional point cloud input tensor
            
        Returns:
            Fused features
        """
        with torch.no_grad():
            # Get texture features
            texture_features = self.texture_encoder(texture_input)
            
            # Get shape features
            if point_cloud is not None:
                shape_features = self.shape_encoder(point_cloud)
            else:
                # If no point cloud is provided, use a tensor of zeros
                shape_features = torch.zeros_like(texture_features)
            
            # Concatenate features
            combined_features = torch.cat([texture_features, shape_features], dim=1)
            
            # Fuse features
            fused_features = self.fusion(combined_features)
        
        return fused_features


class ContrastiveModel(nn.Module):
    """Contrastive learning model for 3D medical image data"""
    
    def __init__(self, in_channels=1, encoder_dim=256, projection_dim=128):
        """
        Initialize the contrastive learning model.
        
        Args:
            in_channels: Number of input channels (default: 1 for grayscale)
            encoder_dim: Dimension of encoder output (default: 256)
            projection_dim: Dimension of projection head output (default: 128)
        """
        super().__init__()
        
        self.encoder = Conv3DEncoder(in_channels=in_channels, feature_dim=encoder_dim)
        self.projection_head = ProjectionHead(input_dim=encoder_dim, output_dim=projection_dim)
        
    def forward(self, x: torch.Tensor, return_features=False) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor, expected shape (N, C, D, H, W) or (N, D, C, H, W)
            return_features: Whether to return features before projection
            
        Returns:
            If return_features is True: (features, projections)
            Otherwise: projections
        """
        features = self.encoder(x) # Encoder handles potential permutation
        projections = self.projection_head(features)
        
        if return_features:
            return features, projections
        else:
            return projections
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input without projection.
        
        Args:
            x: Input tensor, expected shape (N, C, D, H, W) or (N, D, C, H, W)
            
        Returns:
            Features of shape (N, encoder_dim)
        """
        with torch.no_grad():
            features = self.encoder(x) # Encoder handles potential permutation
        return features

# Loss functions for contrastive learning

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Normalized temperature-scaled cross entropy loss (NT-Xent) for contrastive learning.
    Implementation inspired by SimCLR.
    
    Args:
        z1: Projected features of the first augmented views (shape: N, projection_dim)
        z2: Projected features of the second augmented views (shape: N, projection_dim)
        temperature: Temperature scaling parameter
        
    Returns:
        NT-Xent loss value (scalar tensor)
    """
    # Concatenate the representations from both views
    batch_size = z1.shape[0]
    representations = torch.cat([z1, z2], dim=0) # Shape: (2N, projection_dim)
    
    # Normalize representations along the feature dimension
    representations = F.normalize(representations, p=2, dim=1)
    
    # Calculate cosine similarity matrix (all pairs)
    # Shape: (2N, 2N)
    similarity_matrix = torch.matmul(representations, representations.T)
    
    # Create labels: positive pairs are at index 0
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)
    
    # Mask for positive pairs: (i, i+batch_size) and (i+batch_size, i)
    # These should be considered similar
    mask = torch.zeros_like(similarity_matrix)
    
    # Set the positive pairs
    for i in range(batch_size):
        mask[i, i + batch_size] = 1.
        mask[i + batch_size, i] = 1.
    
    # Select positive pairs
    positives = similarity_matrix * mask
    
    # We need to select one positive pair per row
    # For each example in the first half, its positive is in the second half
    # For each example in the second half, its positive is in the first half
    pos_per_row = torch.zeros(2 * batch_size, 1, device=z1.device)
    for i in range(batch_size):
        pos_per_row[i, 0] = positives[i, i + batch_size]
        pos_per_row[i + batch_size, 0] = positives[i + batch_size, i]
    
    # Remove diagonal elements (self-similarity) from similarity_matrix
    diag_mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    # Also remove positive pairs from potential negatives
    neg_mask = diag_mask & (~mask.bool())
    
    # Collect all negative similarities
    negatives = similarity_matrix[neg_mask].view(2 * batch_size, -1)
    
    # Combine positives and negatives and scale by temperature
    logits = torch.cat([pos_per_row, negatives], dim=1) / temperature
    
    # Cross entropy loss with positive pair as target (index 0)
    loss = F.cross_entropy(logits, labels)
    
    return loss 