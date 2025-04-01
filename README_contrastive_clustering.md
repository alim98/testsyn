# Contrastive Learning and Clustering for Synapse Data

This module provides a complete pipeline for unsupervised contrastive learning and clustering of 3D synapse data. The pipeline includes data loading, augmentation, model training, feature extraction, and clustering.

## Overview

The contrastive learning approach works as follows:

1. **Data Loading**: Load raw 3D volumes and segmentation masks
2. **Data Augmentation**: Generate multiple augmented views of the same data
3. **Contrastive Learning**: Train a 3D CNN to learn features by maximizing agreement between augmented views
4. **Feature Extraction**: Extract features from the trained encoder
5. **Clustering**: Apply clustering algorithms (KMeans, DBSCAN) to group similar synapses
6. **Visualization**: Visualize clustering results using dimensionality reduction (PCA, t-SNE, UMAP)

## Key Components

- **ContrastiveAugmentationProcessor**: Applies various augmentations to 3D data
- **ContrastiveAugmentedDataset**: Dataset class for contrastive learning with augmentation
- **ContrastiveModel**: 3D CNN model with encoder and projection head
- **ContrastiveTrainer**: Training and clustering pipeline

## Installation Requirements

```
pip install torch torchvision numpy pandas matplotlib scikit-learn umap-learn tqdm
```

## Usage

### Quick Start

Run the example script:

```bash
python examples/contrastive_clustering_example.py --num_epochs 50 --batch_size 8
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_epochs` | Number of training epochs | 50 |
| `--batch_size` | Batch size for training | 8 |
| `--learning_rate` | Learning rate | 0.0001 |
| `--encoder_dim` | Dimension of encoder output | 256 |
| `--projection_dim` | Dimension of projection head output | 128 |
| `--augmentation_strength` | Strength of data augmentation (light, medium, strong) | medium |
| `--temperature` | Temperature for NT-Xent loss | 0.1 |
| `--n_clusters` | Number of clusters for KMeans | config value |
| `--dbscan_eps` | Epsilon for DBSCAN | config value |
| `--min_samples` | Min samples for DBSCAN | config value |
| `--output_dir` | Output directory | results/contrastive_clustering |
| `--device` | Device to use (cuda or cpu) | auto-detect |
| `--visualize_method` | Visualization method for clusters (pca, tsne, umap) | tsne |
| `--val_split` | Validation split ratio | 0.1 |

### Advanced Usage

You can use the `ContrastiveTrainer` class directly in your code:

```python
from synapse.trainers.contrastive_trainer import ContrastiveTrainer
from synapse.utils.config import config

# Initialize trainer
trainer = ContrastiveTrainer(
    config=config,
    encoder_dim=256,
    projection_dim=128,
    augmentation_strength='medium',
    learning_rate=0.0001,
    temperature=0.1,
    output_dir='results/contrastive_learning'
)

# Train model
training_history = trainer.train(
    synapse_df=synapse_df,
    batch_size=8,
    num_epochs=100,
    patience=10
)

# Extract features
features_dict = trainer.extract_features(synapse_df)

# Cluster features
kmeans_results = trainer.cluster_features(
    features_dict=features_dict,
    algorithm='kmeans',
    n_clusters=10
)

# Visualize clusters
trainer.visualize_clusters(
    cluster_results=kmeans_results,
    method='tsne'
)
```

## Contrastive Learning Process

The contrastive learning approach follows the SimCLR framework:

1. Each 3D volume is augmented twice (different random augmentations)
2. Both augmented views are encoded using the same encoder network
3. A projection head maps the encodings to a space where contrastive loss is applied
4. The NT-Xent loss maximizes agreement between views of the same volume
5. This trains the encoder to extract meaningful features without labels

## Clustering Methods

Two clustering methods are supported:

### KMeans
- Partitions data into k clusters
- Number of clusters (k) must be specified
- Works well when clusters are roughly spherical and similar in size

### DBSCAN
- Density-based clustering
- Does not require specifying the number of clusters
- Can discover clusters of arbitrary shape
- Parameters:
  - `eps`: Maximum distance between samples in the same neighborhood
  - `min_samples`: Minimum samples in a neighborhood to form a core point

## Output Files

The pipeline generates the following outputs:

- `training_progress.png`: Training and validation loss curves
- `best_model.pt`: Best model checkpoint based on validation loss
- `kmeans_clustering_results.csv`: Results of KMeans clustering
- `dbscan_clustering_results.csv`: Results of DBSCAN clustering
- `clustering_kmeans_tsne_visualization.png`: Visualization of KMeans clusters
- `clustering_dbscan_tsne_visualization.png`: Visualization of DBSCAN clusters
- `clustering_summary.txt`: Summary of clustering results

## Data Augmentation Options

The following augmentations are applied:

- Rotation
- Translation
- Zoom
- Intensity scaling
- Horizontal/vertical flip
- Noise addition

Three strength levels are available:
- `light`: Subtle augmentations
- `medium`: Moderate augmentations
- `strong`: Strong augmentations

## Implementation Details

### 3D CNN Architecture

The encoder is a 3D CNN with the following architecture:
- 4 convolutional blocks with batch normalization and max pooling
- Input shape: (N, 1, 80, 80, 80)
- Feature dimension: 256 by default

### NT-Xent Loss

The normalized temperature-scaled cross entropy loss is used for contrastive learning:
- Positive pairs: Augmented views of the same volume
- Negative pairs: Augmented views of different volumes
- Temperature parameter controls the sharpness of the distribution 