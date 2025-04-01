# Contrastive Learning with Synapse Dataset

This document explains how to use the contrastive learning dataset for analyzing synapse data.

## Overview

The contrastive learning dataset provides separate access to:
1. Raw 3D image data (80x80x80 cubes)
2. Corresponding segmentation masks with multiple classes

This is designed for contrastive learning approaches where you need access to both the original data and segmentation masks as separate tensors.

## Key Components

- `ContrastiveProcessor`: Handles processing of raw and mask volumes
- `ContrastiveSynapseDataset`: Dataset class that returns separate raw and mask data
- `ContrastiveSynapseLoader`: Helper class to simplify loading data and creating datasets

## Usage Example

```python
from synapse.data.dataset import ContrastiveSynapseLoader
from synapse.utils.config import config

# Create loader
loader = ContrastiveSynapseLoader(config)

# Load volume data
vol_data_dict = loader.load_data()

# Load your synapse DataFrame (replace with your actual code)
import pandas as pd
synapse_df = pd.read_excel(config.excel_file)

# Create dataset and dataloader
dataset, dataloader = loader.create_dataset(
    vol_data_dict=vol_data_dict,
    synapse_df=synapse_df,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# Use in training loop
for batch_idx, (raw_imgs, mask_imgs, syn_info, bbox_names) in enumerate(dataloader):
    # raw_imgs shape: [batch_size, num_frames, 1, height, width]
    # mask_imgs shape: [batch_size, num_frames, num_classes, height, width]
    
    # Your contrastive learning code here
    # ...
```

## Data Structure

### Raw Images
- Shape: `[batch_size, num_frames, 1, height, width]`
- Values: Normalized to range [0, 1]
- Type: Float32

### Mask Images
- Shape: `[batch_size, num_frames, num_classes, height, width]`
- Values: One-hot encoded masks (0 or 1)
- Class meanings:
  - Class 0: Background
  - Class 1: Synaptic cleft
  - Class 2: Vesicles
  - Class 3: Mitochondria

## Example Code

See `examples/contrastive_learning_example.py` for a complete working example.

## Configuration

All paths and parameters are configured in `config.py`. The important settings for contrastive learning are:

- `raw_base_dir`: Directory containing raw image data
- `seg_base_dir`: Directory containing segmentation data
- `add_mask_base_dir`: Directory containing additional mask data
- `bbox_name`: List of bounding box names to load
- `subvol_size`: Size of subvolume cube (default 80)
- `num_frames`: Number of frames to return (default 80)

## Implementation Details

The dataset extracts raw cubes and corresponding mask cubes from the original volumes using the following steps:

1. Locate the central coordinate of each synapse
2. Extract a cube around that coordinate
3. Create separate raw and mask cubes
4. Process both cubes with the `ContrastiveProcessor`
5. Return both as separate tensors for contrastive learning

This approach allows you to:
- Train models that learn from both raw data and segmentation information
- Implement contrastive loss functions between image and segmentation spaces
- Develop self-supervised learning approaches that use segmentation as auxiliary information 