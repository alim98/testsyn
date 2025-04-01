# Point Cloud Extraction in Synapse Analysis

This document describes the unified approach to point cloud extraction in the synapse analysis pipeline.

## Overview

Point cloud extraction is a critical part of our analysis pipeline. It transforms 3D mask data into sparse point cloud representations for synaptic structures, which can be used for shape analysis, visualization, and as inputs to our deep learning models.

## Consolidated Approach

We have consolidated our point cloud extraction code to ensure consistency across the pipeline:

1. **Core Implementation**: The primary point cloud extraction logic is contained in `data/dataloader.py`, specifically in the `extract_point_cloud` and `extract_separate_point_clouds` methods of the `SynapseDataLoader` class.

2. **Extraction Script**: A unified script `extract_point_clouds.py` is provided to extract point clouds from synapse datasets, storing them for later use in training and visualization.

3. **Training Compatibility**: The training script `train_with_point_clouds.py` works with the extracted point clouds, enabling both texture-only and dual-encoder (texture+shape) model training.

4. **Analysis Tools**: The clustering script `trainers/cluster_synapses.py` also uses the same SynapseDataLoader for data extraction, ensuring consistency across the analysis pipeline.

## Common Usage

### Extracting Point Clouds

To extract point clouds from a synapse dataset:

```bash
python extract_point_clouds.py \
  --raw_base_dir "path/to/raw/data" \
  --seg_base_dir "path/to/segmentation/data" \
  --add_mask_base_dir "path/to/additional/mask/data" \
  --excel_file "path/to/synapse_metadata.xlsx" \
  --output_dir "results/point_clouds" \
  --max_points 512
```

### Training with Extracted Point Clouds

Once point clouds are extracted, you can train models using:

```bash
python train_with_point_clouds.py \
  --point_clouds_path "results/point_clouds/point_clouds.pkl" \
  --output_dir "results/models" \
  --use_dual_encoder \
  --epochs 10
```

### Clustering Synapse Features

After training a model, you can cluster synapses based on their encoded features:

```bash
python trainers/cluster_synapses.py \
  --model_path "results/models/dual_encoder_model_epoch_10.pth" \
  --n_clusters 3 \
  --algorithm kmeans \
  --output_dir "results/clustering"
```

This script extracts features from each synapse using the trained encoder model, then performs clustering to identify patterns in the data. The clustering results are saved as visualizations and a CSV file in the specified output directory.

## Core Implementation Details

The point cloud extraction logic handles:

- Extracting separate point clouds for cleft and presynapse structures
- Ensuring vesicle structures are intentionally excluded
- Sampling points to a consistent size (configurable via `max_points`)
- Converting mask data to normalized 3D coordinates

The implementation in `data/dataloader.py` uses the `PointCloudEncoder.mask_to_point_cloud` method from `models/contrastive_model.py` to maintain consistency between extraction and model training. 