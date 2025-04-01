# Synapse Multimodal Encoder

This README explains how to use the multimodal encoder for synapse analysis, which combines texture information from 3D volumes with separate point clouds for different structure types (cleft, vesicle, presynapse).

## Overview

The multimodal approach uses:
1. A 3D CNN texture encoder for raw volume data
2. Three separate point cloud encoders (one for each structure type)
3. A fusion network to combine all features
4. A projection head for contrastive learning

## Configuration-based Approach

All parameters and paths are managed through a central configuration file (`utils/config.py`). This provides:
- Default values for all parameters
- Command-line overrides for any parameter
- Consistent configuration across visualization and training

## Running the System

The simplest way to run the system is to use the launcher script:

```bash
# Visualize model inputs with default configuration
python run_synapse_multimodal.py visualize

# Train the model with default configuration
python run_synapse_multimodal.py train
```

You can override any configuration parameter by passing it as a command-line argument:

```bash
# Visualize with custom parameters
python run_synapse_multimodal.py visualize --raw_base_dir /path/to/raw_data --show_plots

# Train with custom parameters
python run_synapse_multimodal.py train --epochs 100 --batch_size 16 --max_points 4096
```

## Data Handling

### Real Data

The system is designed to work with real synapse data, using:
- Raw 3D volumes (80x80x80 voxels) containing texture information
- Segmentation masks containing structure labels:
  - Label 1: Cleft
  - Label 2: Vesicle cloud
  - Label 3: Presynapse/mitochondria

### Coordinates File Formats

The system supports two different formats for synapse coordinates:

1. **Single Excel File**: A single Excel file (`synapse_coordinates.xlsx`) containing coordinates for all synapses with a `bbox_name` column specifying which bounding box each synapse belongs to.

2. **Per-Bounding Box Files**: Individual Excel files named after each bounding box (e.g., `bbox1.xlsx`, `bbox2.xlsx`) located in the same directory as specified in `synapse_coordinates_path`. The system will automatically look for these files if the main coordinates file is not found.

If needed, the system will add a `bbox_name` column to dataframes loaded from per-bounding box files.

### Synthetic Data Visualization

If the real data is not available (missing Excel files or permissions issues), the visualization script automatically generates synthetic data for demonstration purposes. This includes:
- Synthetic 3D raw volumes with gaussian intensity simulating synapses
- Synthetic segmentation masks with the three structure types:
  - Cleft: Thin disc in the middle
  - Vesicles: Small spheres on one side
  - Presynapse: Larger structure on the opposite side

This allows you to test the visualization pipeline even without access to the actual data.

## Configuration Parameters

Key parameters that can be configured include:

### Data Paths
- `raw_base_dir`: Path to raw data
- `seg_base_dir`: Path to segmentation data
- `add_mask_base_dir`: Path to additional mask data
- `synapse_coordinates_path`: Path to synapse coordinates Excel file (or base directory for per-bounding box files)

### Model Parameters
- `max_points`: Maximum number of points per structure type (default: 512)
- `encoder_dim`: Dimension of encoder output (default: 256)
- `projection_dim`: Dimension of projection head output (default: 128)
- `in_channels`: Number of input channels (default: 1)

### Training Parameters
- `epochs`: Number of training epochs (default: 5)
- `batch_size`: Batch size for training (default: 16)
- `learning_rate`: Learning rate (default: 1e-4)
- `weight_decay`: Weight decay (default: 1e-6)
- `temperature`: Temperature for contrastive loss (default: 0.1)
- `augmentation_strength`: Strength of augmentations ('light', 'medium', 'strong', default: 'medium')
- `device`: Device to use ('cuda' or 'cpu', default: 'cuda')

### Visualization Parameters
- `num_samples_to_visualize`: Number of samples to visualize (default: 5)
- `show_plots`: Show plots instead of saving them (default: False)
- `visualization_output_dir`: Directory to save visualizations (default: 'results/visualizations')

## Advanced Usage

You can also run the individual scripts directly:

```bash
# Visualization script
python examples/visualize_model_inputs.py --show_plots --num_samples_to_visualize 10

# Training script
python examples/train_multimodal.py --epochs 200 --learning_rate 5e-5
```

## Output Structure

- Visualizations are saved to `{visualization_output_dir}`
- Model checkpoints are saved to `{contrastive_output_dir}/multimodal`
- Individual encoder checkpoints are also saved separately

## Model Architecture

The multimodal encoder architecture:
- **Texture Encoder**: 3D CNN for processing raw voxel data
- **Structure Encoders**: 
  - Cleft encoder (PointNet-like)
  - Vesicle encoder (PointNet-like)
  - Presynapse encoder (PointNet-like)
- **Fusion Network**: Combines features from all encoders
- **Projection Head**: Projects features to a space suitable for contrastive learning 