# Synapse Analysis Training in Google Colab

This document provides instructions for setting up and running the synapse analysis training pipeline in Google Colab.

## Getting Started

1. Open the included `colab_training.ipynb` notebook in Google Colab.
2. Follow the step-by-step instructions in the notebook.

## Data Preparation

The project requires specific data formats:

1. Raw synapse data (grayscale EM images) in `/content/raw/`
2. Segmentation data in `/content/seg/`
3. Additional mask data (vesicles, clefts, etc.)

You can download these from the provided Google Drive links in the notebook.

## Training Process

The training workflow consists of two main steps:

1. **Extract Point Clouds**: Process the raw data to extract point clouds representing 3D structures.
2. **Train the Model**: Train a contrastive learning model using the extracted point clouds.

## Troubleshooting

Common issues and solutions:

1. **Import Errors**: The notebook automatically fixes imports by adding the project root to the Python path.
2. **Missing Files**: Ensure all data directories are correctly set up.
3. **GPU Memory Errors**: Reduce batch size if you encounter CUDA out of memory errors.

## Advanced Configuration

The training script supports several optional parameters:

- `--use_dual_encoder`: Train with both texture and shape encoders
- `--use_scheduler`: Enable learning rate scheduling
- `--clip_grad`: Apply gradient clipping
- `--batch_size`: Adjust based on available GPU memory
- `--epochs`: Control training duration

## Outputs

After training, you'll find:

1. Model checkpoints saved in the output directory
2. Training loss history as a numpy file
3. The final trained model ready for inference

## Downloading Results

The notebook includes a step to download your trained model as a zip file at the end of the training process. 