#!/usr/bin/env python3
"""
Consolidated script to extract point clouds from synapse dataset.
This script uses the standard implementation from data/dataloader.py to ensure consistency.
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Handle imports more robustly
try:
    from data.dataloader import SynapseDataLoader
except ImportError as e:
    print(f"Error importing SynapseDataLoader: {e}")
    print("Make sure you're running this script from the project root directory")
    print("Current working directory:", os.getcwd())
    print("Files in current directory:", os.listdir("."))
    print("Files in data directory (if it exists):", 
          os.listdir("data") if os.path.exists("data") else "data directory not found")
    sys.exit(1)

from models.contrastive_model import PointCloudEncoder


def load_synapse_metadata(excel_file, bbox_name=None, max_samples=None):
    """
    Load synapse metadata from Excel file.
    
    Args:
        excel_file: Path to Excel file with synapse coordinates
        bbox_name: Optional bounding box name to filter by
        max_samples: Maximum number of samples to load (for testing)
        
    Returns:
        DataFrame with synapse coordinates
    """
    try:
        synapse_df = pd.read_excel(excel_file)
        print(f"Loaded {len(synapse_df)} synapses from {excel_file}")
        print(f"Available columns: {synapse_df.columns.tolist()}")
        
        # Filter by bounding box if specified
        if bbox_name is not None:
            if 'bbox_name' in synapse_df.columns:
                synapse_df = synapse_df[synapse_df['bbox_name'] == bbox_name].reset_index(drop=True)
                print(f"Filtered to {len(synapse_df)} synapses in bounding box {bbox_name}")
            else:
                print(f"Warning: Cannot filter by 'bbox_name' as it's not in the columns. Will process all synapses.")
        
        # Limit number of samples if specified
        if max_samples is not None and len(synapse_df) > max_samples:
            synapse_df = synapse_df.sample(max_samples, random_state=42).reset_index(drop=True)
            print(f"Sampled {len(synapse_df)} synapses for testing")
        
        return synapse_df
    
    except Exception as e:
        print(f"Error loading synapse metadata: {e}")
        return pd.DataFrame()


def extract_point_clouds(raw_base_dir, seg_base_dir, add_mask_base_dir, synapse_df, 
                         max_points=512, subvol_size=80, save_interval=10, data_loader=None):
    """
    Extract point clouds for each synapse in the dataset using the standard
    implementation from data/dataloader.py.
    
    Args:
        raw_base_dir: Directory containing raw image data
        seg_base_dir: Directory containing segmentation data
        add_mask_base_dir: Directory containing additional mask data
        synapse_df: DataFrame with synapse information
        max_points: Maximum number of points to sample from each mask
        subvol_size: Size of subvolume cube
        save_interval: How often to save intermediate results
        data_loader: Optional initialized data loader
        
    Returns:
        Dictionary mapping synapse IDs to point clouds
    """
    # Initialize data loader with correct parameters
    if data_loader is None:
        data_loader = SynapseDataLoader(
            raw_base_dir=raw_base_dir, 
            seg_base_dir=seg_base_dir, 
            add_mask_base_dir=add_mask_base_dir
        )
    
    print(f"DEBUG: Initialized SynapseDataLoader with:")
    print(f"  raw_base_dir: {raw_base_dir}")
    print(f"  seg_base_dir: {seg_base_dir}")
    print(f"  add_mask_base_dir: {add_mask_base_dir}")
    
    # Import the processor that handles point cloud extraction
    from data.dataloader import ContrastiveAugmentationProcessor
    processor = ContrastiveAugmentationProcessor(
        size=(subvol_size, subvol_size),
        use_point_cloud=True,
        max_points=max_points
    )
    
    # Dictionary to store point clouds
    point_clouds = {}
    
    # Check if 'bbox_name' column exists, if not, assign a default
    if 'bbox_name' not in synapse_df.columns:
        print("'bbox_name' column not found in Excel file. Using default bbox name 'bbox1'")
        # Use 'bbox1' as the default name since that matches the Excel file name
        default_bbox = 'bbox1'
        synapse_df['bbox_name'] = default_bbox
    
    # Process each bounding box
    unique_bboxes = synapse_df['bbox_name'].unique()
    for bbox_name in unique_bboxes:
        print(f"\nProcessing bounding box: {bbox_name}")
        
        # Debug: Check directories and files explicitly
        raw_dir = os.path.join(raw_base_dir, bbox_name)
        seg_dir = os.path.join(seg_base_dir, bbox_name)
        add_mask_dir = os.path.join(add_mask_base_dir, bbox_name)
        
        # Check if these directories exist
        print(f"Checking directories for bbox {bbox_name}:")
        print(f"  raw_dir: {raw_dir} (exists: {os.path.isdir(raw_dir)})")
        print(f"  seg_dir: {seg_dir} (exists: {os.path.isdir(seg_dir)})")
        print(f"  add_mask_dir: {add_mask_dir} (exists: {os.path.isdir(add_mask_dir)})")
        
        import glob
        raw_tif_files = sorted(glob.glob(os.path.join(raw_dir, 'slice_*.tif')))
        seg_tif_files = sorted(glob.glob(os.path.join(seg_dir, 'slice_*.tif')))
        add_mask_tif_files = sorted(glob.glob(os.path.join(add_mask_dir, 'slice_*.tif')))
        
        print(f"TIF files found:")
        print(f"  raw: {len(raw_tif_files)} files")
        print(f"  seg: {len(seg_tif_files)} files")
        print(f"  add_mask: {len(add_mask_tif_files)} files")
        
        if len(raw_tif_files) == 0 or len(seg_tif_files) == 0 or len(add_mask_tif_files) == 0:
            # If no files found with current bbox_name, try alternate format
            # Some datasets use "bbox_1" format instead of "bbox1"
            alt_bbox_name = f"bbox_{bbox_name[4:]}" if bbox_name.startswith("bbox") else bbox_name
            print(f"Trying alternate bbox format: {alt_bbox_name}")
            
            alt_raw_dir = os.path.join(raw_base_dir, alt_bbox_name)
            alt_seg_dir = os.path.join(seg_base_dir, alt_bbox_name)
            alt_add_mask_dir = os.path.join(add_mask_base_dir, alt_bbox_name)
            
            print(f"  alt_raw_dir: {alt_raw_dir} (exists: {os.path.isdir(alt_raw_dir)})")
            print(f"  alt_seg_dir: {alt_seg_dir} (exists: {os.path.isdir(alt_seg_dir)})")
            print(f"  alt_add_mask_dir: {alt_add_mask_dir} (exists: {os.path.isdir(alt_add_mask_dir)})")
            
            alt_raw_tif_files = sorted(glob.glob(os.path.join(alt_raw_dir, 'slice_*.tif')))
            alt_seg_tif_files = sorted(glob.glob(os.path.join(alt_seg_dir, 'slice_*.tif')))
            alt_add_mask_tif_files = sorted(glob.glob(os.path.join(alt_add_mask_dir, 'slice_*.tif')))
            
            print(f"Alternate format TIF files found:")
            print(f"  raw: {len(alt_raw_tif_files)} files")
            print(f"  seg: {len(alt_seg_tif_files)} files")
            print(f"  add_mask: {len(alt_add_mask_tif_files)} files")
        
        # Now proceed with loading volumes
        raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
        if raw_vol is None:
            print(f"Failed to load volumes for {bbox_name}")
            continue
            
        print(f"Loaded volumes: raw={raw_vol.shape}, seg={seg_vol.shape}, add_mask={add_mask_vol.shape if add_mask_vol is not None else None}")
        
        # Process synapses in this bounding box
        bbox_df = synapse_df[synapse_df['bbox_name'] == bbox_name].reset_index(drop=True)
        print(f"Processing {len(bbox_df)} synapses in {bbox_name}")
        
        for idx, syn_info in tqdm(bbox_df.iterrows(), total=len(bbox_df)):
            # Extract synapse ID
            syn_id = str(syn_info.get('Var1', idx))
            
            # Extract coordinates
            central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
            side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
            side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))
            
            try:
                # Extract raw and mask cubes using data loader's method
                raw_cube, mask_cube = data_loader.extract_raw_and_mask_cubes(
                    raw_vol=raw_vol,
                    seg_vol=seg_vol,
                    add_mask_vol=add_mask_vol,
                    central_coord=central_coord,
                    side1_coord=side1_coord,
                    side2_coord=side2_coord,
                    subvolume_size=subvol_size,
                    bbox_name=bbox_name,
                )
                
                # Extract separate point clouds using the processor
                # This ensures we use the same max_points setting throughout
                from models.contrastive_model import PointCloudEncoder
                
                # Create binary masks for each structure (we skip vesicles)
                if mask_cube is not None:
                    # Create binary mask for cleft (label 1)
                    cleft_mask = (mask_cube == 1)
                    if np.any(cleft_mask):
                        cleft_tensor = torch.from_numpy(cleft_mask.astype(np.float32))
                        cleft_points = PointCloudEncoder.mask_to_point_cloud(cleft_tensor, num_samples=max_points)
                        if isinstance(cleft_points, torch.Tensor):
                            cleft_points = cleft_points.cpu().numpy()
                    else:
                        cleft_points = None
                    
                    # Create binary mask for presynapse (label 3)
                    presynapse_mask = (mask_cube == 3)
                    if np.any(presynapse_mask):
                        presynapse_tensor = torch.from_numpy(presynapse_mask.astype(np.float32))
                        presynapse_points = PointCloudEncoder.mask_to_point_cloud(presynapse_tensor, num_samples=max_points)
                        if isinstance(presynapse_points, torch.Tensor):
                            presynapse_points = presynapse_points.cpu().numpy()
                    else:
                        presynapse_points = None
                else:
                    cleft_points = None
                    presynapse_points = None
                
                # Log extraction results
                if cleft_points is not None:
                    print(f"  Extracted cleft point cloud for synapse {syn_id}: {cleft_points.shape}")
                if presynapse_points is not None:
                    print(f"  Extracted presynapse point cloud for synapse {syn_id}: {presynapse_points.shape}")
                
                # Store point clouds in dictionary
                synapse_data = {
                    'bbox_name': bbox_name,
                    'central_coord': central_coord,
                    'side1_coord': side1_coord,
                    'side2_coord': side2_coord,
                    'subvolume_size': subvol_size,
                    'raw_cube': raw_cube,  # Store raw cube for feature extraction
                    'cleft_points': cleft_points,
                    'presynapse_points': presynapse_points
                }
                
                # Add to the dictionary
                point_clouds[f"{bbox_name}_{syn_id}"] = synapse_data
                print(f"  Extracted point clouds for synapse {syn_id} in {bbox_name}")
                
                # Save intermediate results at defined intervals
                if (idx + 1) % save_interval == 0 and len(point_clouds) > 0:
                    print(f"\nSaving intermediate results after processing {idx + 1}/{len(bbox_df)} synapses...")
                    intermediate_path = os.path.join('results/point_clouds', f"point_clouds_intermediate_{bbox_name}_{idx+1}.pkl")
                    os.makedirs(os.path.dirname(intermediate_path), exist_ok=True)
                    with open(intermediate_path, "wb") as f:
                        pickle.dump(point_clouds, f)
                    print(f"Saved intermediate results to {intermediate_path}")
                    
            except Exception as e:
                print(f"  Error extracting point clouds for synapse {syn_id} in {bbox_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\nExtracted point clouds for {len(point_clouds)} synapses")
    return point_clouds


def main():
    parser = argparse.ArgumentParser(description="Extract point clouds from synapse dataset")
    parser.add_argument("--raw_base_dir", type=str, required=True, help="Base directory for raw data")
    parser.add_argument("--seg_base_dir", type=str, required=True, help="Base directory for segmentation data")
    parser.add_argument("--add_mask_base_dir", type=str, required=True, help="Base directory for additional mask data")
    parser.add_argument("--excel_file", type=str, required=True, help="Path to synapse coordinates Excel file")
    parser.add_argument("--output_dir", type=str, default="results/point_clouds", help="Directory to save point clouds")
    parser.add_argument("--max_points", type=int, default=512, help="Maximum number of points per structure type")
    parser.add_argument("--subvol_size", type=int, default=80, help="Size of subvolume cube")
    parser.add_argument("--bbox_name", type=str, default=None, help="Optional bounding box name to use")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for faster testing)")
    parser.add_argument("--save_interval", type=int, default=5, help="Save intermediate results every N samples")
    parser.add_argument("--direct_folder", action="store_true", help="Use paths directly without bbox subfolder")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load synapse metadata
    synapse_df = load_synapse_metadata(args.excel_file, args.bbox_name, args.max_samples)
    if len(synapse_df) == 0:
        print("No synapse metadata found. Exiting.")
        return
    
    # If direct_folder is used, modify the logic to use base paths directly
    if args.direct_folder:
        print("Using direct folder mode - skipping bbox subfolder lookup")
        # Verify TIF files in the base directories
        import glob
        raw_tif_files = sorted(glob.glob(os.path.join(args.raw_base_dir, 'slice_*.tif')))
        seg_tif_files = sorted(glob.glob(os.path.join(args.seg_base_dir, 'slice_*.tif')))
        add_mask_tif_files = sorted(glob.glob(os.path.join(args.add_mask_base_dir, 'slice_*.tif')))
        
        print(f"TIF files found in direct folders:")
        print(f"  raw: {len(raw_tif_files)} files")
        print(f"  seg: {len(seg_tif_files)} files")
        print(f"  add_mask: {len(add_mask_tif_files)} files")
        
        if len(raw_tif_files) == 0 or len(seg_tif_files) == 0 or len(add_mask_tif_files) == 0:
            print("ERROR: TIF files not found in direct folders. Please check your paths.")
            return
        
        # Create a custom DataLoader for direct folder access
        # Import needed modules
        import imageio.v3 as iio
        
        class DirectFolderDataLoader(SynapseDataLoader):
            """A modified SynapseDataLoader that loads files directly from the base directories"""
            
            def load_volumes(self, bbox_name=None):
                """Load volumes directly from base directories, ignoring bbox_name"""
                try:
                    # Find all TIF files
                    raw_tif_files = sorted(glob.glob(os.path.join(self.raw_base_dir, 'slice_*.tif')))
                    seg_tif_files = sorted(glob.glob(os.path.join(self.seg_base_dir, 'slice_*.tif')))
                    add_mask_tif_files = sorted(glob.glob(os.path.join(self.add_mask_base_dir, 'slice_*.tif')))
                    
                    # Load raw volume and convert to grayscale if needed
                    raw_slices = []
                    for f in raw_tif_files:
                        img = iio.imread(f)
                        # Check if the image has multiple channels (RGB)
                        if len(img.shape) > 2 and img.shape[2] > 1:
                            # Convert RGB to grayscale using luminosity method
                            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                        raw_slices.append(img)
                    raw_vol = np.stack(raw_slices, axis=0)
                    
                    # Load segmentation volume
                    seg_slices = []
                    for f in seg_tif_files:
                        img = iio.imread(f)
                        if len(img.shape) > 2 and img.shape[2] > 1:
                            img = img[..., 0]
                        seg_slices.append(img.astype(np.uint32))
                    seg_vol = np.stack(seg_slices, axis=0)
                    
                    # Load additional mask volume
                    add_mask_slices = []
                    for f in add_mask_tif_files:
                        img = iio.imread(f)
                        if len(img.shape) > 2 and img.shape[2] > 1:
                            img = img[..., 0]
                        add_mask_slices.append(img.astype(np.uint32))
                    add_mask_vol = np.stack(add_mask_slices, axis=0)
                    
                    print(f"Loaded volumes: raw={raw_vol.shape}, seg={seg_vol.shape}, add_mask={add_mask_vol.shape}")
                    return raw_vol, seg_vol, add_mask_vol
                    
                except Exception as e:
                    print(f"Error loading volumes from direct folders: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, None, None
        
        # Initialize our custom dataloader
        data_loader = DirectFolderDataLoader(
            raw_base_dir=args.raw_base_dir,
            seg_base_dir=args.seg_base_dir,
            add_mask_base_dir=args.add_mask_base_dir
        )
    else:
        # Use the standard dataloader with bbox subfolders
        data_loader = SynapseDataLoader(
            raw_base_dir=args.raw_base_dir,
            seg_base_dir=args.seg_base_dir,
            add_mask_base_dir=args.add_mask_base_dir
        )
    
    # Extract point clouds
    point_clouds = extract_point_clouds(
        args.raw_base_dir,
        args.seg_base_dir,
        args.add_mask_base_dir,
        synapse_df,
        args.max_points,
        args.subvol_size,
        args.save_interval,
        data_loader=data_loader  # Pass the initialized data_loader
    )
    
    # Save point clouds to disk
    output_path = os.path.join(args.output_dir, "all_point_clouds.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(point_clouds, f)
    
    print(f"Saved point clouds to {output_path}")
    
    # Save a metadata CSV for easier browsing
    metadata = []
    for synapse_id, data in point_clouds.items():
        bbox_name, syn_id = synapse_id.split("_", 1)
        metadata.append({
            'synapse_id': synapse_id,
            'bbox_name': bbox_name,
            'syn_id': syn_id,
            'has_cleft': data['cleft_points'] is not None,
            'has_presynapse': data['presynapse_points'] is not None,
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_path = os.path.join(args.output_dir, "point_clouds_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main() 