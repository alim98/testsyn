"""
Script to debug data loading issues with the synapse dataset.
"""
import os
import sys
import numpy as np
import imageio.v3 as iio
import glob
import argparse

# Add the project root to the path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def load_volumes(raw_base_dir, seg_base_dir, add_mask_base_dir, bbox_name):
    """
    Load volumes for debugging with detailed error reporting.
    """
    print(f"Loading volumes for {bbox_name}...")
    
    # Check if directories exist
    raw_dir = os.path.join(raw_base_dir, bbox_name)
    seg_dir = os.path.join(seg_base_dir, bbox_name)
    
    if bbox_name.startswith("bbox"):
        bbox_num = bbox_name.replace("bbox", "")
        add_mask_dir = os.path.join(add_mask_base_dir, f"bbox_{bbox_num}")
    else:
        add_mask_dir = os.path.join(add_mask_base_dir, bbox_name)
    
    print(f"Raw directory: {raw_dir}")
    print(f"  Exists: {os.path.exists(raw_dir)}")
    
    print(f"Segmentation directory: {seg_dir}")
    print(f"  Exists: {os.path.exists(seg_dir)}")
    
    print(f"Additional mask directory: {add_mask_dir}")
    print(f"  Exists: {os.path.exists(add_mask_dir)}")
    
    # Find TIF files
    try:
        raw_tif_files = sorted(glob.glob(os.path.join(raw_dir, 'slice_*.tif')))
        print(f"Found {len(raw_tif_files)} raw TIF files.")
        if len(raw_tif_files) > 0:
            print(f"  First file: {raw_tif_files[0]}")
            print(f"  Last file: {raw_tif_files[-1]}")
    except Exception as e:
        print(f"Error finding raw TIF files: {e}")
        raw_tif_files = []
    
    try:
        seg_tif_files = sorted(glob.glob(os.path.join(seg_dir, 'slice_*.tif')))
        print(f"Found {len(seg_tif_files)} segmentation TIF files.")
        if len(seg_tif_files) > 0:
            print(f"  First file: {seg_tif_files[0]}")
            print(f"  Last file: {seg_tif_files[-1]}")
    except Exception as e:
        print(f"Error finding segmentation TIF files: {e}")
        seg_tif_files = []
    
    try:
        add_mask_tif_files = sorted(glob.glob(os.path.join(add_mask_dir, 'slice_*.tif')))
        print(f"Found {len(add_mask_tif_files)} additional mask TIF files.")
        if len(add_mask_tif_files) > 0:
            print(f"  First file: {add_mask_tif_files[0]}")
            print(f"  Last file: {add_mask_tif_files[-1]}")
    except Exception as e:
        print(f"Error finding additional mask TIF files: {e}")
        add_mask_tif_files = []
    
    # Check if we have the same number of files
    if not (len(raw_tif_files) > 0 and len(seg_tif_files) > 0 and len(add_mask_tif_files) > 0):
        print("ERROR: Missing files in one or more directories.")
        return None, None, None
    
    if not (len(raw_tif_files) == len(seg_tif_files) == len(add_mask_tif_files)):
        print("WARNING: Different number of files in directories.")
        print(f"  Raw: {len(raw_tif_files)}")
        print(f"  Segmentation: {len(seg_tif_files)}")
        print(f"  Additional mask: {len(add_mask_tif_files)}")
    
    # Try to load each volume
    try:
        print("Loading raw volume...")
        raw_slices = []
        for f in raw_tif_files:
            try:
                img = iio.imread(f)
                print(f"  Loaded {f}, shape: {img.shape}, dtype: {img.dtype}")
                # Convert RGB to grayscale if needed
                if len(img.shape) > 2 and img.shape[2] > 1:
                    print(f"  Converting RGB to grayscale, original shape: {img.shape}")
                    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                    print(f"  Converted shape: {img.shape}")
                raw_slices.append(img)
            except Exception as e:
                print(f"  Error loading {f}: {e}")
                continue
        
        if len(raw_slices) > 0:
            print(f"Stacking {len(raw_slices)} raw slices...")
            print(f"  First slice shape: {raw_slices[0].shape}")
            raw_vol = np.stack(raw_slices, axis=0)
            print(f"  Stacked raw volume shape: {raw_vol.shape}")
        else:
            print("  No raw slices loaded!")
            raw_vol = None
    except Exception as e:
        print(f"Error loading raw volume: {e}")
        raw_vol = None
    
    try:
        print("Loading segmentation volume...")
        seg_slices = []
        for f in seg_tif_files:
            try:
                img = iio.imread(f)
                print(f"  Loaded {f}, shape: {img.shape}, dtype: {img.dtype}")
                # Handle multi-channel if needed
                if len(img.shape) > 2 and img.shape[2] > 1:
                    print(f"  Taking first channel from multi-channel image, original shape: {img.shape}")
                    img = img[..., 0]
                    print(f"  First channel shape: {img.shape}")
                seg_slices.append(img.astype(np.uint32))
            except Exception as e:
                print(f"  Error loading {f}: {e}")
                continue
        
        if len(seg_slices) > 0:
            print(f"Stacking {len(seg_slices)} segmentation slices...")
            print(f"  First slice shape: {seg_slices[0].shape}")
            seg_vol = np.stack(seg_slices, axis=0)
            print(f"  Stacked segmentation volume shape: {seg_vol.shape}")
        else:
            print("  No segmentation slices loaded!")
            seg_vol = None
    except Exception as e:
        print(f"Error loading segmentation volume: {e}")
        seg_vol = None
    
    try:
        print("Loading additional mask volume...")
        add_mask_slices = []
        for f in add_mask_tif_files:
            try:
                img = iio.imread(f)
                print(f"  Loaded {f}, shape: {img.shape}, dtype: {img.dtype}")
                # Handle multi-channel if needed
                if len(img.shape) > 2 and img.shape[2] > 1:
                    print(f"  Taking first channel from multi-channel image, original shape: {img.shape}")
                    img = img[..., 0]
                    print(f"  First channel shape: {img.shape}")
                add_mask_slices.append(img.astype(np.uint32))
            except Exception as e:
                print(f"  Error loading {f}: {e}")
                continue
        
        if len(add_mask_slices) > 0:
            print(f"Stacking {len(add_mask_slices)} additional mask slices...")
            print(f"  First slice shape: {add_mask_slices[0].shape}")
            add_mask_vol = np.stack(add_mask_slices, axis=0)
            print(f"  Stacked additional mask volume shape: {add_mask_vol.shape}")
        else:
            print("  No additional mask slices loaded!")
            add_mask_vol = None
    except Exception as e:
        print(f"Error loading additional mask volume: {e}")
        add_mask_vol = None
    
    # Final check
    if raw_vol is None or seg_vol is None or add_mask_vol is None:
        print("ERROR: Failed to load at least one volume.")
        return None, None, None
    
    return raw_vol, seg_vol, add_mask_vol

def main():
    parser = argparse.ArgumentParser(description="Debug volume loading for synapse dataset")
    parser.add_argument("--raw_base_dir", type=str, required=True, help="Base directory for raw data")
    parser.add_argument("--seg_base_dir", type=str, required=True, help="Base directory for segmentation data")
    parser.add_argument("--add_mask_base_dir", type=str, required=True, help="Base directory for additional mask data")
    parser.add_argument("--bbox_name", type=str, default="bbox1", help="Bounding box name to load")
    
    args = parser.parse_args()
    
    # Try to load volumes
    raw_vol, seg_vol, add_mask_vol = load_volumes(
        args.raw_base_dir,
        args.seg_base_dir,
        args.add_mask_base_dir,
        args.bbox_name
    )
    
    if raw_vol is not None and seg_vol is not None and add_mask_vol is not None:
        print("\nSUCCESS: All volumes loaded successfully.")
        print(f"Raw volume shape: {raw_vol.shape}")
        print(f"Segmentation volume shape: {seg_vol.shape}")
        print(f"Additional mask volume shape: {add_mask_vol.shape}")
    else:
        print("\nERROR: Failed to load volumes.")

if __name__ == "__main__":
    main() 