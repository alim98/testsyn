import os
import json
import re
import logging
import glob
from typing import Any, Dict, List, Union, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import label
from scipy import ndimage
from torchvision import transforms
import imageio.v3 as iio

# Replace external import with a simple config class
class Config:
    def __init__(self):
        self.gray_color = (0.5, 0.5, 0.5)  # Default gray color

config = Config()

class Synapse3DProcessor:
    def __init__(self, size=(80, 80), mean=(0.485,), std=(0.229,)):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            # Explicitly convert to grayscale with one output channel
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        self.mean = mean
        self.std = std
        self.normalize_volume = True  # New flag to control volume-wide normalization

    def __call__(self, frames, return_tensors=None):
        processed_frames = []
        for frame in frames:
            # Check if input is RGB (3 channels) or has unexpected shape
            if len(frame.shape) > 2 and frame.shape[2] > 1:
                if frame.shape[2] > 3:  # More than 3 channels
                    frame = frame[:, :, :3]  # Take first 3 channels
                # Will be converted to grayscale by the transform
            
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)
            
        pixel_values = torch.stack(processed_frames)
        
        # Ensure we have a single channel
        if pixel_values.shape[1] != 1:
            # This should not happen due to transforms.Grayscale, but just in case
            pixel_values = pixel_values.mean(dim=1, keepdim=True)
        
        # Apply volume-wide normalization to ensure consistent grayscale values across slices
        if self.normalize_volume:
            # Method 1: Min-max normalization across the entire volume
            min_val = pixel_values.min()
            max_val = pixel_values.max()
            if max_val > min_val:  # Avoid division by zero
                pixel_values = (pixel_values - min_val) / (max_val - min_val)
            
            # Method 2: Alternative - Z-score normalization using mean and std
            # pixel_values = (pixel_values - pixel_values.mean()) / (pixel_values.std() + 1e-6)
            # pixel_values = torch.clamp((pixel_values * 0.5) + 0.5, 0, 1)  # Rescale to [0,1]
            
            # Method 3: Histogram matching across slices (would require more complex implementation)
            # This would ensure all slices have similar intensity distributions
            
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values

    def save_segmented_slice(self, cube, output_path, slice_idx=None, consistent_gray=True):
        """
        Save a slice from a segmented cube with controlled normalization.
        
        Args:
            cube (numpy.ndarray): The cube with shape (y, x, c, z) from create_segmented_cube
            output_path (str): Path to save the image
            slice_idx (int, optional): Index of slice to save. If None, center slice is used.
            consistent_gray (bool): Whether to enforce consistent gray normalization
        """
        # Get the slice index (center if not specified)
        if slice_idx is None:
            slice_idx = cube.shape[3] // 2
        
        # Extract the slice - the cube is in (y, x, c, z) format
        slice_data = cube[:, :, :, slice_idx]
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create figure with controlled normalization
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use fixed vmin and vmax to prevent matplotlib's auto-scaling
        if consistent_gray:
            ax.imshow(slice_data, vmin=0, vmax=1)
        else:
            ax.imshow(slice_data)
        
        ax.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        return output_path

class SynapseDataLoader:
    def __init__(self, raw_base_dir, seg_base_dir, add_mask_base_dir, gray_color=None):
        self.raw_base_dir = raw_base_dir
        self.seg_base_dir = seg_base_dir
        self.add_mask_base_dir = add_mask_base_dir
        # Use provided gray_color or get from config
        self.gray_color = gray_color if gray_color is not None else config.gray_color

    def load_volumes(self, bbox_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raw_dir = os.path.join(self.raw_base_dir, bbox_name)
        seg_dir = os.path.join(self.seg_base_dir, bbox_name)
        
        if bbox_name.startswith("bbox"):
            bbox_num = bbox_name.replace("bbox", "")
            add_mask_dir = os.path.join(self.add_mask_base_dir, f"bbox_{bbox_num}")
        else:
            add_mask_dir = os.path.join(self.add_mask_base_dir, bbox_name)
        
        raw_tif_files = sorted(glob.glob(os.path.join(raw_dir, 'slice_*.tif')))
        seg_tif_files = sorted(glob.glob(os.path.join(seg_dir, 'slice_*.tif')))
        add_mask_tif_files = sorted(glob.glob(os.path.join(add_mask_dir, 'slice_*.tif')))
        
        if not (len(raw_tif_files) == len(seg_tif_files) == len(add_mask_tif_files)):
            return None, None, None
        
        try:
            # Load raw volume and convert to grayscale if needed
            raw_slices = []
            multi_channel_detected = False
            for f in raw_tif_files:
                img = iio.imread(f)
                # Check if the image has multiple channels (RGB)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # Convert RGB to grayscale using luminosity method
                    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                raw_slices.append(img)
            raw_vol = np.stack(raw_slices, axis=0)
            
            # Load segmentation volume and ensure it's single channel
            seg_slices = []
            for f in seg_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # For segmentation, take first channel (labels should be consistent)
                    img = img[..., 0]
                seg_slices.append(img.astype(np.uint32))
            seg_vol = np.stack(seg_slices, axis=0)
            
            # Load additional mask volume and ensure it's single channel
            add_mask_slices = []
            for f in add_mask_tif_files:
                img = iio.imread(f)
                if len(img.shape) > 2 and img.shape[2] > 1:
                    multi_channel_detected = True
                    # For masks, take first channel
                    img = img[..., 0]
                add_mask_slices.append(img.astype(np.uint32))
            add_mask_vol = np.stack(add_mask_slices, axis=0)
            
            if multi_channel_detected:
                print(f"WARNING: Multi-channel images detected in {bbox_name} and converted to single-channel")
            
            return raw_vol, seg_vol, add_mask_vol
        except Exception as e:
            print(f"Error loading volumes for {bbox_name}: {e}")
            return None, None, None

    @staticmethod
    def verify_single_channel(volume, name=""):
        """
        Verify if a volume is single-channel.
        
        Args:
            volume (numpy.ndarray): The volume to check
            name (str): Name for logging
            
        Returns:
            bool: True if single-channel, False otherwise
        """
        if len(volume.shape) == 3:  # Z, Y, X - single channel
            return True
            
        if len(volume.shape) == 4 and volume.shape[3] > 1:  # Z, Y, X, C with C > 1
            print(f"WARNING: Multi-channel volume detected in {name}: {volume.shape}")
            return False
            
        return True

    def get_closest_component_mask(
        self, 
        mask: np.ndarray, 
        z_start: int, 
        z_end: int, 
        y_start: int, 
        y_end: int, 
        x_start: int, 
        x_end: int, 
        central_coord: Tuple[int, int, int],
        distance_weight: float = 0.5
    ) -> np.ndarray:
        """
        Get a binary mask for the closest connected component to the central coordinate.
        
        Args:
            mask: Binary mask to find components in
            z_start: Start z-coordinate for the subvolume
            z_end: End z-coordinate for the subvolume
            y_start: Start y-coordinate for the subvolume
            y_end: End y-coordinate for the subvolume
            x_start: Start x-coordinate for the subvolume
            x_end: End x-coordinate for the subvolume
            central_coord: Central coordinate to find closest component to
            distance_weight: Weight to balance distance vs size in component selection 
                             (0.0 = only size matters, 1.0 = only distance matters)
            
        Returns:
            Binary mask with only the closest component
        """
        import time
        from collections import defaultdict
        
        # Dictionary to track execution times for different operations
        timers = defaultdict(float)
        
        from scipy import ndimage
        
        # Check if mask is non-empty
        t_start = time.time()
        if not np.any(mask):
            return np.zeros_like(mask, dtype=bool)
        timers['check_empty'] += time.time() - t_start
        
        # Label connected components
        t_start = time.time()
        labeled_mask, num_components = ndimage.label(mask)
        timers['label_components'] += time.time() - t_start
        
        # If there's only one component, return the original mask
        t_start = time.time()
        if num_components <= 1:
            return mask
        timers['check_single_component'] += time.time() - t_start
        
        # Extract properties for each component
        t_start = time.time()
        component_slices = ndimage.find_objects(labeled_mask)
        central_x, central_y, central_z = central_coord
        timers['find_objects'] += time.time() - t_start
        
        # Store component information for evaluation
        component_info = []
        
        # Calculate distance, size and bounding box for each component
        t_start_loop = time.time()
        for i, component_slice in enumerate(component_slices):
            t_start = time.time()
            # Skip empty slices
            if component_slice is None:
                continue
            timers['check_empty_slice'] += time.time() - t_start
                
            # Get the component mask
            t_start = time.time()
            component_mask = (labeled_mask == i + 1)
            timers['create_component_mask'] += time.time() - t_start
            
            # Calculate the size (number of voxels)
            t_start = time.time()
            size = np.sum(component_mask)
            timers['calculate_size'] += time.time() - t_start
            
            # Calculate the component centroid
            t_start = time.time()
            coords = np.where(component_mask)
            if len(coords[0]) == 0:
                continue
                
            center_z = np.mean(coords[0])
            center_y = np.mean(coords[1])
            center_x = np.mean(coords[2])
            timers['calculate_centroid'] += time.time() - t_start
            
            # Calculate Euclidean distance from component center to central coordinate
            t_start = time.time()
            distance = np.sqrt(
                (center_x - central_x)**2 + 
                (center_y - central_y)**2 + 
                (center_z - central_z)**2
            )
            timers['calculate_distance'] += time.time() - t_start
            
            # Check if component overlaps with target subvolume
            t_start = time.time()
            z_slice, y_slice, x_slice = component_slice
            z_min, z_max = z_slice.start, z_slice.stop
            y_min, y_max = y_slice.start, y_slice.stop
            x_min, x_max = x_slice.start, x_slice.stop
            
            overlaps_subvolume = (
                z_min < z_end and z_max > z_start and
                y_min < y_end and y_max > y_start and
                x_min < x_end and x_max > x_start
            )
            timers['check_overlap'] += time.time() - t_start
            
            # Calculate weighted score (lower is better)
            # Balance between distance and size - normalize both factors
            t_start = time.time()
            max_possible_distance = np.sqrt(
                (mask.shape[2])**2 + 
                (mask.shape[1])**2 + 
                (mask.shape[0])**2
            )
            
            # Normalize distance to [0, 1]
            normalized_distance = distance / max_possible_distance
            
            # Normalize size (larger size = lower score)
            # Use log scale to handle very different sizes
            normalized_size = 1.0 - np.log1p(size) / np.log1p(mask.size)
            
            # Weighted score (lower is better)
            score = (distance_weight * normalized_distance + 
                     (1 - distance_weight) * normalized_size)
            timers['calculate_score'] += time.time() - t_start
            
            # Add to component info
            t_start = time.time()
            component_info.append({
                'id': i + 1,
                'size': size,
                'distance': distance,
                'overlaps_subvolume': overlaps_subvolume,
                'score': score
            })
            timers['append_component_info'] += time.time() - t_start
        
        timers['component_loop'] += time.time() - t_start_loop
        
        t_start = time.time()
        if not component_info:
            return np.zeros_like(mask, dtype=bool)
        timers['check_empty_components'] += time.time() - t_start
        
        # First, try components that overlap with the subvolume
        t_start = time.time()
        overlapping_components = [c for c in component_info if c['overlaps_subvolume']]
        timers['filter_overlapping'] += time.time() - t_start
        
        t_start = time.time()
        if overlapping_components:
            # Find component with best score among overlapping components
            best_component = min(overlapping_components, key=lambda c: c['score'])
            selected_id = best_component['id']
        else:
            # If no overlap, use the component with best score
            best_component = min(component_info, key=lambda c: c['score'])
            selected_id = best_component['id']
        timers['find_best_component'] += time.time() - t_start
        
        # Create mask with only the selected component
        t_start = time.time()
        selected_mask = (labeled_mask == selected_id)
        timers['create_selected_mask'] += time.time() - t_start
        
        # Print timing statistics if this function takes too long (> 0.5 seconds)
        total_time = sum(timers.values())
        if total_time > 0.5:  # Only print if it's slow
            print(f"\nSlow component selection detected ({total_time:.2f}s) - num_components: {num_components}")
            for operation, duration in sorted(timers.items(), key=lambda x: x[1], reverse=True):
                percentage = (duration / total_time) * 100 if total_time > 0 else 0
                if percentage > 5:  # Only print significant contributors
                    print(f"  {operation}: {duration:.2f}s ({percentage:.1f}%)")
        
        return selected_mask

    def create_segmented_cube(
        self,
        raw_vol: np.ndarray,
        seg_vol: np.ndarray,
        add_mask_vol: np.ndarray,
        central_coord: Tuple[int, int, int],
        side1_coord: Tuple[int, int, int],
        side2_coord: Tuple[int, int, int],
        segmentation_type: int,
        subvolume_size: int = 80,
        alpha: float = 0.3,
        bbox_name: str = "",
        normalize_across_volume: bool = True,  # Add parameter to control normalization
    ) -> np.ndarray:
        bbox_num = bbox_name.replace("bbox", "").strip()
        
        if bbox_num in {'2', '5',}:
            mito_label = 1
            vesicle_label = 3
            cleft_label2 = 4
            cleft_label = 2
        elif bbox_num == '7':
            mito_label = 1
            vesicle_label = 2
            cleft_label2 = 3
            cleft_label = 4
        elif bbox_num == '4':
            mito_label = 3
            vesicle_label = 2
            cleft_label2 = 4
            cleft_label = 1
        elif bbox_num == '3':
            mito_label = 6
            vesicle_label = 7
            cleft_label2 = 8
            cleft_label = 9
        else:
            mito_label = 5
            vesicle_label = 6
            cleft_label = 7
            cleft_label2 = 7

        half_size = subvolume_size // 2
        cx, cy, cz = central_coord
        x_start = max(cx - half_size, 0)
        x_end = min(cx + half_size, raw_vol.shape[2])
        y_start = max(cy - half_size, 0)
        y_end = min(cy + half_size, raw_vol.shape[1])
        z_start = max(cz - half_size, 0)
        z_end = min(cz + half_size, raw_vol.shape[0])

        vesicle_full_mask = (add_mask_vol == vesicle_label)
        vesicle_mask = self.get_closest_component_mask(
            vesicle_full_mask,
            z_start, z_end,
            y_start, y_end,
            x_start, x_end,
            (cx, cy, cz)
        )

        def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
            x1, y1, z1 = s1_coord
            x2, y2, z2 = s2_coord
            seg_id_1 = segmentation_volume[z1, y1, x1]
            seg_id_2 = segmentation_volume[z2, y2, x2]
            mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            return mask_1, mask_2

        mask_1_full, mask_2_full = create_segment_masks(seg_vol, side1_coord, side2_coord)

        overlap_side1 = np.sum(np.logical_and(mask_1_full, vesicle_mask))
        overlap_side2 = np.sum(np.logical_and(mask_2_full, vesicle_mask))
        presynapse_side = 1 if overlap_side1 > overlap_side2 else 2

        if segmentation_type == 0:
            combined_mask_full = np.ones_like(add_mask_vol, dtype=bool)
        elif segmentation_type == 1:
            combined_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
        elif segmentation_type == 2:
            combined_mask_full = mask_2_full if presynapse_side == 1 else mask_1_full
        elif segmentation_type == 3:
            combined_mask_full = np.logical_or(mask_1_full, mask_2_full)
        elif segmentation_type == 4:
            vesicle_closest = self.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
            cleft_closest = self.get_closest_component_mask(
                ((add_mask_vol == cleft_label)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
            cleft_closest2 = self.get_closest_component_mask(
                ((add_mask_vol == cleft_label2)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
            combined_mask_full = np.logical_or(vesicle_closest, np.logical_or(cleft_closest,cleft_closest2))
        elif segmentation_type == 5:
            vesicle_closest = self.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
            cleft_closest = self.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
            combined_mask_extra = np.logical_or(vesicle_closest, cleft_closest)
            combined_mask_full = np.logical_or(mask_1_full, np.logical_or(mask_2_full, combined_mask_extra))
        elif segmentation_type == 6:
            combined_mask_full = self.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
        elif segmentation_type == 7:
            cleft_closest = self.get_closest_component_mask(
                ((add_mask_vol == cleft_label)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
            cleft_closest2 = self.get_closest_component_mask(
                ((add_mask_vol == cleft_label2)), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
            combined_mask_full =  np.logical_or(cleft_closest,cleft_closest2)
        elif segmentation_type == 8:
            combined_mask_full = self.get_closest_component_mask(
                (add_mask_vol == mito_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
        elif segmentation_type == 10:
            cleft_closest = self.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
            pre_mask_full = mask_1_full if presynapse_side == 1 else mask_2_full
            combined_mask_full = np.logical_or(cleft_closest,pre_mask_full)
        elif segmentation_type == 9:
            vesicle_closest = self.get_closest_component_mask(
                (add_mask_vol == vesicle_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
            cleft_closest = self.get_closest_component_mask(
                (add_mask_vol == cleft_label), z_start, z_end, y_start, y_end, x_start, x_end, (cx, cy, cz), 0.3
            )
            combined_mask_full = np.logical_or(cleft_closest,vesicle_closest)
        else:
            raise ValueError(f"Unsupported segmentation type: {segmentation_type}")

        sub_raw = raw_vol[z_start:z_end, y_start:y_end, x_start:x_end]
        sub_combined_mask = combined_mask_full[z_start:z_end, y_start:y_end, x_start:x_end]

        pad_z = subvolume_size - sub_raw.shape[0]
        pad_y = subvolume_size - sub_raw.shape[1]
        pad_x = subvolume_size - sub_raw.shape[2]
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            sub_raw = np.pad(sub_raw, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
            sub_combined_mask = np.pad(sub_combined_mask, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=False)

        sub_raw = sub_raw[:subvolume_size, :subvolume_size, :subvolume_size]
        sub_combined_mask = sub_combined_mask[:subvolume_size, :subvolume_size, :subvolume_size]

        sub_raw = sub_raw.astype(np.float32)
        
        # Apply normalization across the entire volume or per slice
        if normalize_across_volume:
            # Global normalization across the entire volume
            min_val = np.min(sub_raw)
            max_val = np.max(sub_raw)
            range_val = max_val - min_val if max_val > min_val else 1.0
            normalized = (sub_raw - min_val) / range_val
            
            # Print for debugging
            print(f"Global normalization: min={min_val:.4f}, max={max_val:.4f}, range={range_val:.4f}")
        else:
            # Original per-slice normalization
            mins = np.min(sub_raw, axis=(1, 2), keepdims=True)
            maxs = np.max(sub_raw, axis=(1, 2), keepdims=True)
            ranges = np.where(maxs > mins, maxs - mins, 1.0)
            normalized = (sub_raw - mins) / ranges
            
            # Print for debugging
            print(f"Per-slice normalization: shape of mins={mins.shape}, maxs={maxs.shape}")

        # Convert to RGB here ONLY for visualization purposes
        # The data processing pipeline uses grayscale (1-channel) format
        raw_rgb = np.repeat(normalized[..., np.newaxis], 3, axis=-1)
        mask_factor = sub_combined_mask[..., np.newaxis]

        if alpha < 1:
            blended_part = alpha * self.gray_color + (1 - alpha) * raw_rgb
        else:
            blended_part = self.gray_color * (1 - mask_factor) + raw_rgb * mask_factor

        overlaid_image = raw_rgb * mask_factor + (1 - mask_factor) * blended_part

        overlaid_cube = np.transpose(overlaid_image, (1, 2, 3, 0))

        return overlaid_cube

    def save_segmented_slice(self, cube, output_path, slice_idx=None, consistent_gray=True):
        """
        Save a slice from a segmented cube with controlled normalization.
        
        Args:
            cube (numpy.ndarray): The cube with shape (y, x, c, z) from create_segmented_cube
            output_path (str): Path to save the image
            slice_idx (int, optional): Index of slice to save. If None, center slice is used.
            consistent_gray (bool): Whether to enforce consistent gray normalization
        """
        # Get the slice index (center if not specified)
        if slice_idx is None:
            slice_idx = cube.shape[3] // 2
        
        # Extract the slice - the cube is in (y, x, c, z) format
        slice_data = cube[:, :, :, slice_idx]
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create figure with controlled normalization
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use fixed vmin and vmax to prevent matplotlib's auto-scaling
        if consistent_gray:
            ax.imshow(slice_data, vmin=0, vmax=1)
        else:
            ax.imshow(slice_data)
        
        ax.axis('off')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.close()
        
        return output_path

    def extract_raw_and_mask_cubes(
        self, 
        raw_vol, 
        seg_vol, 
        add_mask_vol, 
        central_coord, 
        side1_coord, 
        side2_coord, 
        subvolume_size=80, 
        bbox_name="",
        verbose=False
    ):
        """
        Extract raw and segmentation cubes around a synapse.
        IMPROVED APPROACH: First crop volumes to bounding box, then find components.
        
        Args:
            raw_vol: Raw image volume
            seg_vol: Segmentation volume
            add_mask_vol: Additional mask volume
            central_coord: Central coordinate (x, y, z)
            side1_coord: Side 1 coordinate (x, y, z)
            side2_coord: Side 2 coordinate (x, y, z)
            subvolume_size: Size of the subvolume cube
            bbox_name: Name of the bounding box
            verbose: Whether to print detailed debug information
            
        Returns:
            tuple: (raw_cube, mask_cube) - raw and segmentation cubes
        """
        import time
        
        start_time = time.time()
        
        if verbose:
            print(f"===== EXTRACTING CUBES FOR {bbox_name} =====")
        
        # Parse bbox number to get the right labels
        bbox_num = bbox_name.replace("bbox", "").strip()
        
        # Set labels based on the bounding box
        if bbox_num in {'2', '5',}:
            mito_label = 1
            vesicle_label = 3
            cleft_label2 = 4
            cleft_label = 2
        elif bbox_num == '7':
            mito_label = 1
            vesicle_label = 2
            cleft_label2 = 3
            cleft_label = 4
        elif bbox_num == '4':
            mito_label = 3
            vesicle_label = 2
            cleft_label2 = 4
            cleft_label = 1
        elif bbox_num == '3':
            mito_label = 6
            vesicle_label = 7
            cleft_label2 = 8
            cleft_label = 9
        else:
            # Default values for other bboxes
            mito_label = 5
            vesicle_label = 6
            cleft_label = 7
            cleft_label2 = 7

        # Define coordinate for center of cube
        cx, cy, cz = central_coord
        
        # Calculate bounds for subvolume with padding to ensure we include neighboring structures
        # Use 1.2x the half_size to ensure we capture all relevant structures
        padding_factor = 1.2
        half_size = subvolume_size // 2
        padded_half_size = int(half_size * padding_factor)
        
        # Calculate padded bounds
        z_start_padded = max(0, cz - padded_half_size)
        z_end_padded = min(raw_vol.shape[0], cz + padded_half_size)
        y_start_padded = max(0, cy - padded_half_size)
        y_end_padded = min(raw_vol.shape[1], cy + padded_half_size)
        x_start_padded = max(0, cx - padded_half_size)
        x_end_padded = min(raw_vol.shape[2], cx + padded_half_size)
        
        # Calculate bounds for final subvolume (without padding)
        z_start = max(0, cz - half_size)
        z_end = min(raw_vol.shape[0], cz + half_size)
        y_start = max(0, cy - half_size)
        y_end = min(raw_vol.shape[1], cy + half_size)
        x_start = max(0, cx - half_size)
        x_end = min(raw_vol.shape[2], cx + half_size)
        
        # Extract padded regions for processing 
        local_raw_vol = raw_vol[z_start_padded:z_end_padded, y_start_padded:y_end_padded, x_start_padded:x_end_padded].copy()
        local_seg_vol = seg_vol[z_start_padded:z_end_padded, y_start_padded:y_end_padded, x_start_padded:x_end_padded].copy()
        local_add_mask_vol = add_mask_vol[z_start_padded:z_end_padded, y_start_padded:y_end_padded, x_start_padded:x_end_padded].copy()
        
        # Adjust coordinates to local space
        local_cz, local_cy, local_cx = cz - z_start_padded, cy - y_start_padded, cx - x_start_padded
        local_s1z, local_s1y, local_s1x = side1_coord[2] - z_start_padded, side1_coord[1] - y_start_padded, side1_coord[0] - x_start_padded
        local_s2z, local_s2y, local_s2x = side2_coord[2] - z_start_padded, side2_coord[1] - y_start_padded, side2_coord[0] - x_start_padded
        
        # Ensure coordinates are within bounds
        local_s1z = max(0, min(local_s1z, local_raw_vol.shape[0]-1))
        local_s1y = max(0, min(local_s1y, local_raw_vol.shape[1]-1))
        local_s1x = max(0, min(local_s1x, local_raw_vol.shape[2]-1))
        local_s2z = max(0, min(local_s2z, local_raw_vol.shape[0]-1))
        local_s2y = max(0, min(local_s2y, local_raw_vol.shape[1]-1))
        local_s2x = max(0, min(local_s2x, local_raw_vol.shape[2]-1))
        
        # Create local coordinate tuples
        local_central_coord = (local_cx, local_cy, local_cz)
        local_side1_coord = (local_s1x, local_s1y, local_s1z)
        local_side2_coord = (local_s2x, local_s2y, local_s2z)
        
        # Define raw segment boundaries in local coordinates
        local_z_start = max(0, local_cz - half_size)
        local_z_end = min(local_raw_vol.shape[0], local_cz + half_size)
        local_y_start = max(0, local_cy - half_size)
        local_y_end = min(local_raw_vol.shape[1], local_cy + half_size)
        local_x_start = max(0, local_cx - half_size)
        local_x_end = min(local_raw_vol.shape[2], local_cx + half_size)
        
        # Create final output cubes
        raw_cube = np.zeros((subvolume_size, subvolume_size, subvolume_size), dtype=np.float32)
        mask_cube = np.zeros((subvolume_size, subvolume_size, subvolume_size), dtype=np.uint8)
        
        # Calculate offsets in the target cube
        z_offset = max(0, half_size - local_cz)
        y_offset = max(0, half_size - local_cy)
        x_offset = max(0, half_size - local_cx)
        
        # Extract sizes of extracted regions
        extracted_z = local_z_end - local_z_start
        extracted_y = local_y_end - local_y_start
        extracted_x = local_x_end - local_x_start
        
        # Insert raw data into output cube
        raw_cube[
            z_offset:z_offset+extracted_z,
            y_offset:y_offset+extracted_y,
            x_offset:x_offset+extracted_x
        ] = local_raw_vol[local_z_start:local_z_end, local_y_start:local_y_end, local_x_start:local_x_end]
        
        # Normalize raw cube to [0, 1] range
        raw_min = raw_cube.min()
        raw_max = raw_cube.max()
        if raw_max > raw_min:
            raw_cube = (raw_cube - raw_min) / (raw_max - raw_min)
           
        # Function to find the closest component in the CROPPED volume
        from scipy import ndimage
        
        def find_closest_component(mask_vol, label_value, local_central_coord):
            # Create binary mask for this label
            binary_mask = (mask_vol == label_value)
            
            # If no voxels with this label, return empty mask
            if not np.any(binary_mask):
                return np.zeros_like(binary_mask, dtype=bool)
            
            # Label connected components
            labeled_mask, num_components = ndimage.label(binary_mask)
            
            # If only one component, return it
            if num_components <= 1:
                return binary_mask
                
            # Find the closest component
            central_x, central_y, central_z = local_central_coord
            min_distance = float('inf')
            closest_component_id = None
            
            # Process each component to find the closest one
            for i in range(1, num_components + 1):
                component_mask = (labeled_mask == i)
                component_voxels = np.where(component_mask)
                
                if len(component_voxels[0]) == 0:
                    continue
                    
                # Calculate centroid
                center_z = np.mean(component_voxels[0])
                center_y = np.mean(component_voxels[1])
                center_x = np.mean(component_voxels[2])
                
                # Calculate distance to central coordinate
                distance = np.sqrt(
                    (center_x - central_x)**2 + 
                    (center_y - central_y)**2 + 
                    (center_z - central_z)**2
                )
                
                # Update closest component if this one is closer
                if distance < min_distance:
                    min_distance = distance
                    closest_component_id = i
            
            # Return mask for the closest component
            if closest_component_id is not None:
                return labeled_mask == closest_component_id
            else:
                return np.zeros_like(binary_mask, dtype=bool)
        
        # Function to create segment masks from side1 and side2 coordinates
        def create_segment_masks(segmentation_volume, s1_coord, s2_coord):
            x1, y1, z1 = s1_coord
            x2, y2, z2 = s2_coord
            
            # Check if coordinates are within bounds
            z1 = max(0, min(z1, segmentation_volume.shape[0]-1))
            y1 = max(0, min(y1, segmentation_volume.shape[1]-1))
            x1 = max(0, min(x1, segmentation_volume.shape[2]-1))
            z2 = max(0, min(z2, segmentation_volume.shape[0]-1))
            y2 = max(0, min(y2, segmentation_volume.shape[1]-1))
            x2 = max(0, min(x2, segmentation_volume.shape[2]-1))
            
            # Get segmentation IDs at the coordinates
            seg_id_1 = segmentation_volume[z1, y1, x1]
            seg_id_2 = segmentation_volume[z2, y2, x2]
            
            # Create binary masks for each segmentation ID
            mask_1 = (segmentation_volume == seg_id_1) if seg_id_1 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            mask_2 = (segmentation_volume == seg_id_2) if seg_id_2 != 0 else np.zeros_like(segmentation_volume, dtype=bool)
            
            return mask_1, mask_2
        
        # Create masks for side1 and side2 in local coordinates
        mask_1_full, mask_2_full = create_segment_masks(local_seg_vol, local_side1_coord, local_side2_coord)
        
        # Find closest vesicle component (using the cropped volume)
        vesicle_mask = find_closest_component(local_add_mask_vol, vesicle_label, local_central_coord)
        
        # Determine which side is presynapse (has more overlap with vesicle)
        if np.sum(vesicle_mask) > 0:
            # Calculate overlap between vesicle and each side
            overlap_side1 = np.sum(np.logical_and(mask_1_full, vesicle_mask))
            overlap_side2 = np.sum(np.logical_and(mask_2_full, vesicle_mask))
            presynapse_side = 1 if overlap_side1 > overlap_side2 else 2
        else:
            # If vesicle mask is empty, use a fallback approach
            # Choose the side with the smaller mask (typically presynapse is smaller than postsynapse)
            presynapse_side = 1 if np.sum(mask_1_full) < np.sum(mask_2_full) else 2
            
        # Find closest cleft components
        cleft_mask1 = find_closest_component(local_add_mask_vol, cleft_label, local_central_coord)
        cleft_mask2 = find_closest_component(local_add_mask_vol, cleft_label2, local_central_coord)
        combined_cleft = np.logical_or(cleft_mask1, cleft_mask2)
        
        # Extract subvolumes of each mask
        cleft_subvol = combined_cleft[local_z_start:local_z_end, local_y_start:local_y_end, local_x_start:local_x_end]
        vesicle_subvol = vesicle_mask[local_z_start:local_z_end, local_y_start:local_y_end, local_x_start:local_x_end]
        presynapse_mask = mask_1_full if presynapse_side == 1 else mask_2_full
        presynapse_subvol = presynapse_mask[local_z_start:local_z_end, local_y_start:local_y_end, local_x_start:local_x_end]
        
        # Insert masks into output cube
        # Cleft = 1
        mask_cube[
            z_offset:z_offset+extracted_z,
            y_offset:y_offset+extracted_y,
            x_offset:x_offset+extracted_x
        ][cleft_subvol] = 1
        
        # Vesicle = 2
        mask_cube[
            z_offset:z_offset+extracted_z,
            y_offset:y_offset+extracted_y,
            x_offset:x_offset+extracted_x
        ][vesicle_subvol] = 2
        
        # Presynapse = 3
        mask_cube[
            z_offset:z_offset+extracted_z,
            y_offset:y_offset+extracted_y,
            x_offset:x_offset+extracted_x
        ][presynapse_subvol] = 3
        
        # Reshape for consistency with model expectations (Z dimension last)
        raw_cube = np.transpose(raw_cube, (1, 2, 0))
        mask_cube = np.transpose(mask_cube, (1, 2, 0))
        
        # Print timing information if verbose
        if verbose:
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Extraction completed in {elapsed:.2f} seconds")
        
        return raw_cube, mask_cube

def normalize_cube_globally(cube):
    """
    Apply global normalization to a cube to ensure consistent grayscale values across slices.
    
    Args:
        cube (numpy.ndarray): The cube to normalize, expected to be in format (y, x, c, z)
        
    Returns:
        numpy.ndarray: Normalized cube with consistent grayscale values
    """
    # Make a copy to avoid modifying the original
    cube_copy = cube.copy()
    
    # Calculate global min and max across all dimensions
    min_val = np.min(cube_copy)
    max_val = np.max(cube_copy)
    
    # Avoid division by zero
    if max_val > min_val:
        # Apply global normalization
        cube_copy = (cube_copy - min_val) / (max_val - min_val)
        
    return cube_copy 

class ContrastiveProcessor:
    def __init__(self, size=(80, 80), mean=(0.485,), std=(0.229,)):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        self.mean = mean
        self.std = std
        self.normalize_volume = True
        
    def process_raw_volume(self, raw_cube):
        """
        Process a raw 3D volume for contrastive learning.
        
        Args:
            raw_cube: Raw image cube of shape (H, W, D)
            
        Returns:
            Processed tensor of shape (D, 1, H, W)
        """
        # Extract frames from the raw cube
        frames = [raw_cube[:, :, z] for z in range(raw_cube.shape[2])]
        
        # Process each frame
        processed_frames = []
        for frame in frames:
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)
            
        # Stack frames to create volume
        pixel_values = torch.stack(processed_frames)
        
        # Apply volume-wide normalization
        if self.normalize_volume:
            min_val = pixel_values.min()
            max_val = pixel_values.max()
            if max_val > min_val:
                pixel_values = (pixel_values - min_val) / (max_val - min_val)
        
        return pixel_values
        
    def process_mask_volume(self, mask_cube):
        """
        Process a mask 3D volume for contrastive learning.
        
        Args:
            mask_cube: Mask cube of shape (H, W, D) with integer labels
            
        Returns:
            One-hot encoded tensor of shape (D, C, H, W) where C is the number of classes
        """
        # Count the number of unique classes in the mask (including background 0)
        num_classes = len(np.unique(mask_cube))
        
        # Extract frames from the mask cube
        frames = [mask_cube[:, :, z] for z in range(mask_cube.shape[2])]
        
        # Process each frame
        processed_masks = []
        for frame in frames:
            # Convert to tensor
            mask_tensor = torch.from_numpy(frame).long()
            
            # One-hot encode the mask
            one_hot = torch.nn.functional.one_hot(mask_tensor, num_classes=num_classes)
            
            # Transpose to get channels first format (H, W, C) -> (C, H, W)
            one_hot = one_hot.permute(2, 0, 1).float()
            
            processed_masks.append(one_hot)
            
        # Stack frames to create volume (D, C, H, W)
        mask_values = torch.stack(processed_masks)
        
        return mask_values
        
    def __call__(self, raw_cube, mask_cube=None, return_tensors=None):
        """
        Process both raw and mask data for contrastive learning.
        
        Args:
            raw_cube: Raw image cube
            mask_cube: Optional mask cube
            return_tensors: Format to return tensors in
            
        Returns:
            Dict with processed tensors
        """
        pixel_values = self.process_raw_volume(raw_cube)
        
        if mask_cube is not None:
            mask_values = self.process_mask_volume(mask_cube)
            
            if return_tensors == "pt":
                return {"pixel_values": pixel_values, "mask_values": mask_values}
            else:
                return pixel_values, mask_values
        
        if return_tensors == "pt":
            return {"pixel_values": pixel_values}
        else:
            return pixel_values 

class ContrastiveAugmentationProcessor(ContrastiveProcessor):
    def __init__(self, size=(80, 80), mean=(0.485,), std=(0.229,), augmentation_strength='medium', 
                 use_point_cloud=False, max_points=512):
        """
        Extended processor with augmentations for contrastive learning.
        
        Args:
            size: Size to resize images to
            mean: Mean for normalization
            std: Standard deviation for normalization
            augmentation_strength: Strength of augmentation ('light', 'medium', 'strong')
            use_point_cloud: Whether to extract point clouds from masks
            max_points: Maximum number of points to sample from masks
        """
        super().__init__(size, mean, std)
        self.augmentation_strength = augmentation_strength
        self.use_point_cloud = use_point_cloud
        self.max_points = max_points
        
        # Define augmentation parameters based on strength
        if augmentation_strength == 'light':
            self.rotation_range = 10
            self.shift_range = 0.05
            self.zoom_range = (0.95, 1.05)
            self.intensity_range = (0.9, 1.1)
            self.prob_flip = 0.2
            self.prob_noise = 0.2
            self.noise_scale = 0.03
        elif augmentation_strength == 'medium':
            self.rotation_range = 20
            self.shift_range = 0.1
            self.zoom_range = (0.9, 1.1)
            self.intensity_range = (0.8, 1.2)
            self.prob_flip = 0.3
            self.prob_noise = 0.3
            self.noise_scale = 0.05
        elif augmentation_strength == 'strong':
            self.rotation_range = 30
            self.shift_range = 0.15
            self.zoom_range = (0.85, 1.15)
            self.intensity_range = (0.7, 1.3)
            self.prob_flip = 0.5
            self.prob_noise = 0.5
            self.noise_scale = 0.08
        else:
            raise ValueError(f"Unknown augmentation strength: {augmentation_strength}")
            
    def _augment_rotation(self, image):
        """Apply random rotation to the image"""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        return transforms.functional.rotate(image, angle)
    
    def _augment_shift(self, image):
        """Apply random translation to the image"""
        height, width = image.shape[-2:]
        tx = int(width * np.random.uniform(-self.shift_range, self.shift_range))
        ty = int(height * np.random.uniform(-self.shift_range, self.shift_range))
        return transforms.functional.affine(image, angle=0, translate=(tx, ty), scale=1.0, shear=0)
    
    def _augment_zoom(self, image):
        """Apply random zoom to the image"""
        scale = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        return transforms.functional.affine(image, angle=0, translate=(0, 0), scale=scale, shear=0)
    
    def _augment_intensity(self, image):
        """Apply random intensity scaling to the image"""
        scale = np.random.uniform(self.intensity_range[0], self.intensity_range[1])
        return image * scale
    
    def _augment_flip(self, image):
        """Apply random horizontal or vertical flip to the image"""
        if np.random.random() < 0.5:
            return transforms.functional.hflip(image)
        else:
            return transforms.functional.vflip(image)
    
    def _augment_noise(self, image):
        """Apply random noise to the image"""
        noise = torch.randn_like(image) * self.noise_scale
        return torch.clamp(image + noise, 0, 1)
    
    def _apply_augmentations(self, image):
        """Apply a series of augmentations to the image"""
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).float()
            else:
                # Assume it's a PIL image
                image = transforms.functional.to_tensor(image)
        
        # Apply augmentations with probability
        if np.random.random() < self.prob_flip:
            image = self._augment_flip(image)
            
        # Always apply these augmentations
        image = self._augment_rotation(image)
        image = self._augment_shift(image)
        image = self._augment_zoom(image)
        image = self._augment_intensity(image)
        
        if np.random.random() < self.prob_noise:
            image = self._augment_noise(image)
            
        return image
    
    def process_raw_volume(self, raw_cube, apply_augmentation=True):
        """
        Process and optionally augment a raw 3D volume for contrastive learning.
        
        Args:
            raw_cube: Raw image cube of shape (H, W, D)
            apply_augmentation: Whether to apply augmentations
            
        Returns:
            Processed tensor of shape (D, 1, H, W)
        """
        # Extract frames from the raw cube
        frames = [raw_cube[:, :, z] for z in range(raw_cube.shape[2])]
        
        # Process each frame
        processed_frames = []
        for frame in frames:
            processed_frame = self.transform(frame)
            processed_frames.append(processed_frame)
            
        # Stack frames to create volume
        pixel_values = torch.stack(processed_frames)
        
        # Apply volume-wide normalization
        if self.normalize_volume:
            min_val = pixel_values.min()
            max_val = pixel_values.max()
            if max_val > min_val:
                pixel_values = (pixel_values - min_val) / (max_val - min_val)
        
        # Apply augmentations if requested
        if apply_augmentation:
            augmented_frames = []
            for frame in pixel_values:
                augmented_frame = self._apply_augmentations(frame)
                augmented_frames.append(augmented_frame)
            pixel_values = torch.stack(augmented_frames)
        
        return pixel_values
    
    def extract_point_cloud(self, mask_cube):
        """
        Extract a combined point cloud from a 3D mask (excluding vesicles).
        
        NOTE: This extracts a combined point cloud from all mask values > 0,
        EXCEPT vesicles (label 2) which are intentionally excluded.
        For structure-specific point clouds, use extract_separate_point_clouds() instead.
        
        Args:
            mask_cube: Mask cube of shape (H, W, D) with integer labels
            
        Returns:
            Point cloud tensor of shape (N, 3) or None if extraction fails
        """
        if mask_cube is None:
            return None
        
        try:
            from models.contrastive_model import PointCloudEncoder
            
            # Make a copy to avoid modifying the original
            mask_cube_copy = mask_cube.copy() if isinstance(mask_cube, np.ndarray) else mask_cube.clone()
            
            # Convert to torch tensor if needed
            if isinstance(mask_cube_copy, np.ndarray):
                mask_cube_copy = torch.from_numpy(mask_cube_copy).float()
            
            # Create a binary mask excluding vesicles (label 2)
            # Only include cleft (1) and presynapse (3)
            binary_mask = ((mask_cube_copy == 1) | (mask_cube_copy == 3)).float()
            
            # Use the PointCloudEncoder's static method to convert mask to point cloud
            point_cloud = PointCloudEncoder.mask_to_point_cloud(
                binary_mask, num_samples=self.max_points
            )
            
            return point_cloud
        except Exception as e:
            print(f"Error extracting point cloud: {e}")
            return None
    
    def extract_separate_point_clouds(self, mask_cube):
        """
        Extract separate point clouds for each structure type in the mask.
        EXCEPT vesicles - we skip vesicle point clouds entirely.
        
        IMPORTANT: This function extracts point clouds DIRECTLY from the mask data (not from overlaid or raw data).
        It uses the mask values directly, where:
        - 1 = cleft
        - 3 = presynapse/mitochondria
        
        Note: Vesicle point clouds (mask value 2) are intentionally excluded.
        
        Args:
            mask_cube: Mask cube with values 1=cleft, 2=vesicle, 3=presynapse/mitochondria
            
        Returns:
            Dictionary of point clouds for each structure type (EXCEPT vesicles)
        """
        if mask_cube is None:
            return {}
        
        try:
            from models.contrastive_model import PointCloudEncoder
            
            # Make a copy to avoid modifying the original
            mask_cube_copy = mask_cube.copy() if isinstance(mask_cube, np.ndarray) else mask_cube.clone()
            
            # Convert to torch tensor if needed
            if isinstance(mask_cube_copy, np.ndarray):
                mask_cube_copy = torch.from_numpy(mask_cube_copy).float()
            
            # Create completely separate masks for each structure type (no overlaps possible)
            # Exclude vesicles (label 2) entirely
            cleft_mask_cube = torch.zeros_like(mask_cube_copy)
            presynapse_mask_cube = torch.zeros_like(mask_cube_copy)
            
            # Only set the voxels that belong to each specific structure
            cleft_mask_cube[mask_cube_copy == 1] = 1.0
            presynapse_mask_cube[mask_cube_copy == 3] = 1.0
            
            # Extract point clouds from each separate mask cube
            cleft_points = PointCloudEncoder.mask_to_point_cloud(cleft_mask_cube, num_samples=self.max_points)
            presynapse_points = PointCloudEncoder.mask_to_point_cloud(presynapse_mask_cube, num_samples=self.max_points)
            
            # Remove empty point clouds (all zeros)
            def clean_points(points):
                if points is not None and isinstance(points, torch.Tensor):
                    non_zero_mask = torch.sum(torch.abs(points), dim=1) > 0
                    if non_zero_mask.any():
                        return points[non_zero_mask]
                return None
            
            # Clean point clouds
            cleft_points = clean_points(cleft_points)
            presynapse_points = clean_points(presynapse_points)
            
            # Create dictionary of point clouds - EXCLUDING vesicles
            point_clouds = {}
            if cleft_points is not None and len(cleft_points) > 0:
                point_clouds['cleft'] = cleft_points
            if presynapse_points is not None and len(presynapse_points) > 0:
                point_clouds['presynapse'] = presynapse_points
                
            return point_clouds
            
        except Exception as e:
            print(f"Error extracting separate point clouds: {e}")
            return {}
    
    def generate_augmented_pair(self, raw_cube, mask_cube=None):
        """
        Generate two differently augmented versions of the same data.
        
        Args:
            raw_cube: Raw image cube
            mask_cube: Optional mask cube
            
        Returns:
            Dictionary with two augmented versions
        """
        # First augmented version
        aug1_raw = self.process_raw_volume(raw_cube, apply_augmentation=True)
        
        # Second augmented version (different random augmentations)
        aug2_raw = self.process_raw_volume(raw_cube, apply_augmentation=True)
        
        result = {
            "pixel_values_aug1": aug1_raw,
            "pixel_values_aug2": aug2_raw
        }
        
        # Process mask if provided
        if mask_cube is not None:
            # Create basic mask values first
            mask_values = self.process_mask_volume(mask_cube)
            result["mask_values"] = mask_values
            
            # Extract point cloud if requested
            if self.use_point_cloud:
                # Extract combined point cloud (for backward compatibility)
                point_cloud = self.extract_point_cloud(mask_cube)
                if point_cloud is not None:
                    result["point_cloud"] = point_cloud
                
                # Extract separate point clouds for each structure type
                separate_point_clouds = self.extract_separate_point_clouds(mask_cube)
                if separate_point_clouds:
                    result["separate_point_clouds"] = separate_point_clouds
            
        return result
    
    def __call__(self, raw_cube, mask_cube=None, return_tensors=None, generate_pair=False):
        """
        Process both raw and mask data for contrastive learning.
        
        Args:
            raw_cube: Raw image cube
            mask_cube: Optional mask cube
            return_tensors: Format to return tensors in
            generate_pair: Whether to generate a contrastive pair
            
        Returns:
            Dict with processed tensors
        """
        if generate_pair:
            return self.generate_augmented_pair(raw_cube, mask_cube)
        
        pixel_values = self.process_raw_volume(raw_cube)
        result = {"pixel_values": pixel_values}
        
        if mask_cube is not None:
            mask_values = self.process_mask_volume(mask_cube)
            result["mask_values"] = mask_values
            
            # Extract point cloud if requested
            if self.use_point_cloud:
                # Extract combined point cloud (for backward compatibility)
                point_cloud = self.extract_point_cloud(mask_cube)
                if point_cloud is not None:
                    result["point_cloud"] = point_cloud
                    
                # Extract separate point clouds for each structure type
                separate_point_clouds = self.extract_separate_point_clouds(mask_cube)
                if separate_point_clouds:
                    result["separate_point_clouds"] = separate_point_clouds
        
        if return_tensors == "pt":
            return result
        else:
            return pixel_values 