import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
import os

from .dataloader import SynapseDataLoader, ContrastiveAugmentationProcessor

class SynapseDataset(Dataset):
    def __init__(self, vol_data_dict: dict, synapse_df: pd.DataFrame, processor,
                 segmentation_type: int, subvol_size: int = 80, num_frames: int = 80,
                 alpha: float = 0.3, normalize_across_volume: bool = True):
        self.vol_data_dict = vol_data_dict
        self.synapse_df = synapse_df.reset_index(drop=True)
        self.processor = processor
        self.segmentation_type = segmentation_type
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.alpha = alpha
        self.data_loader = None
        self.normalize_across_volume = normalize_across_volume
        # Ensure the processor's normalization setting matches
        if hasattr(self.processor, 'normalize_volume'):
            self.processor.normalize_volume = normalize_across_volume

    def __len__(self):
        return len(self.synapse_df)

    def __getitem__(self, idx):
        syn_info = self.synapse_df.iloc[idx]
        bbox_name = syn_info['bbox_name']
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))
        if raw_vol is None:
            return torch.zeros((self.num_frames, 1, self.subvol_size, self.subvol_size), dtype=torch.float32), syn_info, bbox_name

        central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
        side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
        side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))

        if self.data_loader is None:
            self.data_loader = SynapseDataLoader("", "", "")
            
        overlaid_cube = self.data_loader.create_segmented_cube(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            segmentation_type=self.segmentation_type,
            subvolume_size=self.subvol_size,
            alpha=self.alpha,
            bbox_name=bbox_name,
            normalize_across_volume=self.normalize_across_volume,
        )
        
        # Extract frames from the overlaid cube
        frames = [overlaid_cube[..., z] for z in range(overlaid_cube.shape[3])]
        
        # Ensure we have the correct number of frames
        if len(frames) < self.num_frames:
            # Duplicate the last frame to reach the desired number
            frames += [frames[-1]] * (self.num_frames - len(frames))
        elif len(frames) > self.num_frames:
            # Sample frames evenly across the volume
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Process frames and get pixel values
        inputs = self.processor(frames, return_tensors="pt")
        
        return inputs["pixel_values"].squeeze(0).float(), syn_info, bbox_name

class SynapseDataset2(Dataset):
    def __init__(self, vol_data_dict: dict, synapse_df: pd.DataFrame, processor,
                 segmentation_type: int, subvol_size: int = 80, num_frames: int = 16,
                 alpha: float = 0.3, fixed_samples=None, normalize_across_volume: bool = True):
        self.vol_data_dict = vol_data_dict

        if fixed_samples:
            fixed_samples_df = pd.DataFrame(fixed_samples)
            self.synapse_df = synapse_df.merge(fixed_samples_df, on=['Var1', 'bbox_name'], how='inner')
        else:
            self.synapse_df = synapse_df.reset_index(drop=True)

        self.processor = processor
        self.segmentation_type = segmentation_type
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.alpha = alpha
        self.data_loader = None
        self.normalize_across_volume = normalize_across_volume
        # Ensure the processor's normalization setting matches
        if hasattr(self.processor, 'normalize_volume'):
            self.processor.normalize_volume = normalize_across_volume

    def __len__(self):
        return len(self.synapse_df)

    def __getitem__(self, idx):
        syn_info = self.synapse_df.iloc[idx]
        bbox_name = syn_info['bbox_name']
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))
        if raw_vol is None:
            return torch.zeros((self.num_frames, 1, self.subvol_size, self.subvol_size), dtype=torch.float32), syn_info, bbox_name

        central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
        side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
        side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))

        if self.data_loader is None:
            self.data_loader = SynapseDataLoader("", "", "")
            
        overlaid_cube = self.data_loader.create_segmented_cube(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            segmentation_type=self.segmentation_type,
            subvolume_size=self.subvol_size,
            alpha=self.alpha,
            bbox_name=bbox_name,
            normalize_across_volume=self.normalize_across_volume,
        )
        
        # Extract frames from the overlaid cube
        frames = [overlaid_cube[..., z] for z in range(overlaid_cube.shape[3])]
        
        # Ensure we have the correct number of frames
        if len(frames) < self.num_frames:
            # Duplicate the last frame to reach the desired number
            frames += [frames[-1]] * (self.num_frames - len(frames))
        elif len(frames) > self.num_frames:
            # Sample frames evenly across the volume
            indices = np.linspace(0, len(frames)-1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]

        # Process frames and get pixel values
        inputs = self.processor(frames, return_tensors="pt")
        
        return inputs["pixel_values"].squeeze(0).float(), syn_info, bbox_name 

class ContrastiveSynapseDataset(Dataset):
    def __init__(self, vol_data_dict: dict, synapse_df: pd.DataFrame, processor,
                 subvol_size: int = 80, num_frames: int = 80):
        """
        Dataset for contrastive learning with separate raw and mask data.
        
        Args:
            vol_data_dict: Dictionary of loaded volumes with structure {bbox_name: (raw_vol, seg_vol, add_mask_vol)}
            synapse_df: DataFrame with synapse information
            processor: Processor to convert raw and mask volumes to tensors
            subvol_size: Size of the subvolume cube
            num_frames: Number of frames to return
        """
        self.vol_data_dict = vol_data_dict
        self.synapse_df = synapse_df.reset_index(drop=True)
        self.processor = processor
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.data_loader = None
        
        # Ensure we have the correct processor type
        if not hasattr(self.processor, 'process_raw_volume') or not hasattr(self.processor, 'process_mask_volume'):
            print("Warning: The processor does not have the required methods for contrastive learning. "
                  "Please use ContrastiveProcessor.")

    def __len__(self):
        return len(self.synapse_df)

    def __getitem__(self, idx):
        syn_info = self.synapse_df.iloc[idx]
        bbox_name = syn_info['bbox_name']
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))
        
        if raw_vol is None:
            # Return empty tensors if data not found
            empty_pixel_values = torch.zeros((self.num_frames, 1, self.subvol_size, self.subvol_size), dtype=torch.float32)
            empty_mask_values = torch.zeros((self.num_frames, 4, self.subvol_size, self.subvol_size), dtype=torch.float32)
            return empty_pixel_values, empty_mask_values, syn_info, bbox_name

        central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
        side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
        side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))

        if self.data_loader is None:
            # Paths will be set in config at runtime
            self.data_loader = SynapseDataLoader("", "", "")
            
        # Extract raw and mask cubes
        raw_cube, mask_cube = self.data_loader.extract_raw_and_mask_cubes(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            subvolume_size=self.subvol_size,
            bbox_name=bbox_name,
        )
        
        # Ensure we have the required number of frames by adjusting if needed
        if raw_cube.shape[2] < self.num_frames:
            # Duplicate the last frames to reach the desired number
            raw_cube_padded = np.zeros((raw_cube.shape[0], raw_cube.shape[1], self.num_frames), dtype=np.float32)
            mask_cube_padded = np.zeros((mask_cube.shape[0], mask_cube.shape[1], self.num_frames), dtype=np.uint8)
            
            # Copy the available frames
            raw_cube_padded[:, :, :raw_cube.shape[2]] = raw_cube
            mask_cube_padded[:, :, :mask_cube.shape[2]] = mask_cube
            
            # Duplicate the last frame
            for z in range(raw_cube.shape[2], self.num_frames):
                raw_cube_padded[:, :, z] = raw_cube[:, :, -1]
                mask_cube_padded[:, :, z] = mask_cube[:, :, -1]
                
            raw_cube = raw_cube_padded
            mask_cube = mask_cube_padded
            
        elif raw_cube.shape[2] > self.num_frames:
            # Sample frames evenly
            indices = np.linspace(0, raw_cube.shape[2]-1, self.num_frames, dtype=int)
            raw_cube_sampled = np.zeros((raw_cube.shape[0], raw_cube.shape[1], self.num_frames), dtype=np.float32)
            mask_cube_sampled = np.zeros((mask_cube.shape[0], mask_cube.shape[1], self.num_frames), dtype=np.uint8)
            
            for i, idx in enumerate(indices):
                raw_cube_sampled[:, :, i] = raw_cube[:, :, idx]
                mask_cube_sampled[:, :, i] = mask_cube[:, :, idx]
                
            raw_cube = raw_cube_sampled
            mask_cube = mask_cube_sampled
        
        # Process raw and mask data
        processed_data = self.processor(raw_cube, mask_cube, return_tensors="pt")
        
        return processed_data["pixel_values"], processed_data["mask_values"], syn_info, bbox_name

class ContrastiveSynapseLoader:
    """Helper class to load and prepare data for contrastive learning"""
    
    def __init__(self, config):
        """
        Initialize the ContrastiveSynapseLoader.
        
        Args:
            config: Configuration object containing paths and parameters
        """
        self.config = config
        self.data_loader = SynapseDataLoader(
            raw_base_dir=config.raw_base_dir,
            seg_base_dir=config.seg_base_dir,
            add_mask_base_dir=config.add_mask_base_dir,
            gray_color=config.gray_color
        )
        self.processor = ContrastiveProcessor(size=config.size)
        
    def load_data(self, bbox_names=None):
        """
        Load all volume data.
        
        Args:
            bbox_names: List of bbox names to load (if None, uses config.bbox_name)
            
        Returns:
            Dictionary of loaded volumes {bbox_name: (raw_vol, seg_vol, add_mask_vol)}
        """
        if bbox_names is None:
            bbox_names = self.config.bbox_name
            
        vol_data_dict = {}
        for bbox_name in bbox_names:
            print(f"Loading volumes for {bbox_name}...")
            raw_vol, seg_vol, add_mask_vol = self.data_loader.load_volumes(bbox_name)
            if raw_vol is not None:
                vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
                
        return vol_data_dict
    
    def create_dataset(self, vol_data_dict, synapse_df, batch_size=8, shuffle=True, num_workers=4):
        """
        Create dataset and dataloader for contrastive learning.
        
        Args:
            vol_data_dict: Dictionary of loaded volumes
            synapse_df: DataFrame with synapse information
            batch_size: Batch size for dataloader
            shuffle: Whether to shuffle the dataset
            num_workers: Number of workers for dataloader
            
        Returns:
            Dataset and DataLoader objects
        """
        dataset = ContrastiveSynapseDataset(
            vol_data_dict=vol_data_dict,
            synapse_df=synapse_df,
            processor=self.processor,
            subvol_size=self.config.subvol_size,
            num_frames=self.config.num_frames
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        return dataset, dataloader 

class ContrastiveAugmentedDataset(Dataset):
    def __init__(self, vol_data_dict: dict, synapse_df: pd.DataFrame, processor,
                 subvol_size: int = 80, num_frames: int = 80, use_point_cloud=False):
        """
        Dataset for contrastive learning with augmentation pairs.
        
        Args:
            vol_data_dict: Dictionary of loaded volumes with structure {bbox_name: (raw_vol, seg_vol, add_mask_vol)}
            synapse_df: DataFrame with synapse information
            processor: ContrastiveAugmentationProcessor for data processing and augmentation
            subvol_size: Size of the subvolume cube
            num_frames: Number of frames to return
            use_point_cloud: Whether to use point cloud extraction
        """
        self.vol_data_dict = vol_data_dict
        self.synapse_df = synapse_df.reset_index(drop=True)
        self.processor = processor
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.data_loader = None
        self.use_point_cloud = use_point_cloud
        
        # Ensure we have the correct processor type
        if not hasattr(self.processor, 'generate_augmented_pair'):
            print("Warning: The processor does not have the required methods for contrastive learning. "
                  "Please use ContrastiveAugmentationProcessor.")

    def __len__(self):
        return len(self.synapse_df)

    def __getitem__(self, idx):
        syn_info = self.synapse_df.iloc[idx]
        bbox_name = syn_info['bbox_name']
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))
        
        if raw_vol is None:
            # Return empty tensors if data not found
            empty_pixel_values = torch.zeros((self.num_frames, 1, self.subvol_size, self.subvol_size), dtype=torch.float32)
            return {
                "pixel_values_aug1": empty_pixel_values,
                "pixel_values_aug2": empty_pixel_values.clone(),
                "syn_info": syn_info,
                "bbox_name": bbox_name
            }

        central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
        side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
        side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))

        if self.data_loader is None:
            # Paths will be set in config at runtime
            self.data_loader = SynapseDataLoader("", "", "")
            
        # Extract raw and mask cubes
        raw_cube, mask_cube = self.data_loader.extract_raw_and_mask_cubes(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            subvolume_size=self.subvol_size,
            bbox_name=bbox_name,
        )
        
        # Ensure we have the required number of frames by adjusting if needed
        if raw_cube.shape[2] < self.num_frames:
            # Duplicate the last frames to reach the desired number
            raw_cube_padded = np.zeros((raw_cube.shape[0], raw_cube.shape[1], self.num_frames), dtype=np.float32)
            mask_cube_padded = np.zeros((mask_cube.shape[0], mask_cube.shape[1], self.num_frames), dtype=np.uint8)
            
            # Copy the available frames
            raw_cube_padded[:, :, :raw_cube.shape[2]] = raw_cube
            mask_cube_padded[:, :, :mask_cube.shape[2]] = mask_cube
            
            # Duplicate the last frame
            for z in range(raw_cube.shape[2], self.num_frames):
                raw_cube_padded[:, :, z] = raw_cube[:, :, -1]
                mask_cube_padded[:, :, z] = mask_cube[:, :, -1]
                
            raw_cube = raw_cube_padded
            mask_cube = mask_cube_padded
            
        elif raw_cube.shape[2] > self.num_frames:
            # Sample frames evenly
            indices = np.linspace(0, raw_cube.shape[2]-1, self.num_frames, dtype=int)
            raw_cube_sampled = np.zeros((raw_cube.shape[0], raw_cube.shape[1], self.num_frames), dtype=np.float32)
            mask_cube_sampled = np.zeros((mask_cube.shape[0], mask_cube.shape[1], self.num_frames), dtype=np.uint8)
            
            for i, idx in enumerate(indices):
                raw_cube_sampled[:, :, i] = raw_cube[:, :, idx]
                mask_cube_sampled[:, :, i] = mask_cube[:, :, idx]
                
            raw_cube = raw_cube_sampled
            mask_cube = mask_cube_sampled
        
        # Process raw and mask data with augmentation
        processed_data = self.processor(raw_cube, mask_cube, return_tensors="pt", generate_pair=True)
        
        result = {
            "pixel_values_aug1": processed_data["pixel_values_aug1"],
            "pixel_values_aug2": processed_data["pixel_values_aug2"],
            "mask_values": processed_data.get("mask_values", None),
            "syn_info": syn_info,
            "bbox_name": bbox_name
        }
        
        # Include point cloud data if available
        if self.use_point_cloud and "point_cloud" in processed_data:
            result["point_cloud"] = processed_data["point_cloud"]
            
        return result

class ContrastiveAugmentedLoader:
    """Helper class to load and prepare data for contrastive learning"""
    
    def __init__(self, config, augmentation_strength='medium', use_point_cloud=False, max_points=512):
        self.config = config
        self.augmentation_strength = augmentation_strength
        self.use_point_cloud = use_point_cloud
        self.max_points = max_points
        
        # If use_point_cloud is specified in config, use that value
        if hasattr(config, 'use_point_cloud'):
            self.use_point_cloud = config.use_point_cloud
        # If max_points is specified in config, use that value
        if hasattr(config, 'max_points'):
            self.max_points = config.max_points
            
        self.dataloader = SynapseDataLoader(
            config.raw_base_dir,
            config.seg_base_dir,
            config.add_mask_base_dir
        )
        
    def load_data(self, bbox_names=None):
        """
        Load volumetric data for all bounding boxes.
        
        Args:
            bbox_names: List of bounding box names to load (default: from config)
            
        Returns:
            Dictionary mapping bbox_name -> (raw_vol, seg_vol, add_mask_vol)
        """
        if bbox_names is None:
            bbox_names = self.config.bbox_name
            
        print(f"Loading data for {len(bbox_names)} bounding boxes...")
        vol_data_dict = {}
        for bbox_name in bbox_names:
            print(f"Loading {bbox_name}...")
            # Load the volumes using SynapseDataLoader
            raw_vol, seg_vol, add_mask_vol = self.dataloader.load_volumes(bbox_name)
            
            if raw_vol is not None:
                vol_data_dict[bbox_name] = (raw_vol, seg_vol, add_mask_vol)
                print(f"  Loaded volumes with shapes: raw={raw_vol.shape}, seg={seg_vol.shape}, add_mask={add_mask_vol.shape}")
            else:
                print(f"  Failed to load volumes for {bbox_name}")
                
        return vol_data_dict
    
    def load_synapse_metadata(self):
        """
        Load synapse metadata from Excel file.
        
        Returns:
            DataFrame with synapse information
        """
        # Define required columns upfront so they're available in the error handler
        required_columns = ['central_coord_1', 'central_coord_2', 'central_coord_3',
                           'side_1_coord_1', 'side_1_coord_2', 'side_1_coord_3',
                           'side_2_coord_1', 'side_2_coord_2', 'side_2_coord_3',
                           'bbox_name']
        
        try:
            # Load from paths specified in config
            excel_path = self.config.synapse_coordinates_path
            sheet_name = self.config.synapse_coordinates_sheet
            
            # First try to load from the main coordinates file
            if os.path.exists(excel_path) and os.path.isfile(excel_path):
                print(f"Loading synapse metadata from {excel_path}, sheet: {sheet_name}")
                synapse_df = pd.read_excel(excel_path, sheet_name=sheet_name)
                
                # Fill NaN values in the bbox_name column with "unknown"
                if 'bbox_name' in synapse_df.columns:
                    synapse_df['bbox_name'].fillna('unknown', inplace=True)
                    
                # Check for required columns
                missing_columns = [col for col in required_columns if col not in synapse_df.columns]
                if missing_columns:
                    print(f"Warning: Missing required columns in synapse metadata: {missing_columns}")
                    
                print(f"Loaded metadata for {len(synapse_df)} synapses")
                return synapse_df
            else:
                # If main file not found, try to load coordinate files with bbox name pattern
                print(f"Main coordinates file not found: {excel_path}")
                print("Attempting to locate coordinates files with bbox name pattern...")
                
                # Get the base directory from the main coordinates path
                base_dir = os.path.dirname(excel_path)
                
                # Get the list of bbox names from config
                bbox_names = self.config.bbox_name
                
                all_synapses = []
                
                for bbox_name in bbox_names:
                    # Look for excel file with bbox name
                    bbox_excel_path = os.path.join(base_dir, f"{bbox_name}.xlsx")
                    
                    if os.path.exists(bbox_excel_path) and os.path.isfile(bbox_excel_path):
                        print(f"Loading synapse metadata from {bbox_excel_path}, sheet: {sheet_name}")
                        try:
                            bbox_df = pd.read_excel(bbox_excel_path, sheet_name=sheet_name)
                            
                            # Add bbox_name column if not present
                            if 'bbox_name' not in bbox_df.columns:
                                bbox_df['bbox_name'] = bbox_name
                            
                            all_synapses.append(bbox_df)
                            print(f"Loaded metadata for {len(bbox_df)} synapses from {bbox_name}.xlsx")
                        except Exception as e:
                            print(f"Error loading {bbox_excel_path}: {e}")
                
                if all_synapses:
                    # Combine all dataframes
                    synapse_df = pd.concat(all_synapses, ignore_index=True)
                    
                    # Check for required columns
                    missing_columns = [col for col in required_columns if col not in synapse_df.columns]
                    if missing_columns:
                        print(f"Warning: Missing required columns in synapse metadata: {missing_columns}")
                        
                    print(f"Loaded metadata for a total of {len(synapse_df)} synapses")
                    return synapse_df
                else:
                    raise FileNotFoundError(f"Could not find any coordinate files with bbox name pattern")
        except Exception as e:
            print(f"Error loading synapse metadata: {e}")
            print("Returning an empty DataFrame with required columns.")
            # Return an empty DataFrame with the required columns
            return pd.DataFrame(columns=required_columns)
        
    def create_dataset(self, vol_data_dict=None, synapse_df=None, batch_size=8, shuffle=True, num_workers=4):
        """
        Create a contrastive learning dataset from volumetric data and synapse coordinates.
        
        Args:
            vol_data_dict: Dictionary mapping bbox_name -> (raw_vol, seg_vol, add_mask_vol)
            synapse_df: DataFrame with synapse information
            batch_size: Batch size for the dataloader
            shuffle: Whether to shuffle the dataset
            num_workers: Number of worker processes for the dataloader
            
        Returns:
            (dataset, dataloader) tuple
        """
        # Load data if not provided
        if vol_data_dict is None:
            vol_data_dict = self.load_data()
        
        # Load synapse metadata if not provided
        if synapse_df is None:
            synapse_df = self.load_synapse_metadata()
        
        # Filter synapse_df to only include bounding boxes that we have loaded
        available_bboxes = list(vol_data_dict.keys())
        synapse_df = synapse_df[synapse_df['bbox_name'].isin(available_bboxes)].reset_index(drop=True)
        
        print(f"Creating dataset with {len(synapse_df)} synapses from {len(available_bboxes)} bounding boxes")
        
        # Initialize the processor with point cloud extraction if needed
        processor = ContrastiveAugmentationProcessor(
            size=(self.config.subvol_size, self.config.subvol_size),
            augmentation_strength=self.augmentation_strength,
            use_point_cloud=self.use_point_cloud,
            max_points=self.max_points
        )
        
        # Store the processor for later use
        self.processor = processor
        
        # Create the dataset
        dataset = ContrastiveAugmentedDataset(
            vol_data_dict=vol_data_dict,
            synapse_df=synapse_df,
            processor=processor,
            subvol_size=self.config.subvol_size,
            num_frames=self.config.num_frames,
            use_point_cloud=self.use_point_cloud
        )
        
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        return dataset, dataloader


class RealContrastiveDataset(Dataset):
    """Dataset for contrastive learning with real synapse data"""
    
    def __init__(self, vol_data_dict, synapse_df, processor, subvol_size=80, num_frames=80):
        """
        Initialize the contrastive dataset.
        
        Args:
            vol_data_dict: Dictionary mapping bbox_name -> (raw_vol, seg_vol, add_mask_vol)
            synapse_df: DataFrame with synapse information
            processor: Processor for creating augmented views
            subvol_size: Size of the subvolume cube
            num_frames: Number of frames to use in the z-dimension
        """
        self.vol_data_dict = vol_data_dict
        self.synapse_df = synapse_df.reset_index(drop=True)
        self.processor = processor
        self.subvol_size = subvol_size
        self.num_frames = num_frames
        self.data_loader = SynapseDataLoader("", "", "")
    
    def __len__(self):
        return len(self.synapse_df)
    
    def __getitem__(self, idx):
        # Get synapse information
        syn_info = self.synapse_df.iloc[idx]
        bbox_name = syn_info['bbox_name']
        
        # Get volumes for this bounding box
        raw_vol, seg_vol, add_mask_vol = self.vol_data_dict.get(bbox_name, (None, None, None))
        
        if raw_vol is None:
            # Return default tensors if the volume couldn't be loaded
            empty_tensor = torch.zeros((self.num_frames, 1, self.subvol_size, self.subvol_size), dtype=torch.float32)
            return {
                'pixel_values_aug1': empty_tensor,
                'pixel_values_aug2': empty_tensor,
                'bbox_name': bbox_name,
                'syn_id': str(syn_info.get('Var1', idx))  # Use Var1 as synapse ID or fallback to index
            }
        
        # Extract coordinates for this synapse
        central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
        side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
        side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))
        
        # Extract raw and mask cubes
        raw_cube, mask_cube = self.data_loader.extract_raw_and_mask_cubes(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            subvolume_size=self.subvol_size,
            bbox_name=bbox_name,
        )
        
        # Ensure we have the required number of frames
        if raw_cube.shape[2] < self.num_frames:
            # Pad by duplicating the last frame
            raw_cube_padded = np.zeros((raw_cube.shape[0], raw_cube.shape[1], self.num_frames), dtype=np.float32)
            
            # Copy the available frames
            raw_cube_padded[:, :, :raw_cube.shape[2]] = raw_cube
            
            # Duplicate the last frame
            for z in range(raw_cube.shape[2], self.num_frames):
                raw_cube_padded[:, :, z] = raw_cube[:, :, -1]
                
            raw_cube = raw_cube_padded
            
        elif raw_cube.shape[2] > self.num_frames:
            # Sample frames evenly
            indices = np.linspace(0, raw_cube.shape[2]-1, self.num_frames, dtype=int)
            raw_cube_sampled = np.zeros((raw_cube.shape[0], raw_cube.shape[1], self.num_frames), dtype=np.float32)
            
            for i, idx in enumerate(indices):
                raw_cube_sampled[:, :, i] = raw_cube[:, :, idx]
                
            raw_cube = raw_cube_sampled
        
        # Generate two augmented views using the processor
        aug1 = self.processor.process_raw_volume(raw_cube, apply_augmentation=True)
        aug2 = self.processor.process_raw_volume(raw_cube, apply_augmentation=True)
        
        return {
            'pixel_values_aug1': aug1,
            'pixel_values_aug2': aug2,
            'bbox_name': bbox_name,
            'syn_id': str(syn_info.get('Var1', idx))  # Use Var1 as synapse ID or fallback to index
        } 