#!/usr/bin/env python3
"""
Comprehensive pipeline script for synapse analysis.
This script handles the complete workflow:
1. Data loading from local relative paths
2. Point cloud extraction
3. Model training
4. Clustering and visualization
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime
import concurrent.futures
import multiprocessing
import time
from collections import defaultdict

# Add the project root to Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary modules
from data.dataloader import SynapseDataLoader
from models.contrastive_model import PointCloudEncoder, ContrastiveModel
from trainers.train_contrastive import train_contrastive_model
from trainers.cluster_synapses import extract_features, perform_clustering, visualize_clusters
from utils.config import config

# Define a function to process a single synapse (needed for multiprocessing)
def process_single_synapse(args):
    """Process a single synapse to extract point clouds (used for parallel processing)"""
    import time
    from collections import defaultdict
    
    # Dictionary to track execution times for different operations
    timers = defaultdict(float)
    
    # Unpack arguments
    syn_info, bbox_name, raw_vol, seg_vol, add_mask_vol, subvol_size, max_points = args
    
    # Extract synapse ID
    syn_id = str(syn_info.get('Var1', syn_info.name))
    
    # Extract coordinates
    central_coord = (int(syn_info['central_coord_1']), int(syn_info['central_coord_2']), int(syn_info['central_coord_3']))
    side1_coord = (int(syn_info['side_1_coord_1']), int(syn_info['side_1_coord_2']), int(syn_info['side_1_coord_3']))
    side2_coord = (int(syn_info['side_2_coord_1']), int(syn_info['side_2_coord_2']), int(syn_info['side_2_coord_3']))
    
    try:
        # Create data loader for this process
        t_start = time.time()
        data_loader = SynapseDataLoader(None, None, None)  # No need to load volumes again
        timers['create_dataloader'] += time.time() - t_start
        
        # Extract raw and mask cubes with verbose=False to disable debug prints
        t_start = time.time()
        raw_cube, mask_cube = data_loader.extract_raw_and_mask_cubes(
            raw_vol=raw_vol,
            seg_vol=seg_vol,
            add_mask_vol=add_mask_vol,
            central_coord=central_coord,
            side1_coord=side1_coord,
            side2_coord=side2_coord,
            subvolume_size=subvol_size,
            bbox_name=bbox_name,
            verbose=False  # Important: disable verbose debug printing
        )
        timers['extract_cubes'] += time.time() - t_start
        
        # If no mask cube is found, return None
        if mask_cube is None:
            return None
        
        # Extract point clouds directly from the already selected mask components
        # (The mask selection has already been done in extract_raw_and_mask_cubes)
        cleft_mask = (mask_cube == 1)
        presynapse_mask = (mask_cube == 3)
        
        cleft_points = None
        presynapse_points = None
        
        # Only extract cleft points if any exist
        if np.any(cleft_mask):
            t_start = time.time()
            cleft_tensor = torch.from_numpy(cleft_mask.astype(np.float32))
            cleft_points = PointCloudEncoder.mask_to_point_cloud(
                cleft_tensor, num_samples=max_points
            )
            if isinstance(cleft_points, torch.Tensor):
                cleft_points = cleft_points.cpu().numpy()
            timers['extract_cleft_points'] += time.time() - t_start
        
        # Only extract presynapse points if any exist
        if np.any(presynapse_mask):
            t_start = time.time()
            presynapse_tensor = torch.from_numpy(presynapse_mask.astype(np.float32))
            presynapse_points = PointCloudEncoder.mask_to_point_cloud(
                presynapse_tensor, num_samples=max_points
            )
            if isinstance(presynapse_points, torch.Tensor):
                presynapse_points = presynapse_points.cpu().numpy()
            timers['extract_presynapse_points'] += time.time() - t_start
        
        # Only store if we have at least one type of point cloud
        if cleft_points is not None or presynapse_points is not None:
            t_start = time.time()
            synapse_data = {
                'bbox_name': bbox_name,
                'central_coord': central_coord,
                'side1_coord': side1_coord,
                'side2_coord': side2_coord,
                'subvolume_size': subvol_size,
                'raw_cube': raw_cube,
                'cleft_points': cleft_points,
                'presynapse_points': presynapse_points,
                'timers': dict(timers)  # Include timing information for analysis
            }
            timers['create_result'] += time.time() - t_start
            return (f"{bbox_name}_{syn_id}", synapse_data)
        
        return None
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing synapse {syn_id} in {bbox_name}: {e}\n{traceback.format_exc()}"
        return (None, error_msg)

class SynapsePipeline:
    """Main pipeline class for synapse analysis"""
    
    def __init__(self, args):
        """Initialize the pipeline with given arguments"""
        self.args = args
        self.setup_paths()
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "point_clouds"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "clustering"), exist_ok=True)
        
        # Set device
        self.device = torch.device(self.args.device)
        print(f"Using device: {self.device}")
        
        # Load Excel files with synapse coordinates
        self.load_excel_files()
        
        # Print configuration
        print("\nPipeline configuration:")
        print(f"  Raw data directory: {self.raw_base_dir}")
        print(f"  Segmentation directory: {self.seg_base_dir}")
        print(f"  Additional mask directory: {self.add_mask_base_dir}")
        print(f"  Excel file: {self.excel_file}")
        print(f"  Bounding boxes: {self.args.bbox_name}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Point clouds path: {self.point_clouds_path}")
        print(f"  Model path: {self.model_path}")
        print(f"  Max points: {self.args.max_points}")
        print(f"  Num synapses: {len(self.synapse_df)}")
        if hasattr(self.args, 'use_mp') and self.args.use_mp:
            print(f"  Multiprocessing: Enabled")
        else:
            print(f"  Multiprocessing: Disabled")
    
    def setup_paths(self):
        """Set up file paths for pipeline stages"""
        if hasattr(self.args, 'use_default_paths') and self.args.use_default_paths:
            # Use paths from config
            self.raw_base_dir = self.args.raw_base_dir
            self.seg_base_dir = self.args.seg_base_dir
            self.add_mask_base_dir = self.args.add_mask_base_dir
            self.excel_file = self.args.excel_file  # Fix: use excel_file instead of excel_dir
        else:
            # Use paths from args
            self.raw_base_dir = self.args.raw_base_dir
            self.seg_base_dir = self.args.seg_base_dir
            self.add_mask_base_dir = self.args.add_mask_base_dir
            self.excel_file = self.args.excel_file
        
        # Output directory
        self.output_dir = os.path.join("results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Point clouds
        self.point_clouds_dir = os.path.join(self.output_dir, "point_clouds")
        self.point_clouds_path = os.path.join(self.point_clouds_dir, "synapse_point_clouds.pkl")
        
        # Models
        self.models_dir = os.path.join(self.output_dir, "models")
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_path = os.path.join(self.models_dir, f"contrastive_model_epoch_{self.args.epochs}.pth")
        
        # Clustering
        self.clustering_dir = os.path.join(self.output_dir, "clustering")
        os.makedirs(self.clustering_dir, exist_ok=True)
        self.clustering_results_path = os.path.join(self.clustering_dir, "clustering_results.csv")
        
        # Visualizations
        self.visualizations_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(self.visualizations_dir, exist_ok=True)
    
    def load_excel_files(self):
        """Load synapse metadata from Excel files"""
        all_synapses = []
        
        # For each bounding box, find and load the corresponding Excel file
        for bbox in self.args.bbox_name:
            excel_path = os.path.join(self.excel_file, f"{bbox}.xlsx")
            if os.path.exists(excel_path):
                try:
                    df = pd.read_excel(excel_path)
                    print(f"Loaded {len(df)} synapses from {excel_path}")
                    
                    # Add bbox_name column if not present
                    if 'bbox_name' not in df.columns:
                        df['bbox_name'] = bbox
                    
                    all_synapses.append(df)
                except Exception as e:
                    print(f"Error loading {excel_path}: {e}")
        
        # Combine all DataFrames
        if all_synapses:
            self.synapse_df = pd.concat(all_synapses, ignore_index=True)
            print(f"Combined dataset contains {len(self.synapse_df)} synapses")
            
            # Limit samples if specified
            if self.args.max_samples and len(self.synapse_df) > self.args.max_samples:
                self.synapse_df = self.synapse_df.sample(self.args.max_samples, random_state=42)
                print(f"Limited to {self.args.max_samples} random samples")
            
            return True
        else:
            print("No valid Excel files found")
            return False
    
    def extract_point_clouds(self):
        """Extract point clouds from the dataset"""
        print("\n--- Extracting Point Clouds ---")
        
        # Skip if already completed and not forced
        if os.path.exists(self.point_clouds_path) and not self.args.force_extract:
            print(f"Point clouds already exist at {self.point_clouds_path}. Use --force_extract to regenerate.")
            return True
        
        # Initialize data loader
        data_loader = SynapseDataLoader(
            raw_base_dir=self.raw_base_dir,
            seg_base_dir=self.seg_base_dir,
            add_mask_base_dir=self.add_mask_base_dir
        )
        
        point_clouds = {}
        errors = []
        
        # Check if multiprocessing should be enabled
        if hasattr(self.args, 'use_mp') and self.args.use_mp:
            print("Multiprocessing enabled, using parallel extraction")
            
            # Determine number of workers for parallel processing
            max_workers = min(multiprocessing.cpu_count(), 8)  # Use up to 8 cores
            print(f"Using {max_workers} CPU cores for parallel processing")
            
            # Process each bounding box
            for bbox_name in self.args.bbox_name:
                print(f"\nProcessing bounding box: {bbox_name}")
                
                # Load volumes for this bounding box
                raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
                if raw_vol is None:
                    print(f"Failed to load volumes for {bbox_name}, skipping...")
                    continue
                
                print(f"Loaded volumes: raw={raw_vol.shape}, seg={seg_vol.shape}, add_mask={add_mask_vol.shape if add_mask_vol is not None else None}")
                
                # Get synapses for this bounding box
                bbox_df = self.synapse_df[self.synapse_df['bbox_name'] == bbox_name].reset_index(drop=True)
                if len(bbox_df) == 0:
                    print(f"No synapses found for {bbox_name}, skipping...")
                    continue
                
                print(f"Processing {len(bbox_df)} synapses in {bbox_name} in parallel")
                
                # Prepare arguments for parallel processing
                process_args = [
                    (syn_info, bbox_name, raw_vol, seg_vol, add_mask_vol, self.args.subvol_size, self.args.max_points)
                    for _, syn_info in bbox_df.iterrows()
                ]
                
                # Process synapses in parallel with progress bar
                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = list(tqdm(
                        executor.map(process_single_synapse, process_args),
                        total=len(process_args),
                        desc=f"Processing {bbox_name}"
                    ))
                    
                    # Collect results
                    for result in results:
                        if result is None:
                            continue
                            
                        key, value = result
                        if key is None:
                            # This is an error message
                            errors.append(value)
                        else:
                            # This is a successful result
                            point_clouds[key] = value
        else:
            print("Multiprocessing disabled, using single-process extraction")
            return self._extract_point_clouds_single_process(data_loader)
        
        # Report any errors
        if errors:
            print(f"\n{len(errors)} errors occurred during processing:")
            for i, error in enumerate(errors[:5]):  # Only show first 5 errors
                print(f"Error {i+1}: {error}")
            if len(errors) > 5:
                print(f"... and {len(errors) - 5} more errors.")
        
        # Save results
        if len(point_clouds) > 0:
            print(f"\nSaving {len(point_clouds)} point clouds to {self.point_clouds_path}")
            os.makedirs(os.path.dirname(self.point_clouds_path), exist_ok=True)
            with open(self.point_clouds_path, 'wb') as f:
                pickle.dump(point_clouds, f)
            
            # Save metadata CSV
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
            metadata_path = os.path.join(os.path.dirname(self.point_clouds_path), "point_clouds_metadata.csv")
            metadata_df.to_csv(metadata_path, index=False)
            print(f"Saved metadata to {metadata_path}")
            
            return True
        else:
            print("No valid point clouds extracted")
            return False
            
    def _extract_point_clouds_single_process(self, data_loader):
        """Extract point clouds using a single process (no multiprocessing)"""
        # Dictionary to track execution times for different operations
        timers = defaultdict(float)
        
        point_clouds = {}
        total_synapses = len(self.synapse_df)
        
        # Create overall progress bar
        overall_pbar = tqdm(total=total_synapses, desc="Overall Progress", position=0)
        
        # Process each bounding box
        for bbox_name in self.args.bbox_name:
            print(f"\nProcessing bounding box: {bbox_name}")
            
            # Load volumes for this bounding box
            t_start = time.time()
            raw_vol, seg_vol, add_mask_vol = data_loader.load_volumes(bbox_name)
            timers['load_volumes'] += time.time() - t_start
            
            if raw_vol is None:
                print(f"Failed to load volumes for {bbox_name}, skipping...")
                continue
            
            print(f"Loaded volumes: raw={raw_vol.shape}, seg={seg_vol.shape}, add_mask={add_mask_vol.shape if add_mask_vol is not None else None}")
            
            # Get synapses for this bounding box
            t_start = time.time()
            bbox_df = self.synapse_df[self.synapse_df['bbox_name'] == bbox_name].reset_index(drop=True)
            timers['filter_synapses'] += time.time() - t_start
            
            if len(bbox_df) == 0:
                print(f"No synapses found for {bbox_name}, skipping...")
                continue
            
            print(f"Processing {len(bbox_df)} synapses in {bbox_name}")
            
            # Process each synapse with a dedicated progress bar for this bbox
            bbox_pbar = tqdm(total=len(bbox_df), desc=f"Bbox {bbox_name}", position=1, leave=False)
            
            # Process each synapse
            for idx, syn_info in bbox_df.iterrows():
                # Time extraction of cubes
                t_start = time.time()
                
                # Process single synapse using the wrapper function
                result = process_single_synapse((syn_info, bbox_name, raw_vol, seg_vol, add_mask_vol, self.args.subvol_size, self.args.max_points))
                
                timers['process_synapse'] += time.time() - t_start
                
                if result is not None:
                    key, value = result
                    if key is not None:
                        point_clouds[key] = value
                
                # Update progress bars
                bbox_pbar.update(1)
                overall_pbar.update(1)
            
            # Close bbox progress bar
            bbox_pbar.close()
        
        # Close overall progress bar
        overall_pbar.close()
        
        # Print timing statistics
        print("\nTiming statistics:")
        total_time = sum(timers.values())
        for operation, duration in sorted(timers.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"  {operation}: {duration:.2f}s ({percentage:.1f}%)")
        
        # Save results
        if len(point_clouds) > 0:
            print(f"\nSaving {len(point_clouds)} point clouds to {self.point_clouds_path}")
            os.makedirs(os.path.dirname(self.point_clouds_path), exist_ok=True)
            with open(self.point_clouds_path, 'wb') as f:
                pickle.dump(point_clouds, f)
            
            # Save metadata CSV
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
            metadata_path = os.path.join(os.path.dirname(self.point_clouds_path), "point_clouds_metadata.csv")
            metadata_df.to_csv(metadata_path, index=False)
            print(f"Saved metadata to {metadata_path}")
            
            return True
        else:
            print("No valid point clouds extracted")
            return False
    
    def train_model(self):
        """Train the contrastive learning model"""
        print("\n--- Training Contrastive Model ---")
        
        # Skip if already completed and not forced
        if os.path.exists(self.model_path) and not self.args.force_train:
            print(f"Model already exists at {self.model_path}. Use --force_train to retrain.")
            return True
        
        # Check if point clouds exist
        if not os.path.exists(self.point_clouds_path):
            print(f"Point clouds not found at {self.point_clouds_path}. Please extract point clouds first.")
            return False
        
        # Load point clouds
        print(f"Loading point clouds from {self.point_clouds_path}")
        with open(self.point_clouds_path, 'rb') as f:
            point_clouds = pickle.load(f)
        
        print(f"Loaded {len(point_clouds)} point clouds")
        
        # Set up model configuration
        if self.args.model_type == 'contrastive':
            # Initialize the contrastive model
            model = ContrastiveModel(
                in_channels=config.in_channels,
                encoder_dim=self.args.encoder_dim,
                projection_dim=self.args.projection_dim
            ).to(self.device)
            
            # Train the model
            train_contrastive_model(
                model=model,
                point_clouds=point_clouds,
                batch_size=self.args.batch_size,
                epochs=self.args.epochs,
                learning_rate=self.args.learning_rate,
                model_save_path=os.path.dirname(self.model_path),
                model_save_prefix='contrastive_model',
                device=self.device
            )
        else:
            print(f"Model type '{self.args.model_type}' not supported")
            return False
        
        return True
    
    def cluster_synapses(self):
        """Cluster synapses based on their features from the trained model"""
        print("\n--- Clustering Synapses ---")
        
        # Check if clustering is needed or if results already exist
        clustering_output_dir = os.path.join(self.output_dir, "clustering")
        if os.path.exists(clustering_output_dir) and not self.args.force_cluster:
            print(f"Clustering results already exist at {clustering_output_dir}")
            print("Use --force_cluster to re-run clustering")
            return True
        
        # Check if model exists
        model_path = self._get_model_path()
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Train the model first.")
            return False
        
        # Check if point clouds exist
        point_clouds_path = self._get_point_clouds_path()
        if not os.path.exists(point_clouds_path):
            print(f"Point clouds not found at {point_clouds_path}. Extract point clouds first.")
            return False
            
        # Load point clouds
        print(f"Loading pre-extracted point clouds from {point_clouds_path}...")
        try:
            with open(point_clouds_path, 'rb') as f:
                point_clouds_data = pickle.load(f)
                
            num_synapses = len(point_clouds_data)
            print(f"Loaded point clouds for {num_synapses} synapses")
            
            if num_synapses == 0:
                print("No point clouds found. Extract point clouds first.")
                return False
        except Exception as e:
            print(f"Error loading point clouds: {e}")
            return False
        
        # Load model and extract features
        try:
            from models.contrastive_model import ContrastiveModel
            
            print(f"Loading model from {model_path}...")
            model_data = torch.load(model_path, map_location=self.args.device)
            
            # Check model structure
            if 'model_state_dict' in model_data:
                print("Detected full model structure")
                model_state = model_data['model_state_dict']
                
                # Create a new model with the saved configuration
                encoder_dim = model_data.get('encoder_dim', self.args.encoder_dim)
                projection_dim = model_data.get('projection_dim', self.args.projection_dim)
                max_points = model_data.get('max_points', self.args.max_points)
                
                model = ContrastiveModel(
                    encoder_dim=encoder_dim,
                    projection_dim=projection_dim,
                    max_points=max_points
                )
                model.load_state_dict(model_state)
            elif 'encoder_state_dict' in model_data:
                print("Detected single encoder model structure")
                encoder_state = model_data['encoder_state_dict']
                
                # Create a new model with the saved configuration
                encoder_dim = model_data.get('encoder_dim', self.args.encoder_dim)
                projection_dim = model_data.get('projection_dim', self.args.projection_dim)
                max_points = model_data.get('max_points', self.args.max_points)
                
                model = ContrastiveModel(
                    encoder_dim=encoder_dim,
                    projection_dim=projection_dim,
                    max_points=max_points
                )
                
                # Only load encoder weights
                model.point_cloud_encoder.load_state_dict(encoder_state)
                print("Found nested encoder structure, creating new model and extracting encoder weights")
            else:
                # Try direct loading (legacy models)
                model = ContrastiveModel(
                    encoder_dim=self.args.encoder_dim,
                    projection_dim=self.args.projection_dim,
                    max_points=self.args.max_points
                )
                model.load_state_dict(model_data)
                
            model.to(self.args.device)
            model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
        # Create an empty array to store features
        features = []
        synapse_ids = []
        
        # Set up progress bar
        from tqdm import tqdm
        
        try:
            # Extract features from all point clouds
            with torch.no_grad():
                for idx, (synapse_id, synapse_data) in enumerate(tqdm(point_clouds_data.items(), 
                                                                     desc="Extracting features",
                                                                     total=len(point_clouds_data))):
                    # Get cleft point cloud
                    if 'cleft' in synapse_data:
                        point_cloud = synapse_data['cleft']
                    else:
                        # Skip synapses without cleft point clouds
                        continue
                        
                    if isinstance(point_cloud, np.ndarray):
                        point_cloud = torch.from_numpy(point_cloud).float()
                        
                    if len(point_cloud) == 0:
                        # Skip empty point clouds
                        continue
                        
                    # Ensure points are on the correct device
                    point_cloud = point_cloud.to(self.args.device)
                    
                    # Extract features
                    feature = model.point_cloud_encoder(point_cloud.unsqueeze(0))
                    
                    # Save features
                    features.append(feature.squeeze().cpu().numpy())
                    synapse_ids.append(synapse_id)
                    
            # Convert to numpy array
            features = np.stack(features)
            
            # Perform clustering
            print(f"Performing {self.args.clustering_algorithm.upper()} clustering with {self.args.n_clusters} clusters...")
            
            # Dimensionality reduction using PCA (for visualization and to improve clustering)
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # First, standardize the features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply PCA for dimensionality reduction (50 components or fewer)
            n_components = min(50, features_scaled.shape[0], features_scaled.shape[1])
            pca = PCA(n_components=n_components)
            features_pca = pca.fit_transform(features_scaled)
            
            print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")
            
            # Perform clustering on PCA-reduced features
            if self.args.clustering_algorithm == 'kmeans':
                from sklearn.cluster import KMeans
                
                # Perform KMeans clustering
                kmeans = KMeans(n_clusters=self.args.n_clusters, random_state=0, n_init=10)
                cluster_labels = kmeans.fit_predict(features_pca)
                
            elif self.args.clustering_algorithm == 'dbscan':
                from sklearn.cluster import DBSCAN
                
                # Perform DBSCAN clustering
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = dbscan.fit_predict(features_pca)
                
            else:
                print(f"Unsupported clustering algorithm: {self.args.clustering_algorithm}")
                return False
            
            # Apply UMAP for visualization (replacing t-SNE)
            import umap
            
            print(f"Reducing dimensionality with UMAP...")
            umap_reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=15)
            features_umap = umap_reducer.fit_transform(features_pca)
            
            # Visualize clusters
            import matplotlib.pyplot as plt
            
            # Create the output directory if it doesn't exist
            os.makedirs(clustering_output_dir, exist_ok=True)
            
            # Plot UMAP visualization with cluster colors
            plt.figure(figsize=(10, 10))
            scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], c=cluster_labels, cmap='tab10', s=50, alpha=0.8)
            
            # Add legend
            legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
            plt.gca().add_artist(legend1)
            
            plt.title(f"UMAP visualization of clusters ({self.args.clustering_algorithm.upper()})")
            plt.xlabel("UMAP dimension 1")
            plt.ylabel("UMAP dimension 2")
            plt.tight_layout()
            plt.savefig(os.path.join(clustering_output_dir, f"umap_clusters_{self.args.clustering_algorithm}.png"), dpi=300)
            
            # Save cluster results
            results = {
                'synapse_ids': synapse_ids,
                'features': features,
                'features_pca': features_pca,
                'features_umap': features_umap,  # Renamed from features_tsne
                'cluster_labels': cluster_labels,
                'pca_explained_variance': pca.explained_variance_ratio_,
                'clustering_params': {
                    'algorithm': self.args.clustering_algorithm,
                    'n_clusters': self.args.n_clusters
                }
            }
            
            # Save the results
            with open(os.path.join(clustering_output_dir, 'clustering_results.pkl'), 'wb') as f:
                pickle.dump(results, f)
                
            # Create a mapping from synapse ID to cluster label
            cluster_mapping = dict(zip(synapse_ids, cluster_labels))
            
            # Save as CSV for easy analysis
            import pandas as pd
            df = pd.DataFrame({
                'synapse_id': synapse_ids,
                'cluster': cluster_labels
            })
            df.to_csv(os.path.join(clustering_output_dir, 'cluster_assignments.csv'), index=False)
            
            print(f"Visualization complete. Results saved to {clustering_output_dir}")
            print(f"Clustering results saved to {clustering_output_dir}")
            return True
            
        except Exception as e:
            print(f"Error during clustering: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def visualize_results(self):
        """Visualize the clustering results with additional plots and analysis"""
        print("\n--- Visualizing Results ---")
        
        # Check if clustering results exist
        clustering_output_dir = os.path.join(self.output_dir, "clustering")
        clustering_results_path = os.path.join(clustering_output_dir, 'clustering_results.pkl')
        
        if not os.path.exists(clustering_results_path):
            print(f"Clustering results not found at {clustering_results_path}")
            print("Run clustering first with --run_cluster")
            return False
            
        # Load clustering results
        try:
            with open(clustering_results_path, 'rb') as f:
                results = pickle.load(f)
                
            synapse_ids = results['synapse_ids']
            features = results['features']
            features_pca = results['features_pca']
            
            # Check if we have UMAP features or need to compute them
            if 'features_umap' in results:
                features_umap = results['features_umap']
            else:
                # If we have t-SNE features from previous runs but not UMAP, compute UMAP now
                print("Computing UMAP embeddings (not found in results)...")
                import umap
                umap_reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.1, n_neighbors=15)
                features_umap = umap_reducer.fit_transform(features_pca)
            
            cluster_labels = results['cluster_labels']
            
            print(f"Loaded clustering results for {len(synapse_ids)} synapses")
            
            # Create visualization output directory
            vis_output_dir = os.path.join(self.output_dir, "visualization")
            os.makedirs(vis_output_dir, exist_ok=True)
            
            # Import visualization libraries
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            from sklearn.metrics import silhouette_score, calinski_harabasz_score
            
            # 1. Enhanced UMAP plot with different parameters
            print("Creating enhanced UMAP visualization...")
            plt.figure(figsize=(12, 10))
            
            # Use a more distinctive colormap for better separation
            scatter = plt.scatter(features_umap[:, 0], features_umap[:, 1], 
                                 c=cluster_labels, cmap='viridis', 
                                 s=80, alpha=0.8, edgecolors='w', linewidths=0.5)
            
            # Add legend for clusters
            legend = plt.legend(*scatter.legend_elements(), 
                               title="Clusters", 
                               loc="upper right", 
                               frameon=True,
                               framealpha=0.85)
            
            # Enhance the legend
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            
            plt.title("Synapse Clusters - UMAP Visualization", fontsize=16)
            plt.xlabel("UMAP Component 1", fontsize=14)
            plt.ylabel("UMAP Component 2", fontsize=14)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            # Save high-resolution figure
            plt.savefig(os.path.join(vis_output_dir, "enhanced_umap_clusters.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Cluster statistics
            print("Calculating cluster statistics...")
            
            # Count synapses per cluster
            unique_clusters = np.unique(cluster_labels)
            cluster_counts = {cluster: np.sum(cluster_labels == cluster) for cluster in unique_clusters}
            
            # Calculate cluster quality metrics if there are multiple clusters
            if len(unique_clusters) > 1:
                # Silhouette score (higher is better)
                silhouette_avg = silhouette_score(features_pca, cluster_labels)
                
                # Calinski-Harabasz Index (higher is better)
                calinski_score = calinski_harabasz_score(features_pca, cluster_labels)
                
                # Create statistics DataFrame
                stats_df = pd.DataFrame({
                    'Cluster': list(cluster_counts.keys()),
                    'Synapse Count': list(cluster_counts.values()),
                    'Percentage': [count/len(synapse_ids)*100 for count in cluster_counts.values()]
                })
                
                # Fix: Use concat instead of append (which is deprecated)
                total_row = pd.DataFrame({
                    'Cluster': ['Total'],
                    'Synapse Count': [len(synapse_ids)],
                    'Percentage': [100.0]
                })
                stats_df = pd.concat([stats_df, total_row], ignore_index=True)
                
                # Save statistics to CSV
                stats_df.to_csv(os.path.join(vis_output_dir, "cluster_statistics.csv"), index=False)
                
                # Create bar chart of cluster sizes
                plt.figure(figsize=(12, 6))
                bars = plt.bar(
                    [f"Cluster {c}" for c in unique_clusters], 
                    [cluster_counts[c] for c in unique_clusters],
                    color=plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
                )
                
                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom'
                    )
                
                plt.title(f"Synapse Count per Cluster (Silhouette Score: {silhouette_avg:.3f})", fontsize=16)
                plt.xlabel("Cluster", fontsize=14)
                plt.ylabel("Number of Synapses", fontsize=14)
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(vis_output_dir, "cluster_sizes.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. PCA components visualization
                if 'pca_explained_variance' in results:
                    pca_variance = results['pca_explained_variance']
                    
                    # Plot cumulative explained variance
                    plt.figure(figsize=(10, 6))
                    cumulative_variance = np.cumsum(pca_variance)
                    components = range(1, len(pca_variance) + 1)
                    
                    plt.plot(components, cumulative_variance, 'b-', linewidth=2, marker='o')
                    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% Explained Variance')
                    
                    # Find component that explains 90% variance
                    comp_90 = np.where(cumulative_variance >= 0.9)[0][0] + 1
                    plt.axvline(x=comp_90, color='g', linestyle='--', alpha=0.5, 
                              label=f'90% at {comp_90} Components')
                    
                    plt.title("Cumulative Explained Variance by PCA Components", fontsize=16)
                    plt.xlabel("Number of Components", fontsize=14)
                    plt.ylabel("Cumulative Explained Variance", fontsize=14)
                    plt.grid(alpha=0.3)
                    plt.legend(frameon=True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_output_dir, "pca_explained_variance.png"), dpi=300)
                    plt.close()
            
            # 4. Create a text report with key findings
            with open(os.path.join(vis_output_dir, "visualization_report.txt"), 'w') as f:
                f.write("=== Synapse Clustering Analysis ===\n\n")
                f.write(f"Total synapses analyzed: {len(synapse_ids)}\n")
                f.write(f"Number of clusters: {len(unique_clusters)}\n\n")
                
                f.write("Cluster sizes:\n")
                for cluster in sorted(unique_clusters):
                    count = cluster_counts[cluster]
                    percentage = count / len(synapse_ids) * 100
                    f.write(f"  Cluster {cluster}: {count} synapses ({percentage:.1f}%)\n")
                
                f.write("\nCluster quality metrics:\n")
                if len(unique_clusters) > 1:
                    f.write(f"  Silhouette Score: {silhouette_avg:.3f} (higher is better, range: [-1, 1])\n")
                    f.write(f"  Calinski-Harabasz Index: {calinski_score:.1f} (higher is better)\n\n")
                else:
                    f.write("  N/A - need multiple clusters for quality metrics\n\n")
                
                f.write("Visualization files:\n")
                f.write(f"  - Enhanced UMAP: {os.path.join(vis_output_dir, 'enhanced_umap_clusters.png')}\n")
                f.write(f"  - Cluster sizes: {os.path.join(vis_output_dir, 'cluster_sizes.png')}\n")
                if 'pca_explained_variance' in results:
                    f.write(f"  - PCA variance: {os.path.join(vis_output_dir, 'pca_explained_variance.png')}\n")
                
                f.write("\n=== End of Report ===\n")
            
            print(f"Visualization complete. Results saved to {vis_output_dir}")
            return True
            
        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_model_path(self):
        """Helper method to get the model path"""
        return self.model_path
    
    def _get_point_clouds_path(self):
        """Helper method to get the point clouds path"""
        return self.point_clouds_path
    
    def run(self):
        """Run the complete pipeline"""
        print("\n=== Starting Synapse Analysis Pipeline ===\n")
        
        # Step 1: Load Excel files
        if not self.load_excel_files():
            print("Failed to load Excel files, exiting pipeline")
            return False
        
        # Step 2: Extract point clouds
        if self.args.run_extract and not self.extract_point_clouds():
            print("Failed to extract point clouds, exiting pipeline")
            return False
        
        # Step 3: Train the model
        if self.args.run_train and not self.train_model():
            print("Failed to train the model, exiting pipeline")
            return False
        
        # Step 4: Perform clustering
        if self.args.run_cluster and not self.cluster_synapses():
            print("Failed to perform clustering, exiting pipeline")
            return False
        
        # Step 5: Visualize results
        if self.args.run_visualize and not self.visualize_results():
            print("Failed to visualize results, exiting pipeline")
            return False
        
        print("\n=== Pipeline completed successfully ===")
        return True


def main():
    """Main function to run the synapse pipeline"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Synapse Pipeline")
    
    # Pipeline control flags
    parser.add_argument('--run_all', action='store_true', help='Run all pipeline stages')
    parser.add_argument('--run_extract', action='store_true', help='Run point cloud extraction')
    parser.add_argument('--run_train', action='store_true', help='Run model training')
    parser.add_argument('--run_cluster', action='store_true', help='Run clustering')
    parser.add_argument('--run_visualize', action='store_true', help='Run visualization')
    parser.add_argument('--no_mp', action='store_true', default=True, help='Disable multiprocessing for point cloud extraction (enabled by default)')
    parser.add_argument('--use_mp', action='store_true', help='Enable multiprocessing for point cloud extraction (disabled by default)')
    
    # Force flags
    parser.add_argument('--force_extract', action='store_true', help='Force re-extraction of point clouds')
    parser.add_argument('--force_train', action='store_true', help='Force re-training of model')
    parser.add_argument('--force_cluster', action='store_true', help='Force re-clustering')
    
    # Model configuration
    parser.add_argument('--model_type', type=str, default='contrastive', 
                      choices=['contrastive'], help='Type of model to train')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device for model training (cuda or cpu)')
    
    # Paths and configuration
    parser.add_argument('--use_default_paths', action='store_true', default=True,
                       help='Use default paths from config')
    parser.add_argument('--no_default_paths', action='store_true', 
                       help='Override use_default_paths to disable using default paths')
    parser.add_argument('--raw_base_dir', type=str, help='Base directory for raw TIF files')
    parser.add_argument('--seg_base_dir', type=str, help='Base directory for segmentation TIF files')
    parser.add_argument('--add_mask_base_dir', type=str, help='Base directory for additional mask TIF files')
    parser.add_argument('--bbox_name', type=str, nargs='+', help='Bounding box names to process')
    parser.add_argument('--excel_file', type=str, help='Path to Excel file with synapse coordinates')
    parser.add_argument('--subvol_size', type=int, default=80, help='Size of subvolume to extract')
    parser.add_argument('--output_dir', type=str, default='results', help='Base output directory')
    
    # Model parameters
    parser.add_argument('--encoder_dim', type=int, default=256, help='Encoder output dimension')
    parser.add_argument('--projection_dim', type=int, default=128, help='Projection head output dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_points', type=int, default=2048, help='Maximum points in point cloud')
    
    # Limit samples for debugging/testing
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of samples to process (for debugging)')
    
    # Clustering parameters
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for K-means')
    parser.add_argument('--clustering_algorithm', type=str, default='kmeans',
                       choices=['kmeans', 'dbscan'], help='Clustering algorithm')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle conflicting multiprocessing flags
    if args.use_mp:
        args.no_mp = False  # If --use_mp is specified, override --no_mp
    
    # Handle conflicting paths flags
    if args.no_default_paths:
        args.use_default_paths = False
    
    # If using default paths, override with config values
    if args.use_default_paths:
        args.raw_base_dir = config.raw_base_dir
        args.seg_base_dir = config.seg_base_dir
        args.add_mask_base_dir = config.add_mask_base_dir
        args.bbox_name = config.bbox_name
        args.excel_file = config.excel_file
        args.subvol_size = config.subvol_size
        
        # Model parameters can also be taken from config
        if not hasattr(args, 'max_points') or args.max_points == 2048:
            args.max_points = config.max_points
    
    # Ensure bbox_name has a default value if it's still None
    if args.bbox_name is None:
        args.bbox_name = ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']
        print(f"Using default bounding box names: {args.bbox_name}")
    
    # Create and run the pipeline
    pipeline = SynapsePipeline(args)
    
    # Run the pipeline stages as requested
    if args.run_all or args.run_extract:
        if not pipeline.extract_point_clouds():
            print("Point cloud extraction failed.")
            return
    
    if args.run_all or args.run_train:
        if not pipeline.train_model():
            print("Model training failed.")
            return
    
    if args.run_all or args.run_cluster:
        if not pipeline.cluster_synapses():
            print("Clustering failed.")
            return
    
    if args.run_all or args.run_visualize:
        if not pipeline.visualize_results():
            print("Visualization failed.")
            return
    
    print("\n--- Pipeline completed successfully ---")


if __name__ == "__main__":
    main() 