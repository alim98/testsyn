import argparse
import os

class SynapseConfig:
    def __init__(self):
        self.raw_base_dir = r'synData\7_bboxes_plus_seg\raw'
        self.seg_base_dir = r'synData\7_bboxes_plus_seg\seg'
        self.add_mask_base_dir = r'synData\vesicle_cloud__syn_interface__mitochondria_annotation'
        # self.bbox_name = ['bbox1']
        self.bbox_name = ['bbox1', 'bbox2', 'bbox3', 'bbox4', 'bbox5', 'bbox6', 'bbox7']
        self.excel_file = r'synData\7_bboxes_plus_seg'
        # Add synapse_coordinates_path as an alias for excel_file
        self.synapse_coordinates_path = self.excel_file
        self.size = (80, 80)
        self.subvol_size = 80
        self.num_frames = 80
        self.alpha = 1.0
        self.segmentation_type = 10
        
        self.gray_color = 0.6
        
        self.clustering_output_dir = 'results/clustering_results_final'
        self.report_output_dir = 'results/comprehensive_reports'
        self.csv_output_dir = 'results/csv_output'
        self.save_gifs_dir = 'results/gifs'
        
        # Clustering parameters
        self.clustering_algorithm = 'KMeans'  # Default clustering algorithm
        self.n_clusters = 10 # Default number of clusters for KMeans
        self.dbscan_eps = 0.5  # Default epsilon parameter for DBSCAN
        self.dbscan_min_samples = 5  # Default min_samples parameter for DBSCAN
        
        # Feature extraction parameters
        self.extraction_method = "standard"  # Options: "standard" or "stage_specific"
        self.layer_num = 20  # Layer to extract features from when using stage_specific method
        self.preprocessing = 'intelligent_cropping'  # Options: 'normal' or 'intelligent_cropping'
        self.preprocessing_weights = 0.7 # it has opitons like 0.3 0.5 and 0.7
        
        # Contrastive Learning Parameters
        self.contrastive_output_dir = 'results/contrastive_models'
        self.in_channels = 1  # Input channels for the encoder (1 for grayscale)
        self.encoder_dim = 256  # Output dimension of the encoder
        self.projection_dim = 128  # Output dimension of the projection head
        self.augmentation_strength = 'medium'  # Strength for contrastive augmentations (light, medium, strong)
        self.batch_size = 16 # Batch size for training
        self.epochs = 100 # Number of training epochs
        self.learning_rate = 1e-4 # Learning rate for optimizer
        self.weight_decay = 1e-6 # Weight decay for optimizer
        self.temperature = 0.1  # Temperature for NT-Xent loss
        self.save_every_epochs = 10 # How often to save the model checkpoint
        
        # MultiModal training parameters
        self.max_points = 512  # Maximum number of points per structure type
        self.device = 'cuda'  # Device to use for training (cuda or cpu)
        self.synapse_coordinates_sheet = 'Sheet1'  # Sheet name in the coordinates Excel file
        
        # Visualization parameters
        self.visualization_output_dir = 'results/visualizations'  # Output directory for visualizations
        self.show_plots = False  # Show plots instead of saving them
        self.num_samples_to_visualize = 5  # Number of samples to visualize
        
    def parse_args(self):
        parser = argparse.ArgumentParser(description="Synapse Dataset Configuration")
        parser.add_argument('--raw_base_dir', type=str, default=self.raw_base_dir)
        parser.add_argument('--seg_base_dir', type=str, default=self.seg_base_dir)
        parser.add_argument('--add_mask_base_dir', type=str, default=self.add_mask_base_dir)
        parser.add_argument('--bbox_name', type=str, default=self.bbox_name, nargs='+')
        parser.add_argument('--excel_file', type=str, default=self.excel_file)
        # Add synapse_coordinates_path as a command-line argument alias for excel_file
        parser.add_argument('--synapse_coordinates_path', type=str, default=self.synapse_coordinates_path,
                           help='Path to synapse coordinates Excel file or directory containing bbox*.xlsx files')
        parser.add_argument('--csv_output_dir', type=str, default=self.csv_output_dir)
        parser.add_argument('--size', type=tuple, default=self.size)
        parser.add_argument('--subvol_size', type=int, default=self.subvol_size)
        parser.add_argument('--num_frames', type=int, default=self.num_frames)
        parser.add_argument('--save_gifs_dir', type=str, default=self.save_gifs_dir)
        parser.add_argument('--alpha', type=float, default=self.alpha)
        parser.add_argument('--segmentation_type', type=int, default=self.segmentation_type, 
                           choices=range(0, 13), help='Type of segmentation overlay')
        parser.add_argument('--gray_color', type=float, default=self.gray_color,
                           help='Gray color value (0-1) for overlaying segmentation')
        parser.add_argument('--clustering_output_dir', type=str, default=self.clustering_output_dir)
        parser.add_argument('--report_output_dir', type=str, default=self.report_output_dir)
        
        # Clustering parameters
        parser.add_argument('--clustering_algorithm', type=str, default=self.clustering_algorithm,
                           choices=['KMeans', 'DBSCAN'], help='Clustering algorithm to use')
        parser.add_argument('--n_clusters', type=int, default=self.n_clusters,
                           help='Number of clusters for KMeans')
        parser.add_argument('--dbscan_eps', type=float, default=self.dbscan_eps,
                           help='Epsilon parameter for DBSCAN')
        parser.add_argument('--dbscan_min_samples', type=int, default=self.dbscan_min_samples,
                           help='Minimum samples parameter for DBSCAN')
        
        # Feature extraction parameters
        parser.add_argument('--extraction_method', type=str, default=self.extraction_method,
                           choices=['standard', 'stage_specific'], 
                           help='Method to extract features ("standard" or "stage_specific")')
        parser.add_argument('--layer_num', type=int, default=self.layer_num,
                           help='Layer number to extract features from when using stage_specific method')
        
        # Add contrastive args
        parser.add_argument('--contrastive_output_dir', type=str, default=self.contrastive_output_dir)
        parser.add_argument('--in_channels', type=int, default=self.in_channels)
        parser.add_argument('--encoder_dim', type=int, default=self.encoder_dim)
        parser.add_argument('--projection_dim', type=int, default=self.projection_dim)
        parser.add_argument('--augmentation_strength', type=str, default=self.augmentation_strength,
                           choices=['light', 'medium', 'strong'], 
                           help='Strength of contrastive augmentations')
        parser.add_argument('--batch_size', type=int, default=self.batch_size)
        parser.add_argument('--epochs', type=int, default=self.epochs)
        parser.add_argument('--learning_rate', type=float, default=self.learning_rate)
        parser.add_argument('--weight_decay', type=float, default=self.weight_decay)
        parser.add_argument('--temperature', type=float, default=self.temperature)
        parser.add_argument('--save_every_epochs', type=int, default=self.save_every_epochs)
        
        # Add multimodal training parameters
        parser.add_argument('--max_points', type=int, default=self.max_points,
                           help='Maximum number of points per structure type')
        parser.add_argument('--device', type=str, default=self.device,
                           choices=['cuda', 'cpu'], help='Device to use for training')
        parser.add_argument('--synapse_coordinates_sheet', type=str, default=self.synapse_coordinates_sheet,
                           help='Sheet name in the coordinates Excel file')
        
        # Add visualization parameters
        parser.add_argument('--visualization_output_dir', type=str, default=self.visualization_output_dir,
                           help='Output directory for visualizations')
        parser.add_argument('--show_plots', action='store_true', default=self.show_plots,
                           help='Show plots instead of saving them')
        parser.add_argument('--num_samples_to_visualize', type=int, default=self.num_samples_to_visualize,
                           help='Number of samples to visualize')
        
        args, _ = parser.parse_known_args()
        
        # Apply the arguments
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        # Ensure excel_file and synapse_coordinates_path are synchronized
        if args.excel_file != self.excel_file:
            self.synapse_coordinates_path = args.excel_file
        elif args.synapse_coordinates_path != self.synapse_coordinates_path:
            self.excel_file = args.synapse_coordinates_path
        
        return self
    
    def get_feature_paths(self, segmentation_types=None, alphas=None, extraction_method=None, layer_num=None):
        """
        Get paths to feature CSV files based on segmentation types, alphas, and extraction method.
        
        Args:
            segmentation_types: List of segmentation types to include (defaults to [9, 10])
            alphas: List of alpha values to include (defaults to [1.0])
            extraction_method: Feature extraction method ("standard" or "stage_specific")
            layer_num: Layer number for stage-specific extraction method
            
        Returns:
            list: List of file paths to feature CSV files
        """
        if segmentation_types is None:
            segmentation_types = [9, 10]
        
        if alphas is None:
            alphas = [1.0]
            
        if extraction_method is None:
            extraction_method = self.extraction_method
            
        if layer_num is None:
            layer_num = self.layer_num
        
        paths = []
        for seg_type in segmentation_types:
            for alpha in alphas:
                alpha_str = str(alpha).replace('.', '_')
                
                if extraction_method == 'stage_specific':
                    filename = f'features_layer{layer_num}_seg{seg_type}_alpha{alpha_str}.csv'
                else:
                    filename = f'features_seg{seg_type}_alpha{alpha_str}.csv'
                    
                paths.append(os.path.join(self.csv_output_dir, filename))
        
        return paths

config = SynapseConfig() 