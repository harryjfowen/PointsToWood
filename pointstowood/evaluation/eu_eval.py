#!/usr/bin/env python3
"""
Compare P2W model predictions against existing FSCT and KPConv predictions
on collated evaluation files containing label, kpconv, and fsct columns.

Usage:
    python compare_collated.py --eval_data_dir /path/to/collated/files --models_dir ./model --output_dir ./comparison_results
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from src.io import load_file
from src.predicter import SemanticSegmentation
from sklearn.metrics import balanced_accuracy_score, f1_score, jaccard_score
import argparse
import warnings
import torch
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# -----------------------------------------------------------------------------
# Voxel-based label mixing analysis (from compare.py)
# -----------------------------------------------------------------------------

def compute_voxel_mixing(coords: np.ndarray, labels: np.ndarray, voxel_size: float = 0.20, verbose: bool = False):
    """Compute voxel-based label mixing values [1-2] where 1=pure, 2=max mixed."""
    if verbose:
        print(f"  Computing {voxel_size*100:.0f}cm voxel mixing...")
    
    voxel_keys = np.floor(coords / voxel_size).astype(np.int64)
    _, inverse_indices = np.unique(voxel_keys, axis=0, return_inverse=True)
    n_voxels = len(np.unique(voxel_keys, axis=0))
    
    wood_counts = np.bincount(inverse_indices[labels == 1], minlength=n_voxels)
    total_counts = np.bincount(inverse_indices, minlength=n_voxels)
    
    wood_props = np.divide(wood_counts, total_counts, out=np.zeros(n_voxels), where=total_counts>0)
    
    voxel_mixing = 1 + 2 * np.minimum(wood_props, 1 - wood_props)
    point_mixing = voxel_mixing[inverse_indices]
    
    pure, mixed, highly_mixed = np.sum(point_mixing == 1), np.sum(point_mixing > 1.1), np.sum(point_mixing > 1.8)
    if verbose:
        print(f"  {len(coords)} points, {n_voxels} voxels | Pure: {pure/len(coords)*100:.1f}%, Mixed: {mixed/len(coords)*100:.1f}%, Highly mixed: {highly_mixed/len(coords)*100:.1f}%")
    
    return point_mixing

def compute_mixed_pathlength_weights(pathlength: np.ndarray, mixing: np.ndarray):
    """Multiply path length by mixing factor."""
    return pathlength * mixing

# -----------------------------------------------------------------------------

class CollatedEvaluator:
    def __init__(self, eval_data_dir, models_dir, model_name='fbeta-eu.pth', model_type='optimised', add_voxel_mixing=True, max_probability=False, grid_size=[1.0, 2.0, 4.0], hysteresis=False, any_wood=0.5, min_pts=512, max_pts=16384, batch_size=0, resolution=0.0, grid_method='max', collect_grid_size=0.04):
        """
        Initialize evaluator for collated files
        
        Args:
            eval_data_dir: Directory containing collated PLY files with label, kpconv, fsct columns
            models_dir: Directory containing trained models
            model_name: Name of the P2W model to use for predictions
            model_type: Model type ('optimised', 'harmonic', 'reflectance')
            add_voxel_mixing: Whether to add voxel mixing analysis for complex weighting
            max_probability: Whether to use max probability prediction strategy
            grid_size: List of voxel grid sizes to use for preprocessing (default: [2.0, 4.0])
        """
        self.eval_data_dir = eval_data_dir
        self.models_dir = models_dir
        self.model_name = model_name
        self.model_type = model_type
        self.add_voxel_mixing = add_voxel_mixing
        self.max_probability = max_probability
        self.grid_size = grid_size
        self.results = []
        self.hysteresis = hysteresis
        self.any_wood = any_wood
        self.min_pts = min_pts
        self.max_pts = max_pts
        self.batch_size = batch_size
        self.resolution = resolution
        self.grid_method = grid_method
        self.collect_grid_size = collect_grid_size
        
    def get_collated_files(self):
        """Get all collated PLY files"""
        pattern = os.path.join(self.eval_data_dir, "*_collated.ply")
        files = glob.glob(pattern)
        print(f"Found {len(files)} collated files in {self.eval_data_dir}")
        return files
        
    def run_p2w_prediction(self, file_path):
        """
        Run P2W model prediction on a single file using SemanticSegmentation pipeline
        """
        model_path = os.path.join(self.models_dir, self.model_name)
        
        if not os.path.exists(model_path):
            print(f"Error: Model {model_path} not found")
            return None
        
        try:
            # Create temporary output directory
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            temp_output = f"temp_pred_p2w_{base_name}"
            
            # Create args object similar to biome_matrix_eval.py
            class Args:
                def __init__(self, model_name, model_type, max_probability, grid_size, hysteresis, any_wood, min_pts, max_pts, batch_size, knn_radius, resolution, grid_method, collect_grid_size):
                    self.point_cloud = [file_path]
                    self.file = file_path
                    self.odir = temp_output
                    self.model = model_name
                    self.wdir = os.path.dirname(os.path.dirname(model_path))
                    self.vxfile = os.path.join(temp_output, "voxels")
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    self.batch_size = batch_size
                    self.is_wood = 0.5
                    self.any_wood = any_wood
                    self.max_probability = max_probability
                    self.hysteresis = hysteresis
                    self.verbose = False
                    self.resolution = resolution
                    self.grid_size = grid_size
                    self.knn_radius = knn_radius
                    self.min_pts = min_pts
                    self.max_pts = max_pts
                    self.grid_method = grid_method
                    self.collect_grid_size = collect_grid_size
                    self.zero_reflectance = False
                    self.boost_perspective = False
                    self.denoise = False
                    self.denoise_k = 16
                    self.denoise_std = 1.0
                    self.model_type = model_type
                    self.mode = 'predict'
                    self.reflectance = False
                    self.num_procs = -1
                    
            args = Args(self.model_name, self.model_type, self.max_probability, self.grid_size, self.hysteresis, self.any_wood, self.min_pts, self.max_pts, self.batch_size, getattr(self, 'knn_radius', 0.025), self.resolution, self.grid_method, self.collect_grid_size)
            
            # Load and preprocess the point cloud data
            from src.io import load_file
            from predict import preprocess_point_cloud_data
            import glob as glob_module
            
            os.makedirs(args.vxfile, exist_ok=True)
            args.pc, args.headers = load_file(filename=file_path, additional_headers=True, verbose=False)
            args.pc, args.headers, args.reflectance = preprocess_point_cloud_data(args.pc, args.zero_reflectance)
            
            # Adjust model path based on reflectance
            def get_model_suffix(has_reflectance, model_type):
                if model_type == 'optimised':
                    return '' if has_reflectance else '-xyz'
                elif model_type == 'harmonic':
                    return '-harmonic'
                elif model_type == 'reflectance':
                    return ''
            
            base_model = args.model.replace('.pth', '')
            # Normalize: strip known suffixes to avoid double-appending
            for sfx in ['-xyz', '-harmonic']:
                if base_model.endswith(sfx):
                    base_model = base_model[: -len(sfx)]
            model_suffix = get_model_suffix(args.reflectance, args.model_type)
            if model_suffix:
                # Insert suffix before trailing region (e.g., fbeta-<sfx>-eu)
                if '-' in base_model:
                    prefix, region = base_model.rsplit('-', 1)
                    args.model = f"{prefix}{model_suffix}-{region}.pth"
                else:
                    args.model = f"{base_model}{model_suffix}.pth"
            else:
                args.model = f"{base_model}.pth"
            
            # Run voxelization preprocessing
            from src.preprocessing import preprocess
            print(f"Running preprocessing for {len(args.pc)} points...")
            print(f"Voxel directory: {args.vxfile}")
            preprocess(args)
            
            # Check voxel files written (.pt)
            pt_files = glob_module.glob(os.path.join(args.vxfile, "*.pt"))
            if len(pt_files) == 0:
                print(f"No voxel .pt files found in {args.vxfile}")
            else:
                sample_list = [os.path.basename(f) for f in pt_files[:5]]
                suffix = '...' if len(pt_files) > 5 else ''
                print(f"Found {len(pt_files)} voxel files (.pt) in {args.vxfile}; e.g., {sample_list}{suffix}")
            
            # The progress bars showed voxel creation, so files exist somewhere
            # The important thing is that preprocessing completed successfully
            
            # Run semantic segmentation
            print(f"Running SemanticSegmentation with model {args.model}...")
            print(f"Expected {len(args.pc)} output points...")
            result_args = SemanticSegmentation(args)
            
            # Verify the output size matches input
            if hasattr(result_args, 'pc'):
                print(f"SemanticSegmentation returned {len(result_args.pc)} points")
            
            # Extract predictions
            if hasattr(result_args, 'pc') and 'prediction' in result_args.pc.columns:
                predictions = result_args.pc['prediction'].values
                pred_stats = {
                    'total': len(predictions),
                    'wood': np.sum(predictions == 1),
                    'leaf': np.sum(predictions == 0),
                    'wood_pct': np.mean(predictions) * 100
                }
                print(f"P2W prediction complete: {pred_stats['total']} predictions "
                      f"({pred_stats['wood']} wood, {pred_stats['leaf']} leaf, {pred_stats['wood_pct']:.1f}% wood)")
                
                # Cleanup temp directory
                if os.path.exists(temp_output):
                    import shutil
                    shutil.rmtree(temp_output)
                
                return predictions
            else:
                print(f"Warning: No prediction column found in P2W result for {file_path}")
                return None
                
        except Exception as e:
            print(f"Error running P2W prediction on {file_path}: {e}")
            return None
    
    def compute_metrics(self, y_true, y_pred, weights=None, coords=None, method_name=""):
        """Compute evaluation metrics with optional voxel mixing analysis"""
        if y_true is None or y_pred is None:
            return {
                'balanced_accuracy': 0.0,
                'weighted_balanced_accuracy': 0.0,
                'mixed_weighted_balanced_accuracy': 0.0,
                'f1': 0.0,
                'iou': 0.0
            }
        
        # Ensure we have valid data points (remove NaN values)
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        valid_count = np.sum(valid_mask)
        total_count = len(y_true)
        
        if valid_count == 0:
            print(f"Warning: No valid data points for {method_name}")
            return {
                'balanced_accuracy': 0.0,
                'weighted_balanced_accuracy': 0.0,
                'mixed_weighted_balanced_accuracy': 0.0,
                'f1': 0.0,
                'iou': 0.0
            }
        
        if valid_count < total_count and getattr(self, 'verbose', False):
            print(f"Warning: {method_name} using {valid_count}/{total_count} valid points ({valid_count/total_count*100:.1f}%)")
        
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        weights_clean = weights[valid_mask] if weights is not None else None
        coords_clean = coords[valid_mask] if coords is not None else None
        
        # Use simple 1D arrays for metrics (cleaner approach)
        y_true_binary = y_true_clean.astype(int)
        y_pred_binary = y_pred_clean.astype(int)
        
        # Compute metrics
        balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)
        
        # Wood-only path length weighted balanced accuracy
        if weights_clean is not None:
            # Strategy: Only weight wood points by their path length, leaves get weight = 1
            # This emphasizes correct classification of structurally important wood points
            wood_weights = np.where(y_true_binary == 1, weights_clean, 1.0)
            weighted_balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary, sample_weight=wood_weights)
            
            wood_count = np.sum(y_true_binary == 1)
            if getattr(self, 'verbose', False):
                print(f"  Wood-weighted: using path length weights for {wood_count} wood points")
        else:
            weighted_balanced_acc = balanced_acc
        
        # Mixed voxel + path length weighted balanced accuracy (advanced weighting)
        if self.add_voxel_mixing and weights_clean is not None and coords_clean is not None:
            try:
                # Compute voxel mixing for this point cloud
                mixing_values = compute_voxel_mixing(coords_clean, y_true_binary, voxel_size=0.20, verbose=getattr(self, 'verbose', False))
                
                # Combine path length and mixing weights
                mixed_weights = compute_mixed_pathlength_weights(weights_clean, mixing_values)
                
                # Apply wood-only weighting strategy with mixed weights
                wood_mixed_weights = np.where(y_true_binary == 1, mixed_weights, 1.0)
                mixed_weighted_balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary, sample_weight=wood_mixed_weights)
                
                if getattr(self, 'verbose', False):
                    print(f"  Mixed-weighted: path length × voxel mixing for {wood_count} wood points")
            except Exception as e:
                print(f"  Warning: Voxel mixing failed for {method_name}: {e}")
                mixed_weighted_balanced_acc = weighted_balanced_acc
        else:
            mixed_weighted_balanced_acc = weighted_balanced_acc
        
        f1 = f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        iou = jaccard_score(y_true_binary, y_pred_binary, average='binary', zero_division=0)
        
        return {
            'balanced_accuracy': balanced_acc,
            'weighted_balanced_accuracy': weighted_balanced_acc,  # Wood-only path length weighted
            'mixed_weighted_balanced_accuracy': mixed_weighted_balanced_acc,  # Wood-only path length × mixing weighted
            'f1': f1,
            'iou': iou
        }
    
    def evaluate_file(self, file_path):
        """Evaluate all methods on a single collated file"""
        if getattr(self, 'verbose', False):
            print(f"\nEvaluating {os.path.basename(file_path)}")
        
        try:
            # Load collated file
            data_raw = load_file(file_path)
            
            # Convert to DataFrame if it's a numpy array
            if isinstance(data_raw, np.ndarray):
                print(f"Converting numpy array to DataFrame")
                # Try to load with additional headers to get column names
                try:
                    data, headers = load_file(file_path, additional_headers=True)
                    if isinstance(data, pd.DataFrame):
                        pass  # Already a DataFrame
                    else:
                        # Still an array, create DataFrame with default column names
                        if len(headers) == data.shape[1]:
                            data = pd.DataFrame(data, columns=headers)
                        else:
                            # Fallback to generic column names
                            n_cols = data.shape[1]
                            columns = ['x', 'y', 'z'] + [f'col_{i}' for i in range(3, n_cols)]
                            data = pd.DataFrame(data, columns=columns)
                except:
                    # If that fails, assume it's x,y,z + additional columns
                    n_cols = data_raw.shape[1]
                    columns = ['x', 'y', 'z'] + [f'col_{i}' for i in range(3, n_cols)]
                    data = pd.DataFrame(data_raw, columns=columns)
            else:
                data = data_raw
            
            if getattr(self, 'verbose', False):
                print(f"Loaded file with {len(data)} points")
                print(f"Available columns: {list(data.columns)}")
            
            # Check required columns
            required_cols = ['label']  # Ground truth
            prediction_cols = []
            
            if 'kpconv' in data.columns:
                prediction_cols.append('kpconv')
            if 'fsct' in data.columns:
                prediction_cols.append('fsct')
                
            if 'label' not in data.columns:
                print(f"Error: No 'label' column found in {file_path}")
                return None
                
            if len(prediction_cols) == 0:
                print(f"Warning: No prediction columns (kpconv, fsct) found in {file_path}")
            
            # Apply FSCT-specific cleaning only (like compare.py does)
            def clean_fsct_predictions(df):
                """Apply cleaning logic to FSCT data specifically"""
                if 'fsct' in df.columns:
                    # Check if FSCT has classes beyond 0/1
                    unique_vals = df['fsct'].dropna().unique()
                    if getattr(self, 'verbose', False):
                        print(f"FSCT unique values before cleaning: {unique_vals}")
                    
                    if len(unique_vals) > 2 or 2 in unique_vals or 3 in unique_vals:
                        if getattr(self, 'verbose', False):
                            print("Applying FSCT data cleaning...")
                        # Remove class 2 predictions
                        df_clean = df[df['fsct'] != 2].copy()
                        
                        # Convert class 3 to wood (class 1) 
                        if df_clean['fsct'].nunique() > 2:
                            df_clean.loc[:, 'fsct'] = (df_clean['fsct'] == 3).astype(int)
                        
                        if getattr(self, 'verbose', False):
                            print(f"FSCT unique values after cleaning: {df_clean['fsct'].dropna().unique()}")
                        return df_clean
                return df
            
            # Clean FSCT data only
            if 'fsct' in prediction_cols:
                data = clean_fsct_predictions(data)
            
            # Ensure pathlength column exists (like compare.py)
            if 'pathlength' not in data.columns:
                data['pathlength'] = 1
                if getattr(self, 'verbose', False):
                    print("Added default pathlength = 1")
                
            # Get ground truth, weights, and coordinates
            if isinstance(data['label'], pd.Series):
                y_true = data['label'].values
            else:
                y_true = np.array(data['label'])
                
            # Handle weights  
            if isinstance(data['pathlength'], pd.Series):
                weights = data['pathlength'].values
            else:
                weights = np.array(data['pathlength'])
                
            # Get coordinates for voxel mixing analysis
            coords = data[['x', 'y', 'z']].values
            
            if getattr(self, 'verbose', False):
                print(f"Ground truth range: {y_true.min()} to {y_true.max()}")
                print(f"Ground truth distribution: {np.bincount(y_true.astype(int))}")
            
            # Run P2W prediction
            if getattr(self, 'verbose', False):
                print("Running P2W prediction...")
            p2w_pred = self.run_p2w_prediction(file_path)
            
            # Collect all results
            results = {}
            base_name = os.path.basename(file_path).replace('_collated.ply', '')
            
            # Evaluate existing predictions
            for pred_col in prediction_cols:
                if pred_col in data.columns:
                    # Handle both pandas Series and numpy arrays
                    if isinstance(data[pred_col], pd.Series):
                        y_pred = data[pred_col].values
                    else:
                        y_pred = np.array(data[pred_col])
                    
                    # Debug NaN values
                    nan_count = np.sum(np.isnan(y_pred))
                    if getattr(self, 'verbose', False):
                        if nan_count > 0:
                            print(f"{pred_col.upper()} has {nan_count}/{len(y_pred)} NaN values ({nan_count/len(y_pred)*100:.1f}%)")
                            # Show non-NaN range
                            valid_pred = y_pred[~np.isnan(y_pred)]
                            if len(valid_pred) > 0:
                                print(f"{pred_col.upper()} valid prediction range: {valid_pred.min()} to {valid_pred.max()}")
                            else:
                                print(f"{pred_col.upper()} has no valid predictions!")
                        else:
                            print(f"{pred_col.upper()} prediction range: {y_pred.min()} to {y_pred.max()}")
                    
                    # Also show unique values for debugging
                    if getattr(self, 'verbose', False):
                        unique_vals = np.unique(y_pred[~np.isnan(y_pred)])
                        print(f"{pred_col.upper()} unique values: {unique_vals[:10]}{'...' if len(unique_vals) > 10 else ''}")
                    
                    metrics = self.compute_metrics(y_true, y_pred, weights, coords, pred_col.upper())
                    results[pred_col] = metrics
                    
                    if getattr(self, 'verbose', False):
                        print(f"{pred_col.upper()} - Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
                              f"Weighted Acc: {metrics['weighted_balanced_accuracy']:.4f}, "
                              f"Mixed Weighted Acc: {metrics['mixed_weighted_balanced_accuracy']:.4f}, "
                              f"F1: {metrics['f1']:.4f}, IoU: {metrics['iou']:.4f}")
            
            # Evaluate P2W prediction
            if p2w_pred is not None and len(p2w_pred) == len(y_true):
                if getattr(self, 'verbose', False):
                    print(f"P2W prediction range: {p2w_pred.min()} to {p2w_pred.max()}")
                
                metrics = self.compute_metrics(y_true, p2w_pred, weights, coords, "P2W")
                results['p2w'] = metrics
                
                if getattr(self, 'verbose', False):
                    print(f"P2W - Balanced Acc: {metrics['balanced_accuracy']:.4f}, "
                          f"Weighted Acc: {metrics['weighted_balanced_accuracy']:.4f}, "
                          f"Mixed Weighted Acc: {metrics['mixed_weighted_balanced_accuracy']:.4f}, "
                          f"F1: {metrics['f1']:.4f}, IoU: {metrics['iou']:.4f}")
            else:
                print("P2W prediction failed or size mismatch")
                results['p2w'] = self.compute_metrics(None, None, None, None)
            
            # Concise per-file summary lines
            params_line = (
                f"[Params] file={base_name} model={self.model_name} type={self.model_type} "
                f"grid={self.grid_size} grid_method={self.grid_method} collect_grid={self.collect_grid_size} "
                f"pts=[{self.min_pts},{self.max_pts}] hysteresis={'on' if self.hysteresis else 'off'} "
                f"any_wood={self.any_wood} max_prob={self.max_probability} resolution={self.resolution}"
            )
            print(params_line)

            def fmt_ba(d):
                try:
                    return f"{d['balanced_accuracy']:.4f}"
                except Exception:
                    return "N/A"

            ba_p2w = fmt_ba(results.get('p2w', {}))
            ba_kpc = fmt_ba(results.get('kpconv', {}))
            ba_fsct = fmt_ba(results.get('fsct', {}))

            # Highlight the winning score in bright green
            scores = {'P2W': ba_p2w, 'KPConv': ba_kpc, 'FSCT': ba_fsct}
            valid_scores = {k: float(v) for k, v in scores.items() if v != "N/A"}

            if valid_scores:
                winner = max(valid_scores, key=valid_scores.get)
                # ANSI escape codes: bright green for P2W wins, red for others
                green = '\033[92m'
                red = '\033[91m'
                reset = '\033[0m'

                # Apply highlighting: green if P2W wins, red if others win
                if winner == 'P2W':
                    ba_p2w = f"{green}{ba_p2w}{reset}"
                elif winner == 'KPConv':
                    ba_kpc = f"{red}{ba_kpc}{reset}"
                elif winner == 'FSCT':
                    ba_fsct = f"{red}{ba_fsct}{reset}"

            print(f"[BA] P2W={ba_p2w} | KPConv={ba_kpc} | FSCT={ba_fsct}")

            # Add file info to results
            file_results = {'file': base_name}
            for method, metrics in results.items():
                for metric, value in metrics.items():
                    file_results[f"{method}_{metric}"] = value
            
            return file_results
            
        except Exception as e:
            print(f"Error evaluating {file_path}: {e}")
            return None
    
    def run_full_evaluation(self):
        """Run evaluation on all collated files"""
        print("Running comparison evaluation on collated files...")
        
        files = self.get_collated_files()
        if not files:
            print("No collated files found!")
            return None
        
        # Process all files
        for file_path in tqdm(files, desc="Processing files"):
            result = self.evaluate_file(file_path)
            if result:
                self.results.append(result)
        
        # Convert to DataFrame
        if self.results:
            results_df = pd.DataFrame(self.results)
            print(f"\nProcessed {len(results_df)} files successfully")
            return results_df
        else:
            print("No successful evaluations")
            return None
    
    def add_region_info(self, results_df):
        """Add region information based on filename prefixes"""
        if results_df is None or results_df.empty:
            return results_df
        
        def classify_region(filename):
            """Classify filename into regions based on prefixes"""
            filename = filename.lower()
            
            # Region mapping based on prefixes
            if filename.startswith('pol'):
                return 'Poland'
            elif filename.startswith('spa'):
                return 'Spain'
            elif filename.startswith('fin'):
                return 'Finland'
            elif filename.startswith('ger'):
                return 'Germany'
            elif filename.startswith('china'):
                return 'China'
            elif filename.startswith('uk'):
                return 'UK'
            elif filename.startswith('lei'):
                return 'PhaseShift'  # Leicester PhaseShift samples
            elif filename.startswith('cameroon'):
                return 'Cameroon'
            elif filename.startswith('wood'):
                return 'UK LeafOff (Wood)'  # Pure wood samples
            else:
                return 'Other'
        
        # Add region column
        results_df['region'] = results_df['file'].apply(classify_region)
        return results_df
    
    def create_summary_table(self, results_df):
        """Create summary table with average metrics grouped by region"""
        if results_df is None or results_df.empty:
            return None, None
        
        # Add region information
        results_df = self.add_region_info(results_df)
        
        # Overall summary (all regions combined) - ordered by method priority
        overall_summary = {}
        overall_summary['Method'] = []
        overall_summary['Balanced Accuracy'] = []
        overall_summary['Weighted Balanced Accuracy'] = []
        overall_summary['Mixed Weighted Balanced Accuracy'] = []
        overall_summary['F1 Score'] = []
        overall_summary['IoU'] = []
        
        # Order: P2W first, then KPConv, then FSCT
        method_order = ['p2w', 'kpconv', 'fsct'] 
        display_names = ['P2W (Ours)', 'KPConv', 'FSCT']
        
        for method, display_name in zip(method_order, display_names):
            ba_col = f"{method}_balanced_accuracy"
            wba_col = f"{method}_weighted_balanced_accuracy"
            mwba_col = f"{method}_mixed_weighted_balanced_accuracy"
            f1_col = f"{method}_f1"
            iou_col = f"{method}_iou"
            
            if all(col in results_df.columns for col in [ba_col, wba_col, mwba_col, f1_col, iou_col]):
                overall_summary['Method'].append(display_name)
                overall_summary['Balanced Accuracy'].append(results_df[ba_col].mean())
                overall_summary['Weighted Balanced Accuracy'].append(results_df[wba_col].mean())
                overall_summary['Mixed Weighted Balanced Accuracy'].append(results_df[mwba_col].mean())
                overall_summary['F1 Score'].append(results_df[f1_col].mean())
                overall_summary['IoU'].append(results_df[iou_col].mean())
        
        overall_df = pd.DataFrame(overall_summary).round(4)
        
        # Regional summary with ordered columns
        regions = sorted(results_df['region'].unique())
        regional_summary = {}
        regional_summary['Region'] = []
        
        # Order: P2W first, then KPConv, then FSCT - for each metric
        method_order = ['p2w', 'kpconv', 'fsct'] 
        display_names = ['P2W (Ours)', 'KPConv', 'FSCT']
        
        # Add columns in metric-first order: BA P2W, BA KPConv, BA FSCT, then WBA P2W, etc.
        metrics = [('BA', 'Balanced Accuracy'), ('WBA', 'Wood-Weighted BA'), 
                  ('MWBA', 'Mixed-Weighted BA'), ('F1', 'F1 Score'), ('IoU', 'IoU')]
        
        for metric_short, metric_long in metrics:
            for method, display_name in zip(method_order, display_names):
                regional_summary[f'{metric_short} {display_name}'] = []
        
        # Calculate regional averages with ordered columns
        for region in regions:
            region_data = results_df[results_df['region'] == region]
            if len(region_data) == 0:
                continue
                
            regional_summary['Region'].append(f"{region} (n={len(region_data)})")
            
            # Fill in data in metric-first order
            for metric_short, metric_long in metrics:
                for method, display_name in zip(method_order, display_names):
                    col_mapping = {
                        'BA': f"{method}_balanced_accuracy",
                        'WBA': f"{method}_weighted_balanced_accuracy",
                        'MWBA': f"{method}_mixed_weighted_balanced_accuracy",
                        'F1': f"{method}_f1",
                        'IoU': f"{method}_iou"
                    }
                    
                    data_col = col_mapping[metric_short]
                    column_name = f'{metric_short} {display_name}'
                    
                    if data_col in region_data.columns:
                        regional_summary[column_name].append(region_data[data_col].mean())
                    else:
                        regional_summary[column_name].append(0.0)
        
        regional_df = pd.DataFrame(regional_summary).round(4)
        
        return overall_df, regional_df
    
    def plot_comparison(self, overall_df, regional_df, output_dir):
        """Create comparison plots"""
        if overall_df is None or overall_df.empty:
            return
        
        # Overall comparison plot - updated to include mixed weighted metric
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        metrics = ['Balanced Accuracy', 'Weighted Balanced Accuracy', 'Mixed Weighted Balanced Accuracy', 'F1 Score', 'IoU']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(overall_df['Method'], overall_df[metric], 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            
            # Add value labels on bars
            for bar, value in zip(bars, overall_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{metric}')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3)
        
        # Hide the empty subplot
        axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Overall Model Performance Comparison', fontsize=16, y=1.02)
        
        overall_plot_path = os.path.join(output_dir, 'overall_comparison.png')
        plt.savefig(overall_plot_path, dpi=300, bbox_inches='tight')
        print(f"Overall comparison plot saved to {overall_plot_path}")
        plt.close()
        
        # Regional comparison plot (if we have regional data)
        if regional_df is not None and not regional_df.empty:
            # Create a heatmap-style plot for regional comparison
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.flatten()
            
            methods = ['P2W (Ours)', 'KPConv', 'FSCT']
            metric_suffixes = ['BA', 'WBA', 'MWBA', 'F1', 'IoU']
            metric_names = ['Balanced Accuracy', 'Weighted Balanced Accuracy', 'Mixed Weighted Balanced Accuracy', 'F1 Score', 'IoU']
            
            for i, (suffix, name) in enumerate(zip(metric_suffixes, metric_names)):
                ax = axes[i]
                
                # Prepare data for grouped bar chart
                regions = regional_df['Region'].values
                x = np.arange(len(regions))
                width = 0.25
                
                for j, method in enumerate(methods):
                    col_name = f'{suffix} {method}'
                    if col_name in regional_df.columns:
                        values = regional_df[col_name].values
                    else:
                        print(f"Warning: Column '{col_name}' not found in regional_df")
                        print(f"Available columns: {list(regional_df.columns)}")
                        continue
                    ax.bar(x + j*width, values, width, label=method, 
                          color=['#1f77b4', '#ff7f0e', '#2ca02c'][j])
                
                ax.set_title(f'{name} by Region')
                ax.set_xlabel('Region')
                ax.set_ylabel(name)
                ax.set_xticks(x + width)
                ax.set_xticklabels(regions, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1.0)
            
            # Hide the empty subplot
            axes[5].set_visible(False)
            
            plt.tight_layout()
            plt.suptitle('Regional Model Performance Comparison', fontsize=16, y=0.98)
            
            regional_plot_path = os.path.join(output_dir, 'regional_comparison.png')
            plt.savefig(regional_plot_path, dpi=300, bbox_inches='tight')
            print(f"Regional comparison plot saved to {regional_plot_path}")
            plt.close()
    
    def save_results(self, results_df, overall_df, regional_df, output_dir):
        """Save results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        if results_df is not None:
            detailed_path = os.path.join(output_dir, "detailed_results.csv")
            results_df.to_csv(detailed_path, index=False)
            print(f"Detailed results saved to {detailed_path}")
        
        if overall_df is not None:
            overall_path = os.path.join(output_dir, "overall_summary.csv")
            overall_df.to_csv(overall_path, index=False)
            print(f"Overall summary saved to {overall_path}")
            
        if regional_df is not None:
            regional_path = os.path.join(output_dir, "regional_summary.csv")
            regional_df.to_csv(regional_path, index=False)
            print(f"Regional summary saved to {regional_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare P2W model against existing predictions')
    parser.add_argument('--eval_data_dir', required=True,
                       help='Directory with collated PLY files (containing label, kpconv, fsct columns)')
    parser.add_argument('--models_dir', default='./model',
                       help='Directory with trained models')
    parser.add_argument('--model_name', default='fbeta-eu.pth',
                       help='P2W model name to use for predictions')
    parser.add_argument('--model_type', default='optimised', 
                       choices=['optimised', 'harmonic', 'reflectance'],
                       help='Model type to use for predictions')
    parser.add_argument('--output_dir', default='./comparison_results',
                       help='Output directory for results')
    parser.add_argument('--add_voxel_mixing', action='store_true',
                       help='Add voxel mixing analysis for complex path length × mixing weighting')
    parser.add_argument('--max_probability', action='store_true',
                       help='Use max probability prediction strategy for potentially better mixed region performance')
    parser.add_argument('--grid_size', nargs='+', type=float, default=[1.0, 2.0, 4.0],
                       help='Voxel grid sizes to use for preprocessing (e.g., --grid_size 1.0 2.0 4.0)')
    parser.add_argument('--resolution', type=float, default=0.0,
                       help='Override point down-sample spacing [m]. 0 → adaptive per grid (1m→0.01, 2m→0.02, 4m→0.04).')
    parser.add_argument('--min_pts', type=int, default=512,
                       help='Minimum number of points per voxel')
    parser.add_argument('--max_pts', type=int, default=16384,
                       help='Maximum number of points per voxel')
    parser.add_argument('--batch_size', type=int, default=0,
                       help='Mini-batch size for model inference (0=adaptive GPU memory-aware batching)')
    parser.add_argument('--hysteresis', action='store_true',
                       help='Enable hysteresis post-processing for P2W predictions (same behavior as predict.py)')
    parser.add_argument('--any_wood', type=float, default=0.5,
                       help='ANY-wood threshold for P2W classification (passed into predicter pipeline)')
    parser.add_argument('--knn_radius', type=float, default=0.025,
                       help='Max neighbor distance (m) for KNN mapping; 0 disables radius gating')
    parser.add_argument('--grid_method', type=str, default='max', choices=['mean', 'max'],
                       help='Voxel representative method used in preprocessing (mean or max reflectance)')
    parser.add_argument('--collect_grid_size', type=float, default=0.04,
                       help='Optional post-aggregation voxel size used to assign labels per voxel (replaces KNN aggregation if > 0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose per-step logging; otherwise print concise per-file summary only')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CollatedEvaluator(args.eval_data_dir, args.models_dir, args.model_name, args.model_type, args.add_voxel_mixing, args.max_probability, args.grid_size, args.hysteresis, args.any_wood, args.min_pts, args.max_pts, args.batch_size, args.resolution, args.grid_method, args.collect_grid_size)
    evaluator.verbose = args.verbose
    
    # Run evaluation
    results_df = evaluator.run_full_evaluation()
    
    if results_df is not None:
        # Create summaries
        overall_df, regional_df = evaluator.create_summary_table(results_df)
        
        # Print results
        print("\n" + "="*60)
        print("COMPARISON EVALUATION RESULTS")
        print("="*60)
        
        if overall_df is not None:
            print("\nOverall Summary Results:")
            print(overall_df.to_string(index=False))
        
        if regional_df is not None:
            print(f"\nRegional Summary Results:")
            print(regional_df.to_string(index=False))
            
            # Show sample counts per region (add region info if not already there)
            results_with_regions = evaluator.add_region_info(results_df)
            region_counts = results_with_regions.groupby('region').size().sort_values(ascending=False)
            print(f"\nSample counts by region:")
            for region, count in region_counts.items():
                print(f"  {region}: {count} files")
        
        # Save results
        evaluator.save_results(results_df, overall_df, regional_df, args.output_dir)
        
        # Create and save plots
        evaluator.plot_comparison(overall_df, regional_df, args.output_dir)
    else:
        print("No results to display")

if __name__ == "__main__":
    main()