#!/usr/bin/env python3
"""
Biome Transfer Learning Evaluation Matrix
Creates 4x3 performance matrix: (Poland/Spain/Finland/EU models) × (Poland/Spain/Finland test sets)
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
from sklearn.metrics import balanced_accuracy_score
import argparse
import warnings
import torch
from sklearn.exceptions import UndefinedMetricWarning
from matplotlib.colors import Normalize
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

class BiomeMatrixEvaluator:
    def __init__(self, eval_data_dir, models_dir):
        """
        Initialize evaluator
        
        Args:
            eval_data_dir: Directory containing PLY files with prefixes (pol*, spa*, fin*)
            models_dir: Directory containing trained models
        """
        self.eval_data_dir = eval_data_dir
        self.models_dir = models_dir
        
        # Define biomes and their prefixes
        self.biomes = {
            'poland': 'pol',
            'spain': 'spa', 
            'finland': 'fin'
        }
        
        # Model mapping using fbeta models
        self.models = {
            'poland': 'fbeta-poland.pth',
            'spain': 'fbeta-spain.pth', 
            'finland': 'fbeta-finland.pth',
            'eu': 'fbeta-eu.pth'
        }
        
        self.results_matrix = {}
        
    def get_test_files_by_biome(self, biome):
        """Get all test files for a specific biome"""
        prefix = self.biomes[biome]
        pattern = os.path.join(self.eval_data_dir, f"{prefix}*.ply")
        files = glob.glob(pattern)
        # Ignore existing P2W prediction files in the directory
        files = [f for f in files if not os.path.basename(f).lower().endswith('_p2w.ply')]
        print(f"Looking for {biome} files with pattern: {pattern}")
        print(f"Found {len(files)} files: {files}")
        
        
        return files
        
    def load_predictions(self, model_name, test_files):
        """
        Run model predictions for test files using your SemanticSegmentation pipeline
        """
        import gc
        import torch

        model_path = os.path.join(self.models_dir, self.models[model_name])

        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found, returning zeros")
            return {file_path: None for file_path in test_files}
        
        predictions = {}
        
        print(f"Running {model_name} model on {len(test_files)} files...")
        for file_path in tqdm(test_files, desc=f"Predicting with {model_name}"):
            print(f"Processing file: {file_path}")
            if not os.path.exists(file_path):
                print(f"ERROR: File does not exist: {file_path}")
                predictions[file_path] = None
                continue
            try:
                # Create temporary output directory
                base_name = os.path.splitext(os.path.basename(file_path))[0]  # Remove .ply extension
                temp_output = f"temp_pred_{model_name}_{base_name}"
                
                # Create args object similar to your predict.py
                class Args:
                    def __init__(self):
                        self.point_cloud = [file_path]
                        self.file = file_path
                        self.odir = temp_output
                        self.model = os.path.basename(model_path)  # Just the filename
                        self.wdir = os.path.dirname(os.path.dirname(model_path))  # Parent of model directory
                        self.vxfile = os.path.join(temp_output, "voxels")
                        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        self.batch_size = 0  # Enable adaptive batching
                        self.is_wood = 0.5
                        self.any_wood = 0.5
                        self.max_probability = False
                        self.verbose = False
                        self.resolution = 0.00
                        self.grid_size = [1.0, 2.0, 4.0]
                        self.overlap = 0.0
                        self.min_pts = 512
                        self.max_pts = 16384
                        self.zero_reflectance = False
                        self.boost_perspective = False
                        self.denoise = False
                        self.denoise_k = 16
                        self.denoise_std = 1.0
                        self.model_type = 'optimised'
                        self.mode = 'predict'
                        self.reflectance = False
                        self.num_procs = -1
                        self.grid_method = 'max'
                        self.collect_grid_size = 0.05
                        self.memory_fraction = 0.7
                        
                args = Args()
                
                # Load and preprocess the point cloud data (like predict.py does)
                from src.io import load_file
                from predict import preprocess_point_cloud_data
                
                os.makedirs(args.vxfile, exist_ok=True)
                args.pc, args.headers = load_file(filename=file_path, additional_headers=True, verbose=False)
                args.pc, args.headers, args.reflectance = preprocess_point_cloud_data(args.pc, args.zero_reflectance)
                
                # Adjust model path based on reflectance (like predict.py does)
                def get_model_suffix(has_reflectance, model_type):
                    if model_type == 'optimised':
                        return '' if has_reflectance else '-xyz'
                    elif model_type == 'harmonic':
                        return '-harmonic'
                    elif model_type == 'reflectance':
                        return ''
                
                base_model = args.model.replace('.pth', '')
                model_suffix = get_model_suffix(args.reflectance, args.model_type)
                args.model = f"{base_model}{model_suffix}.pth"
                
                # Run voxelization preprocessing (this is what was missing!)
                from src.preprocessing import preprocess
                preprocess(args)
                
                # Run semantic segmentation
                try:
                    result_args = SemanticSegmentation(args)
                    
                    # Extract predictions directly from the result (no need for file I/O)
                    if hasattr(result_args, 'pc'):
                        print(f"Available columns in result: {list(result_args.pc.columns)}")
                        if 'prediction' in result_args.pc.columns:
                            predictions[file_path] = result_args.pc['prediction'].values
                            print(f"Successfully extracted {len(predictions[file_path])} predictions from {os.path.basename(file_path)}")
                            print(f"Prediction range: {predictions[file_path].min()} to {predictions[file_path].max()}")
                        else:
                            predictions[file_path] = None
                            print(f"Warning: No prediction column found in result for {file_path}")
                    else:
                        predictions[file_path] = None
                        print(f"Warning: No 'pc' attribute found in result for {file_path}")
                        
                except Exception as inner_e:
                    # If preprocessing results in empty dataset, skip this file
                    if "empty" in str(inner_e).lower() or "min()" in str(inner_e):
                        print(f"Warning: Skipping {file_path} - preprocessing resulted in empty dataset")
                        predictions[file_path] = None
                        continue
                    else:
                        raise inner_e
                
                # Cleanup temp directory
                if os.path.exists(temp_output):
                    import shutil
                    shutil.rmtree(temp_output)

                # Force cleanup to prevent file handle buildup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                print(f"Error predicting {file_path} with {model_name}: {e}")
                predictions[file_path] = None

                # Cleanup even on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
        return predictions
        
    def load_ground_truth(self, file_path):
        """Load ground truth labels from PLY file"""
        try:
            data = load_file(file_path)  # Using your existing PLY loader
            
            # Look for both 'truth' and 'label' columns as requested
            if 'truth' in data.columns:
                labels = data['truth'].values
                print(f"Using 'truth' column from {os.path.basename(file_path)}")
            elif 'label' in data.columns:
                labels = data['label'].values
                print(f"Using 'label' column from {os.path.basename(file_path)}")
            else:
                # Try to infer label column
                label_cols = [col for col in data.columns if 'truth' in col.lower() or 'label' in col.lower() or 'class' in col.lower()]
                if label_cols:
                    labels = data[label_cols[0]].values
                    print(f"Using '{label_cols[0]}' column from {os.path.basename(file_path)}")
                else:
                    print(f"Available columns in {os.path.basename(file_path)}: {list(data.columns)}")
                    raise ValueError(f"No truth/label column found in {file_path}")
                    
            return labels
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
            
    def compute_metrics(self, y_true, y_pred):
        """Compute balanced accuracy metric only"""
        if y_true is None or y_pred is None:
            return {'balanced_accuracy': 0.0}
            
        # Convert to binary if needed (assuming 0=leaf, 1=wood)
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        
        metrics = {
            'balanced_accuracy': balanced_accuracy_score(y_true_binary, y_pred_binary)
        }
        
        return metrics
        
    def evaluate_model_on_biome(self, model_name, test_biome):
        """Evaluate a specific model on a specific biome's test set"""
        print(f"Evaluating {model_name} model on {test_biome} test set...")
        
        test_files = self.get_test_files_by_biome(test_biome)
        if not test_files:
            print(f"No test files found for {test_biome}")
            return {'accuracy': 0.0, 'f1': 0.0, 'iou': 0.0}
            
        predictions = self.load_predictions(model_name, test_files)
        
        all_metrics = []
        for file_path in tqdm(test_files, desc=f"{model_name} on {test_biome}"):
            # Load ground truth
            y_true = self.load_ground_truth(file_path)
            y_pred = predictions.get(file_path)
            
            # Compute metrics for this file
            metrics = self.compute_metrics(y_true, y_pred)
            all_metrics.append(metrics)
            
        # Average metrics across all files
        if all_metrics:
            avg_metrics = {
                'balanced_accuracy': np.mean([m['balanced_accuracy'] for m in all_metrics])
            }
        else:
            avg_metrics = {'balanced_accuracy': 0.0}
            
        return avg_metrics
        
    def run_full_evaluation(self):
        """Run complete 4x3 evaluation matrix"""
        print("Running biome transfer learning evaluation...")
        
        model_names = ['poland', 'spain', 'finland', 'eu']
        test_biomes = ['poland', 'spain', 'finland']
        
        # Initialize results matrix for balanced accuracy only
        self.results_matrix['balanced_accuracy'] = pd.DataFrame(
            index=model_names,
            columns=test_biomes,
            dtype=float
        )
            
        # Run all combinations
        for model_name in model_names:
            for test_biome in test_biomes:
                metrics = self.evaluate_model_on_biome(model_name, test_biome)
                
                # Store results
                self.results_matrix['balanced_accuracy'].loc[model_name, test_biome] = metrics['balanced_accuracy']
                    
        return self.results_matrix
        
    def plot_heatmap(self, metric='balanced_accuracy', save_path=None):
        """Create heatmap visualization of results"""
        if metric not in self.results_matrix:
            print(f"Metric {metric} not found in results")
            return
            
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        matrix = self.results_matrix[metric]
        ax = sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            cbar_kws={'label': ''},
            square=True
        )
        # Remove colorbar label (legend title)
        cbar = None
        try:
            # seaborn attaches colorbar to the first QuadMesh in collections
            if ax.collections and hasattr(ax.collections[0], 'colorbar'):
                cbar = ax.collections[0].colorbar
        except Exception:
            cbar = None
        if cbar is not None:
            cbar.set_label('')
            try:
                cbar.ax.set_ylabel('')
                cbar.ax.set_title('')
            except Exception:
                pass
        # Single bold title on axes
        ax.set_title('Balanced Accuracy', fontweight='bold')
        plt.xlabel('Test Dataset')
        plt.ylabel('Model')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()

    def plot_viridis_table(self, metric='balanced_accuracy', save_path=None):
        """Create a publication-ready viridis-colored table highlighting strengths/weaknesses and best performers."""
        if metric not in self.results_matrix:
            print(f"Metric {metric} not found in results")
            return

        matrix = self.results_matrix[metric]
        values = matrix.values.astype(float)
        # Handle NaNs gracefully by treating them as min
        finite_vals = values[np.isfinite(values)]
        if finite_vals.size == 0:
            print("No finite values to plot")
            return
        vmin, vmax = finite_vals.min(), finite_vals.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.viridis

        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        im = ax.imshow(values, cmap=cmap, norm=norm)

        # Set ticks and labels
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_xticklabels(matrix.columns, fontsize=10)
        ax.set_yticklabels(matrix.index, fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        # Remove legend title on viridis table
        try:
            cbar.set_label('')
            cbar.ax.set_ylabel('')
            cbar.ax.set_title('')
        except Exception:
            pass

        # Determine per-column bests (model that performs best per test)
        col_max = np.nanargmax(values, axis=0)
        row_names = list(matrix.index)
        col_names = list(matrix.columns)

        # Overlay text with contrast-aware coloring
        mid = (vmin + vmax) / 2.0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                disp = 'NA' if not np.isfinite(val) else f"{val:.3f}"

                is_eu_row = (row_names[i].lower() == 'eu')
                is_eu_col_best = is_eu_row and (i == col_max[j])
                is_biome_diagonal = (row_names[i] == col_names[j]) and (not is_eu_row)

                # Choose base text color: white if value < 0.86, else black
                base_color = 'white' if (np.isfinite(val) and val < 0.86) else 'black'

                if is_eu_col_best:
                    # Bold dark red for EU when it is best on a test dataset
                    txt = ax.text(j, i, disp, ha='center', va='center', color='#8B0000', fontsize=10, fontweight='bold')
                    txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])
                elif is_biome_diagonal:
                    # Bold black on biome diagonals for emphasis
                    ax.text(j, i, disp, ha='center', va='center', color='black', fontsize=10, fontweight='bold')
                else:
                    # Other cells: threshold-based color
                    ax.text(j, i, disp, ha='center', va='center', color=base_color, fontsize=9, fontweight='normal')

        # Gridlines to create a clean table look
        ax.set_xticks(np.arange(-.5, values.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-.5, values.shape[0], 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.0)
        ax.tick_params(which='minor', bottom=False, left=False)

        # Labels and single bold title
        ax.set_title('Balanced Accuracy', fontweight='bold')
        ax.set_xlabel('Test Dataset')
        ax.set_ylabel('Model')
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Viridis table saved to {save_path}")
        plt.close(fig)
        
    def save_results(self, output_dir):
        """Save results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for metric, matrix in self.results_matrix.items():
            output_path = os.path.join(output_dir, f"biome_matrix_{metric}.csv")
            matrix.to_csv(output_path)
            print(f"Saved {metric} results to {output_path}")
            
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "="*60)
        print("BIOME TRANSFER LEARNING EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nBALANCED ACCURACY Results:")
        print(self.results_matrix['balanced_accuracy'].round(3))
        
        # Highlight diagonal vs off-diagonal performance (if full matrix available)
        matrix = self.results_matrix['balanced_accuracy']
        
        # Check if we have the full matrix or just test mode
        available_models = matrix.index.tolist()
        available_tests = matrix.columns.tolist()
        
        if len(available_models) == 4 and len(available_tests) == 3:
            # Full evaluation mode
            diagonal_avg = np.mean([matrix.loc['poland', 'poland'], 
                                  matrix.loc['spain', 'spain'], 
                                  matrix.loc['finland', 'finland']])
            
            # Off-diagonal for biome models only (exclude EU)
            off_diagonal = []
            for model in ['poland', 'spain', 'finland']:
                for test in ['poland', 'spain', 'finland']:
                    if model != test:
                        off_diagonal.append(matrix.loc[model, test])
            off_diagonal_avg = np.mean(off_diagonal)
            
            # EU model average
            eu_avg = np.mean([matrix.loc['eu', test] for test in ['poland', 'spain', 'finland']])
            
            print(f"  Diagonal avg (specialized): {diagonal_avg:.3f}")
            print(f"  Off-diagonal avg (transfer): {off_diagonal_avg:.3f}")
            print(f"  EU model avg (generalist): {eu_avg:.3f}")
            print(f"  Specialization advantage: {diagonal_avg - off_diagonal_avg:.3f}")
            print(f"  EU vs Transfer gap: {eu_avg - off_diagonal_avg:.3f}")
        else:
            # Test mode or partial results
            print(f"  Test mode results - Models: {available_models}, Tests: {available_tests}")
            for model in available_models:
                for test in available_tests:
                    score = matrix.loc[model, test]
                    print(f"  {model} on {test}: {score:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate biome model transfer learning')
    parser.add_argument('--eval_data_dir', required=True, 
                       help='Directory with PLY files (pol*, spa*, fin* prefixes)')
    parser.add_argument('--models_dir', default='./model',
                       help='Directory with trained models')
    parser.add_argument('--output_dir', default='./biome_eval_results',
                       help='Output directory for results')
    parser.add_argument('--metric', default='balanced_accuracy', choices=['balanced_accuracy'],
                       help='Metric for heatmap visualization')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = BiomeMatrixEvaluator(args.eval_data_dir, args.models_dir)
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results(args.output_dir)
    
    # Create heatmap
    heatmap_path = os.path.join(args.output_dir, f'biome_matrix_{args.metric}.png')
    evaluator.plot_heatmap(metric=args.metric, save_path=heatmap_path)

    # Create publication-ready viridis PNG table
    viridis_path = os.path.join(args.output_dir, f'biome_matrix_viridis_{args.metric}.png')
    evaluator.plot_viridis_table(metric=args.metric, save_path=viridis_path)

if __name__ == "__main__":
    main()