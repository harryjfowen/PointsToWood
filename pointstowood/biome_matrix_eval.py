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
        return files
        
    def load_predictions(self, model_name, test_files):
        """
        Run model predictions for test files using your SemanticSegmentation pipeline
        """
        model_path = os.path.join(self.models_dir, self.models[model_name])
        
        if not os.path.exists(model_path):
            print(f"Warning: Model {model_path} not found, returning zeros")
            return {file_path: None for file_path in test_files}
        
        predictions = {}
        
        print(f"Running {model_name} model on {len(test_files)} files...")
        for file_path in tqdm(test_files, desc=f"Predicting with {model_name}"):
            try:
                # Create temporary output directory
                temp_output = f"temp_pred_{model_name}_{os.path.basename(file_path)}"
                
                # Create args object similar to your predict.py
                class Args:
                    def __init__(self):
                        self.file = file_path
                        self.odir = temp_output
                        self.model = model_path
                        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        self.batch_size = 4
                        self.is_wood = 0.5
                        self.any_wood = 1
                        self.max_probability = False
                        self.verbose = False
                        
                args = Args()
                
                # Run semantic segmentation
                result_args = SemanticSegmentation(args)
                
                # Load the predicted point cloud
                output_files = glob.glob(os.path.join(temp_output, "*.ply"))
                if output_files:
                    predicted_data = load_file(output_files[0])
                    if 'prediction' in predicted_data.columns:
                        predictions[file_path] = predicted_data['prediction'].values
                    else:
                        predictions[file_path] = None
                        print(f"Warning: No prediction column in {output_files[0]}")
                else:
                    predictions[file_path] = None
                    print(f"Warning: No output file generated for {file_path}")
                
                # Cleanup temp directory
                if os.path.exists(temp_output):
                    import shutil
                    shutil.rmtree(temp_output)
                    
            except Exception as e:
                print(f"Error predicting {file_path} with {model_name}: {e}")
                predictions[file_path] = None
                
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
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            cbar_kws={'label': f'{metric.upper()} Score'},
            square=True
        )
        
        plt.title(f'Biome Model Transfer Learning Performance\n{metric.upper()} Scores')
        plt.xlabel('Test Dataset')
        plt.ylabel('Model')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()
        
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
        
        # Highlight diagonal vs off-diagonal performance
        matrix = self.results_matrix['balanced_accuracy']
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

if __name__ == "__main__":
    main()