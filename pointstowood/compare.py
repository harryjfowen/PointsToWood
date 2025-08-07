import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.io import load_file, save_file
import sys
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from collections import defaultdict
import argparse
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# -----------------------------------------------------------------------------
# Voxel-based label mixing analysis
# -----------------------------------------------------------------------------

def compute_voxel_mixing(coords: np.ndarray, labels: np.ndarray, voxel_size: float = 0.10):
    """Compute voxel-based label mixing values [1-2] where 1=pure, 2=max mixed."""
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
    print(f"  {len(coords)} points, {n_voxels} voxels | Pure: {pure/len(coords)*100:.1f}%, Mixed: {mixed/len(coords)*100:.1f}%, Highly mixed: {highly_mixed/len(coords)*100:.1f}%")
    
    return point_mixing

def compute_mixed_pathlength_weights(pathlength: np.ndarray, mixing: np.ndarray):
    """Multiply path length by mixing factor."""
    return pathlength * mixing

# -----------------------------------------------------------------------------

class PointCloudDownsampler:
    def __init__(self, pc, vlength):
        self.pc = pc
        self.vlength = vlength
    
    def random_voxelisation(self):
        voxel_indices = np.floor(self.pc[:, :3] / self.vlength).astype(int)
        voxel_dict = defaultdict(list)
        for i, voxel_index in enumerate(voxel_indices):
            voxel_key = tuple(voxel_index)
            voxel_dict[voxel_key].append(i)  # Store the index instead of the point
        selected_indices = [voxel_points_indices[np.random.randint(len(voxel_points_indices))] for voxel_points_indices in voxel_dict.values()]
        return selected_indices

statistics_df = pd.DataFrame(columns=[
    'File', 
    'F1_fsct', 'IoU_fsct', 'Accuracy_fsct', 'Weighted_accuracy_fsct',
    'F1_ours', 'IoU_ours', 'Accuracy_ours', 'Weighted_accuracy_ours',
    'F1_kpconv', 'IoU_kpconv', 'Accuracy_kpconv', 'Weighted_accuracy_kpconv'
])

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Compare our model predictions against FSCT and KPConv baselines for a "
        "specific geographic region."
    )
)

parser.add_argument(
    "--region",
    required=False,
    default="",
    help="Optional region name used only for labeling output files (e.g. 'global', 'spain', 'eu'). If omitted, files are named 'results.csv' and 'summary.csv'.",
)

parser.add_argument(
    "--ours-dir",
    dest="ours_dir",
    required=True,
    help="Directory containing *_ours.ply prediction files for the chosen region.",
)

parser.add_argument(
    "--fsct-dir",
    dest="fsct_dir",
    required=True,
    help="Directory containing *_fsct.ply baseline files.",
)

parser.add_argument(
    "--kpconv-dir",
    dest="kpconv_dir",
    required=True,
    help="Directory containing *_kpconv.ply baseline files.",
)

parser.add_argument(
    "--add-edge-weight",
    dest="add_edge_weight",
    action="store_true",
    help="Add voxel-based label mixing analysis to enhance path length weighting. Points with high path length AND high mixing get extra emphasis.",
)

args = parser.parse_args()

# -----------------------------------------------------------------------------
# Gather our prediction files for this region
# -----------------------------------------------------------------------------

ours_dir = os.path.abspath(args.ours_dir)
fsct_dir = os.path.abspath(args.fsct_dir)
kpconv_dir = os.path.abspath(args.kpconv_dir)

if not all(os.path.isdir(p) for p in (ours_dir, fsct_dir, kpconv_dir)):
    raise ValueError("One or more supplied directories do not exist.")

our_files = glob.glob(os.path.join(ours_dir, '*_p2w.ply'))

if len(our_files) == 0:
    raise RuntimeError(f"No *_p2w.ply files found in {ours_dir}.")

# -----------------------------------------------------------------------------
# Output locations – save inside the region directory so results stay together
# -----------------------------------------------------------------------------

region_suffix = f"_{args.region}" if args.region else ""

output_csv   = os.path.join(ours_dir, f"results{region_suffix}.csv")
output_img   = os.path.join(ours_dir, f"results{region_suffix}.png")
summary_csv  = os.path.join(ours_dir, f"summary{region_suffix}.csv")
summary_img  = os.path.join(ours_dir, f"summary{region_suffix}.png")

# -----------------------------------------------------------------------------
# Main processing loop – iterate over *_ours.ply in the region directory
# -----------------------------------------------------------------------------

file_names = our_files

def clean_point_cloud(df):
    df = df.copy()
    df.rename(columns=lambda x: x.replace('scalar_', '') if 'scalar_' in x else x, inplace=True)
    
    # ------------------------------------------------------------------
    # Harmonise prediction column – ensure every DataFrame ends up with
    # a binary 0/1 column called 'prediction'. Source columns may be:
    #   • 'prediction'   (our files)
    #   • 'pred'        (our files - renamed from 'pred')
    #   • 'label'       (fsct)
    #   • 'preds'       (kpconv)
    # 
    # NOTE: 'truth' columns are kept as ground truth labels
    # ------------------------------------------------------------------

    if 'prediction' not in df.columns:
        if 'pred' in df.columns:
            df.rename(columns={'pred': 'prediction'}, inplace=True)
            print("Renamed 'pred' column to 'prediction'")
        elif 'preds' in df.columns:
            df.rename(columns={'preds': 'prediction'}, inplace=True)
            print("Renamed 'preds' column to 'prediction'")
        elif 'label' in df.columns:
            df.rename(columns={'label': 'prediction'}, inplace=True)
            print("Renamed 'label' column to 'prediction'")

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    if 'prediction' in df.columns:
        df = df[df['prediction'] != 2]
        if df['prediction'].nunique() > 2:
            df.loc[:, 'prediction'] = (df['prediction'] == 3).astype(int)
    else:
        print(f"Warning: No 'prediction' column found. Available columns: {list(df.columns)}")
    
    if 'pathlength' not in df.columns:
        df.loc[:, 'pathlength'] = 1
        
    return df

for file_name in tqdm(file_names, desc='Processing files', unit='file'):
    base_name = os.path.basename(file_name).replace('_p2w.ply', '')
    
    ours_file   = os.path.join(ours_dir,  base_name + '_p2w.ply')
    fsct_file   = os.path.join(fsct_dir,  base_name + '_fsct.ply')
    kpconv_file = os.path.join(kpconv_dir, base_name + '_kpconv.ply')
    
    if not (os.path.exists(ours_file) and os.path.exists(fsct_file) and os.path.exists(kpconv_file)):
        print(f"Skipping {base_name} - missing one or more model files")
        continue
    
    print(f"Processing {base_name}")
    
    ours = clean_point_cloud(load_file(ours_file))
    fsct = clean_point_cloud(load_file(fsct_file))
    kpconv = clean_point_cloud(load_file(kpconv_file))
    
    # ------------------------------------------------------------------
    # SPECIAL CASE: Pure-wood reference scans (file names containing
    # "wood") do not ship with ground-truth labels.  We assume every
    # point in such samples is wood (truth == 1) so that the evaluation
    # can proceed without skipping them.
    # ------------------------------------------------------------------
    if "wood" in base_name.lower():
        for df in [ours, fsct, kpconv]:
            if 'truth' not in df.columns:
                df['truth'] = 1 
    models = {"ours": ours, "fsct": fsct, "kpconv": kpconv}
    models_with_truth = {name: df for name, df in models.items() if 'truth' in df.columns}

    if not models_with_truth:
        print(f"Skipping {base_name} - no 'truth' column found in any of the three files")
        continue

    # Use the first available truth column as ground truth for all models
    truth_source_name = list(models_with_truth.keys())[0]
    truth_source_df = models_with_truth[truth_source_name]
    print(f"Using 'truth' column from {truth_source_name} as ground truth")

    # Copy truth to all models that don't have it
    for name, df in models.items():
        if 'truth' not in df.columns:
            if len(df) == len(truth_source_df):
                df['truth'] = truth_source_df['truth'].values
                print(f"Copied 'truth' column from {truth_source_name} to {name} (point count match)")
            else:
                print(f"Skipping {base_name} - cannot copy 'truth' to {name}; point counts differ")
                continue

    if not all('truth' in df.columns for df in models.values()):
        print(f"Skipping {base_name} - 'truth' column still missing after attempts")
        continue
    
    ours, fsct, kpconv = models['ours'], models['fsct'], models['kpconv']
    
    if args.add_edge_weight:
        print(f"Computing voxel mixing analysis for {base_name}...")
        models_data = [(ours, "Ours"), (fsct, "FSCT"), (kpconv, "KPConv")]
        weights = []
        
        for df, name in models_data:
            coords = df[['x', 'y', 'z']].values
            print(f"  {name}: {len(df)} points")
            mixing = compute_voxel_mixing(coords, df['truth'].values)
            df['mixing'] = mixing
            weights.append(compute_mixed_pathlength_weights(df['pathlength'].values, mixing))
        
        mixed_weights_ours, mixed_weights_fsct, mixed_weights_kpconv = weights
    else:
        mixed_weights_ours = ours['pathlength'].values
        mixed_weights_fsct = fsct['pathlength'].values  
        mixed_weights_kpconv = kpconv['pathlength'].values
        iou_fsct = jaccard_score(fsct[['truth']].astype(int), fsct[['prediction']].astype(int), average='binary', zero_division=0)
    f1_fsct = f1_score(fsct[['truth']].astype(int), fsct[['prediction']].astype(int), average='binary', zero_division=0)
    accuracy_fsct = balanced_accuracy_score(fsct[['truth']].astype(int), fsct[['prediction']].astype(int))
    weighted_accuracy_fsct = balanced_accuracy_score(fsct[['truth']].astype(int), fsct[['prediction']].astype(int), sample_weight=fsct['pathlength'])
    mixed_weighted_accuracy_fsct = balanced_accuracy_score(fsct[['truth']].astype(int), fsct[['prediction']].astype(int), sample_weight=mixed_weights_fsct)

    iou_ours = jaccard_score(ours[['truth']].astype(int), ours[['prediction']].astype(int), average='binary', zero_division=0)
    f1_ours = f1_score(ours[['truth']].astype(int), ours[['prediction']].astype(int), average='binary', zero_division=0)
    accuracy_ours = balanced_accuracy_score(ours[['truth']].astype(int), ours[['prediction']].astype(int))
    weighted_accuracy_ours = balanced_accuracy_score(ours[['truth']].astype(int), ours[['prediction']].astype(int), sample_weight=ours['pathlength'])
    mixed_weighted_accuracy_ours = balanced_accuracy_score(ours[['truth']].astype(int), ours[['prediction']].astype(int), sample_weight=mixed_weights_ours)

    iou_kpconv = jaccard_score(kpconv[['truth']].astype(int), kpconv[['prediction']].astype(int), average='binary', zero_division=0)
    f1_kpconv = f1_score(kpconv[['truth']].astype(int), kpconv[['prediction']].astype(int), average='binary', zero_division=0)
    accuracy_kpconv = balanced_accuracy_score(kpconv[['truth']].astype(int), kpconv[['prediction']].astype(int))
    weighted_accuracy_kpconv = balanced_accuracy_score(kpconv[['truth']].astype(int), kpconv[['prediction']].astype(int), sample_weight=kpconv['pathlength'])
    mixed_weighted_accuracy_kpconv = balanced_accuracy_score(kpconv[['truth']].astype(int), kpconv[['prediction']].astype(int), sample_weight=mixed_weights_kpconv)
    
    print(f'Accuracy - FSCT: {accuracy_fsct:.4f}, Ours: {accuracy_ours:.4f}, KPConv: {accuracy_kpconv:.4f}')
    print(f'Weighted Accuracy - FSCT: {weighted_accuracy_fsct:.4f}, Ours: {weighted_accuracy_ours:.4f}, KPConv: {weighted_accuracy_kpconv:.4f}')
    
    enhanced_label = "Mixed+Path Weighted" if args.add_edge_weight else "Enhanced Weighted"
    print(f'{enhanced_label} Accuracy - FSCT: {mixed_weighted_accuracy_fsct:.4f}, Ours: {mixed_weighted_accuracy_ours:.4f}, KPConv: {mixed_weighted_accuracy_kpconv:.4f}')
    print(f'IoU - FSCT: {iou_fsct:.4f}, Ours: {iou_ours:.4f}, KPConv: {iou_kpconv:.4f}')
    print(f'F1 - FSCT: {f1_fsct:.4f}, Ours: {f1_ours:.4f}, KPConv: {f1_kpconv:.4f}')

    enhanced_suffix = "mixed_weighted" if args.add_edge_weight else "enhanced_weighted"
    
    statistics = pd.DataFrame({
        'File': [base_name],
        'F1_fsct': [f1_fsct],
        'IoU_fsct': [iou_fsct],
        'Accuracy_fsct': [accuracy_fsct],
        'Accuracy_weighted_fsct': [weighted_accuracy_fsct],
        f'Accuracy_{enhanced_suffix}_fsct': [mixed_weighted_accuracy_fsct],
        'F1_ours': [f1_ours],
        'IoU_ours': [iou_ours],
        'Accuracy_ours': [accuracy_ours],
        'Accuracy_weighted_ours': [weighted_accuracy_ours],
        f'Accuracy_{enhanced_suffix}_ours': [mixed_weighted_accuracy_ours],
        'F1_kpconv': [f1_kpconv],
        'IoU_kpconv': [iou_kpconv],
        'Accuracy_kpconv': [accuracy_kpconv],
        'Accuracy_weighted_kpconv': [weighted_accuracy_kpconv],
        f'Accuracy_{enhanced_suffix}_kpconv': [mixed_weighted_accuracy_kpconv]
    })

    statistics_df = pd.concat([statistics_df, statistics], ignore_index=True)

import dataframe_image as dfi

def replace_country_code(filename):
    country_mapping = {'pol': 'Poland', 'spa': 'Spain', 'fin': 'Finland'}
    for code, country in country_mapping.items():
        if code in filename:
            return country
    return filename

statistics_df['Country'] = statistics_df['File'].str[:3].apply(replace_country_code)
statistics_df = statistics_df.drop(columns='File')

numeric_columns = statistics_df.select_dtypes(include=[np.number]).columns.tolist()
statistics_df = statistics_df.groupby('Country')[numeric_columns].mean().reset_index()

country_df = statistics_df['Country']
statistics_df = statistics_df.drop(columns='Country')
statistics_df = statistics_df.sort_index(axis=1, key=lambda x: x.str[:3])

country_df.reset_index(drop=True, inplace=True)
statistics_df.reset_index(drop=True, inplace=True)
statistics_df = pd.concat([country_df, statistics_df], axis=1).round(8)
statistics_df.columns = statistics_df.columns.str.replace('_', ' ')

# -----------------------------------------------------------------------------
# Persist detailed and summary tables in the region directory
# -----------------------------------------------------------------------------

statistics_df.to_csv(output_csv, index=False)
dfi.export(statistics_df, output_img)

summary_df = pd.DataFrame()
summary_df['Country'] = country_df

for model in ['fsct', 'ours', 'kpconv']:
    summary_df[f'{model.upper()} Accuracy'] = statistics_df[f'Accuracy {model}']
    summary_df[f'{model.upper()} IoU'] = statistics_df[f'IoU {model}']
    summary_df[f'{model.upper()} F1'] = statistics_df[f'F1 {model}']

summary_df = summary_df.round(4)
summary_df.to_csv(summary_csv, index=False)
dfi.export(summary_df, summary_img)
