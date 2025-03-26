import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.io import load_file, save_file
import sys
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict

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

# Create an empty DataFrame to store the statistics
statistics_df = pd.DataFrame(columns=['File', 'Precision_fsct', 'Recall_fsct', 'Precision_wood', 'Recall_wood', 'Accuracy_fsct', 'Accuracy_wood', 'Weighted_accuracy_fsct', 'Weighted_accuracy_wood'])

# Get a list of all the file names in the folder
file_names = glob.glob(os.path.join(sys.argv[1], '*fsct.ply'))

# Wrap file_names with tqdm for the progress bar
for file_name in tqdm(file_names, desc='Processing files', unit='file'):
    # Load the wood and fsct point clouds
    base_name = os.path.basename(file_name).replace('_fsct.ply', '')
    wood_file = os.path.join(sys.argv[1], base_name + '_wood.ply')
    fsct_file = os.path.join(sys.argv[1], base_name + '_fsct.ply')
    print(file_name)
    wood = load_file(wood_file)
    wood.rename(columns=lambda x: x.replace('scalar_', '') if 'scalar_' in x else x, inplace=True)
    wood = wood[(wood['label'] != 2)]
    fsct = load_file(fsct_file)
    fsct.rename(columns=lambda x: x.replace('scalar_', '') if 'scalar_' in x else x, inplace=True)
    fsct = fsct[(fsct['label'] != 2)]
    #fsct = fsct[(fsct['label'] != 0) & (fsct['label'] != 2)]
    #fsct['label'] = fsct['label'].replace({1: 0, 3: 1})

    # Assuming 'df' is your DataFrame
    if 'label' in wood.columns:
        # Get all occurrences of 'label' in the columns
        label_cols = [col for col in wood.columns if col == 'label']
        if len(label_cols) > 1:
            # Find the first occurrence and drop it
            first_label_index = wood.columns.get_loc('label')
            # Create a new DataFrame without the first 'label' column
            wood = pd.concat([
                wood.iloc[:, :first_label_index],  # Columns before the first 'label'
                wood.iloc[:, first_label_index+1:]  # Columns after the first 'label'
            ], axis=1)

        # Assuming 'df' is your DataFrame
    if 'label' in fsct.columns:
        # Get all occurrences of 'label' in the columns
        label_cols = [col for col in fsct.columns if col == 'label']
        if len(label_cols) > 1:
            # Find the first occurrence and drop it
            first_label_index = fsct.columns.get_loc('label')
            # Create a new DataFrame without the first 'label' column
            fsct = pd.concat([
                fsct.iloc[:, :first_label_index],  # Columns before the first 'label'
                fsct.iloc[:, first_label_index+1:]  # Columns after the first 'label'
            ], axis=1)

    fsct['label'] = (fsct['label'] == 3).astype(int) if fsct['label'].nunique() > 2 else fsct['label']

    # Downsample the point clouds
    # downsampler = PointCloudDownsampler(wood[['x','y','z']].values, 0.02)
    # wood = wood.iloc[downsampler.random_voxelisation()]
    # wood.reset_index(drop=True, inplace=True)
    
    # downsampler = PointCloudDownsampler(fsct[['x','y','z']].values, 0.02)
    # fsct = fsct.iloc[downsampler.random_voxelisation()]
    # fsct.reset_index(drop=True, inplace=True)
    
    # Compare the two point clouds and calculate the statistics
    f1_fsct = f1_score(fsct[['truth']].astype(int), fsct[['label']].astype(int), average='binary', zero_division=0)
    f1_wood = f1_score(wood[['truth']].astype(int), wood[['label']].astype(int), average='binary', zero_division=0)

    precision_fsct = precision_score(fsct[['truth']].astype(int), fsct[['label']].astype(int), average='binary', zero_division=0)
    recall_fsct = recall_score(fsct[['truth']].astype(int), fsct[['label']].astype(int), average='binary', zero_division=0)

    precision_wood = precision_score(wood[['truth']].astype(int), wood[['label']].astype(int), average='binary', zero_division=0)
    recall_wood = recall_score(wood[['truth']].astype(int), wood[['label']].astype(int), average='binary', zero_division=0)

    accuracy_fsct = balanced_accuracy_score(fsct[['truth']].astype(int), fsct[['label']].astype(int))
    accuracy_wood = balanced_accuracy_score(wood[['truth']].astype(int), wood[['label']].astype(int))

    print(file_name)
    print(f'Accuracy fsct: {accuracy_fsct}, Accuracy wood: {accuracy_wood}')

    if 'pathlength' not in fsct.columns:
        fsct['pathlength'] = 1
    if 'pathlength' not in wood.columns:
        wood['pathlength'] = 1

    weighted_accuracy_fsct = balanced_accuracy_score(fsct[['truth']].astype(int), fsct[['label']].astype(int), sample_weight=fsct['pathlength'])
    weighted_accuracy_wood = balanced_accuracy_score(wood[['truth']].astype(int), wood[['label']].astype(int), sample_weight=wood['pathlength'])

    # Create a DataFrame with the statistics
    statistics = pd.DataFrame({
        'File': [base_name],
        #'F1_fsct': [f1_fsct],
        #'F1_wood': [f1_wood],
        'Precision_fsct': [precision_fsct],
        'Recall_fsct': [recall_fsct],
        'Precision_wood': [precision_wood],
        'Recall_wood': [recall_wood],
        'Accuracy_fsct': [accuracy_fsct],
        'Accuracy_wood': [accuracy_wood],
        'Accuracy_weighted_fsct': [weighted_accuracy_fsct],
        'Accuracy_weighted_wood': [weighted_accuracy_wood]
    })

    # Concatenate the statistics DataFrame with the existing statistics_df DataFrame
    statistics_df = pd.concat([statistics_df, statistics], ignore_index=True)

import dataframe_image as dfi

# Define a function to replace country code in filename
def replace_country_code(filename):
    country_mapping = {'pol': 'Poland', 'spa': 'Spain', 'fin': 'Finland'}
    for code, country in country_mapping.items():
        if code in filename:
            return country
    return filename

# Create 'Country' column from 'File' column
statistics_df['Country'] = statistics_df['File'].str[:3].apply(replace_country_code)

# Drop 'File' column
statistics_df = statistics_df.drop(columns='File')

# Calculate the mean of numeric columns grouped by country
numeric_columns = statistics_df.select_dtypes(include=[np.number]).columns.tolist()
statistics_df = statistics_df.groupby('Country')[numeric_columns].mean().reset_index()

# Separate 'Country' column
country_df = statistics_df['Country']

# Drop 'Country' column from statistics_df
statistics_df = statistics_df.drop(columns='Country')

# Sort remaining columns by the first three characters
statistics_df = statistics_df.sort_index(axis=1, key=lambda x: x.str[:3])

# Concatenate 'Country' column back
country_df.reset_index(drop=True, inplace=True)
statistics_df.reset_index(drop=True, inplace=True)
statistics_df = pd.concat([country_df, statistics_df], axis=1).round(8)

# Replace underscore with space in column names
statistics_df.columns = statistics_df.columns.str.replace('_', ' ')

# Save the DataFrame to a CSV file
output_file = os.path.join(sys.argv[1], 'results.csv')
statistics_df.to_csv(output_file, index=False)

# Export the DataFrame as an image
dfi.export(statistics_df, os.path.join(sys.argv[1], 'results.png'))
