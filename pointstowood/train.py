import datetime
start = datetime.datetime.now()
import resource

import argparse, glob, os
import numpy as np
import shutil
from src.trainer import SemanticTraining
from src.preprocessing import *
from src.io import load_file
import shutil 
import sys
import re

'''
Minor functions-------------------------------------------------------------------------------------------------------------
'''

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def get_path(location_in_pointstowood: str = "") -> str:
    current_wdir = os.getcwd()
    match = re.search(r'PointsToWood.*?pointstowood', current_wdir, re.IGNORECASE)
    if not match:
        raise ValueError('"PointsToWood/pointstowood" not found in the current working directory path')
    last_index = match.end()
    output_path = current_wdir[:last_index]
    if location_in_pointstowood:
        output_path = os.path.join(output_path, location_in_pointstowood)
    return output_path.replace("\\", "/")

def preprocess_point_cloud_data(df):
    # -----------------------------------------------------------------------------
    # Robust column name standardisation
    #   1. Lower-case everything for easy matching.
    #   2. Remove any leading "scalar_" prefix.
    #   3. If a column *contains* one of our canonical keywords, map it directly to
    #      that canonical keyword – regardless of any extra digits/symbols before
    #      or after (e.g. "label_", "my_label123", "reflectance-val").
    # -----------------------------------------------------------------------------

    canon_map = {
        'label': ['label'],  # keep 'truth' as separate column
        'reflectance': ['reflectance', 'refl', 'intensity'],
    }

    new_columns = {}
    for col in df.columns:
        clean = col.lower().replace('scalar_', '') 

        mapped = None
        for target, aliases in canon_map.items():
            if any(alias in clean for alias in aliases):
                mapped = target
                break

        new_columns[col] = mapped if mapped is not None else clean

    df = df.rename(columns=new_columns)
    
    # ------------------------------------------------------------------
    # Ensure there is a single canonical label column called 'label'.
    #   • If both 'truth' and 'label' are present we keep the existing
    #     'truth' (assumed to be cleaner) and drop the duplicate 'label'.
    #   • If only 'truth' exists we rename it to 'label' so downstream
    #     training code always sees the expected column.
    # ------------------------------------------------------------------

    # If both exist, drop the original 'label' and rename 'truth' → 'label'
    if 'truth' in df.columns and 'label' in df.columns:
        df = df.drop(columns=['label'])
    if 'label' not in df.columns and 'truth' in df.columns:
        df = df.rename(columns={'truth': 'label'})

    # Remove duplicated column names if any
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'reflectance' not in df.columns:
        df['reflectance'] = np.zeros(len(df))
        print('No reflectance detected, column added with zeros.')
    else:
        print('Reflectance detected')
    
    xyz_cols = ['x', 'y', 'z']
    
    required_order = xyz_cols + ['reflectance', 'label']
    
    other_cols = [col for col in df.columns if col not in required_order]
    
    final_cols = required_order + other_cols
    df = df[final_cols]
    
    return df, 'reflectance' in df.columns

'''
---------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------
'''

if __name__ == '__main__':

        parser = argparse.ArgumentParser()
        

        parser.add_argument('--device', type=str, default='cuda', help='Insert either "cuda" or "cpu"')
        parser.add_argument('--region', type=str, default='eu', help='Data region (e.g. eu, spain, germany, global)')
        parser.add_argument('--num-procs', type=int, default=1, help='Number of cpu cores to use')
        parser.add_argument('--num-epochs', default=180, type=int, metavar='N', help='number of total epochs to run')
        parser.add_argument('--checkpoint-saves', default=1, type=int, metavar='N', help='number of times to save model')
        parser.add_argument('--model', type=str, default=None, help='Name of global model [e.g. model.pth]')
        parser.add_argument('--resolution', type=float, default=0.02, help='Resolution to which point cloud is downsampled [m]')
        parser.add_argument('--grid-size', type=float, nargs='+', default=[1.0], help='Grid sizes for voxelization')
        parser.add_argument('--min-pts', type=int, default=1024, help='Minimum number of points in voxel')
        parser.add_argument('--max-pts', type=int, default=16384, help='Maximum number of points in voxel')
        parser.add_argument('--batch-size', type=int, default=4, help='Batch size for cuda processing [Lower less memory usage]')
        parser.add_argument('--augmentation', action='store_true', help="Perform data augmentation")
        parser.add_argument('--preprocess', action='store_true', help="Preprocess point clouds into voxels")
        parser.add_argument('--test', action='store_true', help="Perform model testing during training")
        parser.add_argument('--tune', action='store_true', help="Tune model hyperparameters with lower learning rate schedule")
        parser.add_argument('--stop-early', dest='stop_early', action='store_true', help="Break training run if testing loss continually increases")
        parser.add_argument('--wandb', action='store_true', help="Use wandb for logging")
        parser.add_argument('--verbose', action='store_true', help="print stuff")
        parser.add_argument('--balance-mode', dest='balance_mode', type=str, default='downsampling', choices=['downsampling', 'upsampling'], help='Class balancing mode: downsampling or upsampling')
        parser.add_argument('--overlap', type=float, default=0.0, help='Voxel overlap in metres (default: 0.0)')
        parser.add_argument('--denoise', action='store_true', default=False,
                        help="Enable denoising")
        parser.add_argument('--denoise-k', type=int, default=16, help='Number of neighbors for denoising (default: 16)')
        parser.add_argument('--denoise-std', type=float, default=1.0, help='Standard deviation threshold for denoising (default: 1.0)')

        args = parser.parse_args()
        args.wdir = get_path()
        args.mode = 'predict' if 'predict' in sys.argv[0] else 'train'

        if args.model is None:
            args.model = f"{args.region}.pth"
        
        if args.verbose: print('Mode: {}'.format(args.mode))

        '''
        Organise model checkpointing-------------------------------------------------------------------------------------------
        '''

        args.checkpoints = np.arange(0, args.num_epochs+1, int(args.num_epochs / args.checkpoint_saves))

        old_checkpoints = glob.glob(os.path.join(args.wdir,'checkpoints/*.pth'))
        if len(old_checkpoints) > 0:
                shutil.make_archive(os.path.join(args.wdir,'checkpoints_backup'), 'zip', os.path.join(args.wdir,'checkpoints'))
        for f in old_checkpoints:
                os.remove(f)
        
        '''
        Preprocess data into voxels and write to disk----------------------------------------------------------------------------
        '''

        args.train_dir = os.path.join(args.wdir, f'data/{args.region}_train')
        args.test_dir = os.path.join(args.wdir, f'data/{args.region}_test')

        args.trfile = os.path.join(args.train_dir, "voxels")
        args.tefile = os.path.join(args.test_dir, "voxels")

        train_files = glob.glob(os.path.join(args.train_dir, '*.ply'))
        test_files = glob.glob(os.path.join(args.test_dir, '*.ply'))

        if args.preprocess:

                if os.path.exists(args.trfile): shutil.rmtree(args.trfile)

                if args.verbose: print('\n----- Preprocessing started -----')

                for i, p in enumerate(train_files):
                        
                        os.makedirs(args.trfile, exist_ok=True)
                        args.pc, args.headers = load_file(filename=p, additional_headers=True, verbose=True)
                        args.pc, args.reflectance = preprocess_point_cloud_data(args.pc)
                        args.vxfile = args.trfile
                        
                        if args.verbose: print(f'Voxelising to {args.grid_size} grid sizes')
                        preprocess(args)

                if args.test:

                        if os.path.exists(args.tefile): shutil.rmtree(args.tefile)
                        
                        if args.verbose: print("\nTesting")

                        args.mode = 'test'

                        for i, p in enumerate(test_files):

                                os.makedirs(args.tefile, exist_ok=True)
                                args.pc, args.headers = load_file(filename=p, additional_headers=True, verbose=True)
                                args.pc, args.reflectance = preprocess_point_cloud_data(args.pc)
                                args.vxfile = args.tefile

                                if args.verbose: print(f'Voxelising to {args.grid_size} grid sizes')
                                preprocess(args)

                if args.verbose:
                        print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
                        print(f'runtime: {(datetime.datetime.now() - start).seconds}')

        if args.augmentation:
                if args.verbose: print('Training with data augmentation')


        '''
        Sanity checks-------------------------------------------------------------------------------------------------------------
        '''

        if len(args.checkpoints) == 0:
                args.checkpoints == np.asarray([args.num_epochs-1])


        '''
        Run semantic training-----------------------------------------------------------------------------------------------------
        '''
        if args.verbose: print('\n----- Semantic segmenation started -----')

        SemanticTraining(args)

        if args.verbose:
                print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
                print(f'runtime: {(datetime.datetime.now() - start).seconds}')

