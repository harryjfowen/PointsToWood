import datetime
start = datetime.datetime.now()
import resource
import os
import os.path as OP
import argparse
from src.preprocessing import preprocess
from src.predicter import SemanticSegmentation
import torch
import shutil
import sys
import numpy as np
import re
from src.io import load_file
from src.utils import configure_threads

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

def preprocess_point_cloud_data(df, zero_reflectance=False):

    canon_map = {
        'label': ['label'],  
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

    if 'truth' in df.columns and 'label' in df.columns:
        df = df.drop(columns=['label'])

    df = df.loc[:, ~df.columns.duplicated()]

    drop_tokens = ["pred", "pwood"]
    cols_to_drop = [c for c in df.columns if any(tok in c for tok in drop_tokens)]
    if len(cols_to_drop):
        df = df.drop(columns=cols_to_drop, errors='ignore')

    if 'reflectance' not in df.columns:
        df['reflectance'] = np.zeros(len(df))
        print('No reflectance detected, column added with zeros.')
    else:
        print('Reflectance detected')
    
    if zero_reflectance:
        df['reflectance'] = np.zeros(len(df))
        print('Reflectance set to zeros as requested.')
    
    xyz_cols = ['x', 'y', 'z']
    required_order = xyz_cols + ['reflectance']
    other_cols = [col for col in df.columns if col not in required_order]
    final_cols = required_order + other_cols
    df = df[final_cols]

    headers = [c for c in df.columns if c not in xyz_cols]
    return df, headers, ('reflectance' in df.columns)

'''
-------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------
'''

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--point-cloud', '-p', default=[], nargs='+', type=str, help='list of point cloud files')    
    parser.add_argument('--odir', type=str, default='.', help='output directory')
    parser.add_argument('--batch-size', default=4, type=int,
                        help="Mini-batch size (reduce if you hit CUDA OOM).")
    parser.add_argument('--num-procs', default=-1, type=int, help="Number of CPU cores you want to use. If you run out of RAM, lower this.")
    parser.add_argument('--resolution', type=float, default=0.02,
                        help='Voxel down-sample resolution [m] (default: 0.02)')
    parser.add_argument('--grid-size', type=float, nargs='+', default=[2.0],
                        help='Voxel grid size in metres (default: 2.0)')
    parser.add_argument('--overlap', type=float, default=0.0,
                        help='Voxel overlap in metres (default: 0.0)')
    parser.add_argument('--min-pts', type=int, default=512,
                        help='Minimum number of points per voxel (default: 512)')
    parser.add_argument('--max-pts', type=int, default=16384, help='Maximum number of points in voxel')
    parser.add_argument('--model', type=str, default='fbeta-eu.pth',
                        help='Model checkpoint name inside pointstowood/model (default: fbeta-eu.pth)')
    parser.add_argument('--is-wood', default=0.5, type=float, help='a probability above which points within KNN are classified as wood')
    parser.add_argument('--any-wood', default=1, type=float, help='a probability above which ANY point within KNN is classified as wood')
    parser.add_argument('--output-fmt', default='ply', help="file type of output")
    parser.add_argument('--verbose', action='store_true', help="print stuff")
    parser.add_argument('--max-probability', action='store_true', default=False,
                        help="Use arg-max (most confident) prediction per KNN neighborhood instead of mean/median aggregation")
    parser.add_argument('--boost-perspective', action='store_true', default=False,
                         help="Enable multi-perspective inference with 7 augmented views")
    parser.add_argument('--denoise', action='store_true', default=False,
                        help="Enable denoising")
    parser.add_argument('--denoise-k', type=int, default=16, help='Number of neighbors for denoising (default: 16)')
    parser.add_argument('--denoise-std', type=float, default=1.0, help='Standard deviation threshold for denoising (default: 1.0)')
    parser.add_argument('--zero-reflectance', action='store_true', default=False,
                        help="Set all reflectance values to zero (for testing XYZ-only performance)")


    args = parser.parse_args()

    configure_threads(args.num_procs)

    if args.verbose:
        print('\n---- parameters used ----')
        for k, v in args.__dict__.items():
            if k == 'pc': v = '{} points'.format(len(v))
            if k == 'global_shift': v = v.values
            print('{:<35}{}'.format(k, v)) 

    args.wdir = get_path()
    args.mode = 'predict' if 'predict' in sys.argv[0] else 'train'
    args.reflectance = False

    '''
    Sanity check---------------------------------------------------------------------------------------------------------
    '''
    if args.point_cloud == '':
        raise Exception('no input specified, please specify --point-cloud')
    
    for point_cloud_file in args.point_cloud:
        if not os.path.isfile(point_cloud_file):
            raise FileNotFoundError(f'Point cloud file not found: {point_cloud_file}')
    
    '''
    If voxel file on disc, delete it.
    '''    
    
    path = OP.dirname(args.point_cloud[0])
    args.vxfile = OP.join(path, "voxels")

    if os.path.exists(args.vxfile): shutil.rmtree(args.vxfile)

    for point_cloud_file in args.point_cloud:

        '''
        Handle input and output file paths-----------------------------------------------------------------------------------
        '''
        
        path = OP.dirname(point_cloud_file)
        file = OP.splitext(OP.basename(point_cloud_file))[0] + "_p2w.ply"
        args.odir = OP.join(path, file)

        if os.path.exists(args.odir):
            try:
                os.remove(args.odir)
            except Exception as e:
                print(f"Warning: could not delete existing output {args.odir}: {e}")

        '''
        Preprocess data into voxels------------------------------------------------------------------------------------------
        '''

        if args.verbose: print('\n----- Preprocessing started -----')

        os.makedirs(args.vxfile, exist_ok=True)
        args.pc, args.headers = load_file(filename=point_cloud_file, additional_headers=True, verbose=False)
        args.pc, args.headers, args.reflectance = preprocess_point_cloud_data(args.pc, args.zero_reflectance)
        
        if args.verbose: print(f'Voxelising to {args.grid_size} grid sizes')
        preprocess(args)
        
        if args.verbose:
            print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
            print(f'runtime: {(datetime.datetime.now() - start).seconds}')
        
        '''
        Run semantic training------------------------------------------------------------------------------------------------
        '''
        if args.verbose: print('\n----- Semantic segmenation started -----')
        
        SemanticSegmentation(args)
        torch.cuda.empty_cache()

        if os.path.exists(args.vxfile):
            shutil.rmtree(args.vxfile)

        if args.verbose:
            print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
            print(f'runtime: {(datetime.datetime.now() - start).seconds}')
