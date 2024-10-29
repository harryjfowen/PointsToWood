import glob
import argparse
import warnings
import string
import shutil
from src.preprocessing import PointCloudDownsampler
from src.io import load_file, save_file
import networkx as nx
from scipy.spatial import ConvexHull
import fast_hdbscan
from pykdtree.kdtree import KDTree
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os
import resource
import datetime
import sys
start = datetime.datetime.now()
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import ConvexHull
from shapely.geometry import Point, Polygon
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

class dict2class:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

class PointCloudDownsampler:
    def __init__(self, pc, vlength):
        self.pc = pc
        self.vlength = vlength

    def random(self):
        voxel_indices = np.floor(self.pc[:, :3] / self.vlength).astype(int)
        voxel_dict = defaultdict(list)
        for i, voxel_index in enumerate(voxel_indices):
            voxel_dict[tuple(voxel_index)].append(i)
        selected_indices = [voxel_points_indices[np.random.randint(len(voxel_points_indices))] for voxel_points_indices in voxel_dict.values()]
        return selected_indices


def generate_path(samples, origins, n_neighbours=200, max_length=0):

    nn = KDTree(samples[['x', 'y', 'z']].values.astype('float32'))
    distances, indices = nn.query(samples[['x', 'y', 'z']].values.astype('float32'), k=n_neighbours)

    from_to_all = pd.DataFrame(np.vstack([np.repeat(samples.clstr.values, n_neighbours),
                                          samples.iloc[indices.ravel()
                                                       ].clstr.values,
                                          distances.ravel()]).T,
                               columns=['source', 'target', 'length'])
    from_to_all = from_to_all.loc[from_to_all.target != from_to_all.source]
    edges = from_to_all.groupby(['source', 'target']).length.min().reset_index()
    edges = edges.loc[edges.length <= max_length]
    origins = [s for s in origins if s in edges.source.values]
    G = nx.from_pandas_edgelist(edges, edge_attr=['length'])
    distance, shortest_path = nx.multi_source_dijkstra(G,
                                                       sources=origins,
                                                       weight='length')
    paths = pd.DataFrame(index=distance.keys(),
                         data=distance.values(), columns=['distance'])
    paths.loc[:, 'base'] = params.not_base
    for p in paths.index:
        paths.loc[p, 'base'] = shortest_path[p][0]
    paths.reset_index(inplace=True)
    paths.columns = ['clstr', 'distance', 't_clstr']
    node_occurance = {}
    for v in shortest_path.values():
        for n in v:
            if n in node_occurance.keys():
                node_occurance[n] += 1
            else:
                node_occurance[n] = 1

    tips = [k for k, v in node_occurance.items() if v == 1]

    paths.loc[:, 'is_tip'] = False
    paths.loc[paths.clstr.isin(tips), 'is_tip'] = True
    return paths


def cube(pc):
    try:
        if len(pc) > 5:
            vertices = ConvexHull(pc[['x', 'y', 'z']]).vertices
            idx = np.random.choice(vertices, size=len(vertices), replace=False)
            return pc.loc[pc.index[idx]]
    except:
        pass
    return pc


def voxelise(tmp, length, method='random', z=True):
    tmp.loc[:, 'xx'] = tmp.x // length * length
    tmp.loc[:, 'yy'] = tmp.y // length * length
    if z:
        tmp.loc[:, 'zz'] = tmp.z // length * length
    if method == 'random':
        def code(): return ''.join(np.random.choice(
            [x for x in string.ascii_letters], size=8))
        xD = {x: code() for x in tmp.xx.unique()}
        yD = {y: code() for y in tmp.yy.unique()}
        if z:
            zD = {z: code() for z in tmp.zz.unique()}
        tmp.loc[:, 'VX'] = tmp.xx.map(xD) + tmp.yy.map(yD)
        if z:
            tmp.VX += tmp.zz.map(zD)
    elif method == 'bytes':
        def code(row): return np.array(
            [row.xx, row.yy] + [row.zz] if z else []).tobytes()
        tmp.loc[:, 'VX'] = tmp.apply(code, axis=1)
    else:
        raise Exception('method {} not recognised: choose "random" or "bytes"')
    return tmp


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tiles', '-t', type=str, default='',
                        required=False, help='fsct directory')
    parser.add_argument('--resolution', '-r', type=float, default=0.02,
                        required=False, help='voxel length')
    parser.add_argument('--odir', '-o', type=str,
                        help='path to tile index')
    parser.add_argument('--prefix', '-p', type=str, default='PLOT',
                    help='prefix to append to tree filenames')
    parser.add_argument('--scanpos', type=str, required=True,
                        help='path to scan positions file')
    parser.add_argument('--tile-buffer-distance', default=10, type=int,
                        help='distance thresholddetermining which neighbouring tiles to load')
    parser.add_argument('--all-tiles', action='store_true',
                        help='reads in all tiles')

    parser.add_argument('--overlap', default=False, type=float,
                        help='buffer to crop adjacent tiles')
    parser.add_argument('--slice-thickness', default=0.1,
                        type=float, help='slice thickness for constructing graph')
    parser.add_argument('--find-stems-height', default=1.5,
                        type=float, help='height for identifying stems')
    parser.add_argument('--find-stems-thickness', default=.5, type=float,
                        help='thickness of slice used for identifying stems')
    parser.add_argument('--find-stems-min-radius', default=.075,
                        type=float, help='minimum radius of found stems')
    parser.add_argument('--find-stems-min-points', default=200,
                        type=int, help='minimum number of points for found stems')
    parser.add_argument('--graph-edge-length', default=0.20, type=float,
                        help='maximum distance used to connect points in graph')
    parser.add_argument('--graph-maximum-cumulative-gap', default=np.inf, type=float,
                        help='maximum cumulative distance between a base and a cluster')
    parser.add_argument('--min-points-per-tree', default=0, type=int,
                        help='minimum number of points for a identified tree')
    parser.add_argument('--min-height-per-tree', default=0, type=int,
                        help='minimum number of points for a identified tree')
    parser.add_argument('--add-leaves', action='store_true',
                        help='add leaf points')
    parser.add_argument('--add-leaves-voxel-length', default=0.5,
                        type=float, help='voxel sixe when add leaves')
    parser.add_argument('--add-leaves-edge-length', default=0.10, type=float,
                        help='maximum distance used to connect points in leaf graph')
    parser.add_argument('--save-diameter-class', action='store_true',
                        help='save into dimater class directories')
    parser.add_argument('--ignore-missing-tiles', action='store_true',
                        help='ignore missing neighbouring tiles')
    parser.add_argument('--pandarallel', action='store_true',
                        help='use pandarallel')
    parser.add_argument('--verbose', action='store_true',
                        help='print something')
    params = parser.parse_args()

    if params.pandarallel:
        try:
            from pandarallel import pandarallel
            pandarallel.initialize(
                nb_workers=32, progress_bar=True if params.verbose else False)
        except:
            print('--- pandarallel not installed ---')
            params.pandarallel = False

    if params.verbose:
        print('---- parameters ----')
        for k, v in params.__dict__.items():
            print(f'{k:<35}{v}')

    # clear existing output directory
    if os.path.exists(params.odir):
        shutil.rmtree(params.odir)
    os.makedirs(params.odir)

    params.not_base = -1
    xyz = ['x', 'y', 'z']  # shorthand

    # read in all tiles
    params.dir = os.path.dirname(params.tiles)
    matching_files = glob.glob(os.path.join(params.dir, '*lw.ply'))

    def process_tile(tile_file):
        # Extract the part of the file name between the underscore and the hyphen
        n = os.path.basename(tile_file).split('_')[1].split('-')[0]
        n = int(n)  # Convert to integer
        tmp, params.headers = load_file(tile_file, additional_headers=True)
        downsampler = PointCloudDownsampler(tmp[['x','y','z']].values, params.resolution)
        tmp = tmp.iloc[downsampler.random()]
        tmp['buffer'] = False
        tmp['fn'] = n
        return tmp

    if len(matching_files) == 1:
        params.pc, params.headers = load_file(matching_files[0], additional_headers=True)
        params.pc['buffer'] = False
        params.pc['fn'] = 0
    else:
        with ProcessPoolExecutor(max_workers=32) as executor:
            total_tiles = len(matching_files)
            futures = {executor.submit(process_tile, tile_file): tile_file for tile_file in matching_files}
            results = []
            with tqdm(total=total_tiles, desc='Processing Tiles') as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                    pbar.update(1)
            params.pc = pd.concat(results, ignore_index=True)
        
        # --- this can be dropeed soon ---
        if 'nz' in params.pc.columns:
            params.pc.rename(columns={'nz': 'n_z'}, inplace=True)

    # save space
    params.pc = params.pc[[c for c in ['x', 'y', 'z','n_z', 'label', 'pwood', 'buffer', 'fn']]]
    params.pc[['x', 'y', 'z', 'n_z', 'pwood']] = params.pc[['x', 'y', 'z', 'n_z', 'pwood']] .astype(np.float32)
    params.pc[['label','fn']] = params.pc[['label', 'fn']].dropna().astype(np.int16)

    #Label adjustment based on pwood
    #params.pc.loc[params.pc.pwood > 0.4, 'label'] = 1

    # generate skeleton points
    if params.verbose:
        print('\n----- skeletonisation started -----')

    # extract stems points and slice slice
    stem_pc = params.pc.loc[params.pc.label == 1]
    #stem_pc = params.pc[params.pc.pwood > 0.4].copy()
    #stem_pc['label'] = 1

    print("Downsampling stem pc")
    downsampler = PointCloudDownsampler(stem_pc[['x','y','z']].values, params.resolution)
    stem_pc = stem_pc.iloc[downsampler.random()]

    # slice stem_pc
    stem_pc.loc[:, 'slice'] = (stem_pc.z // params.slice_thickness).astype(int) * params.slice_thickness
    stem_pc.loc[:, 'n_slice'] = (stem_pc.n_z // params.slice_thickness).astype(int)

    # cluster within height slices
    stem_pc.loc[:, 'clstr'] = -1
    label_offset = 0

    for slice_height in tqdm(np.sort(stem_pc.n_slice.unique()),
                                disable=False if params.verbose else True,
                                desc='slice data vertically and clustering'):

        new_slice = stem_pc.loc[stem_pc.n_slice == slice_height]

        if len(new_slice) > 3:
            dbscan = fast_hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(new_slice[xyz])
            new_slice.loc[:, 'clstr'] = dbscan
            new_slice.loc[new_slice.clstr > -1, 'clstr'] += label_offset
            stem_pc.loc[new_slice.index, 'clstr'] = new_slice.clstr
            label_offset = stem_pc.clstr.max() + 1

    # group skeleton points
    grouped = stem_pc.loc[stem_pc.clstr != -1].groupby('clstr')
    if params.verbose:
        print('fitting convex hulls to clusters')
    if params.pandarallel:
        chull = grouped.parallel_apply(cube)
    else:
        chull = grouped.apply(cube)
    chull = chull.reset_index(drop=True)

    # identify possible stems
    if params.verbose: print('identifying stems...')
    skeleton = grouped[xyz + ['n_z', 'n_slice', 'slice']].median().reset_index()
    skeleton.loc[:, 'dbh_node'] = False

    find_stems_min = int(params.find_stems_height // params.slice_thickness)
    find_stems_max = int((params.find_stems_height + params.find_stems_thickness) // params.slice_thickness) + 1
    dbh_slice = stem_pc.loc[stem_pc.n_slice.between(find_stems_min, find_stems_max)]
    dbh_slice = dbh_slice[dbh_slice.pwood > 0.66]

    if len(dbh_slice) > 0:
        dbscan = fast_hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(dbh_slice[['x', 'y']])
        dbh_slice.loc[:, 'clstr_db'] = dbscan
        dbh_slice = dbh_slice.loc[dbh_slice.clstr_db > -1]
        dbh_slice.loc[:, 'cclstr'] = dbh_slice.groupby('clstr_db').ngroup()
        skeleton.loc[skeleton.clstr.isin(dbh_slice.cclstr.unique()), 'dbh_node'] = True

    in_tile_stem_nodes = skeleton.loc[skeleton.dbh_node, 'clstr']
    num_dbh_nodes = len(in_tile_stem_nodes)
    print(f'Number of DBH nodes found: {num_dbh_nodes}')

    dbh_node_clusters = dbh_slice[dbh_slice.cclstr.isin(in_tile_stem_nodes)]

    if in_tile_stem_nodes.empty:
        sys.exit()
    
    ### READ IN SCAN POSITIONS TEXT FILE 
    scan_positions = pd.read_csv(params.scanpos, sep=',', names=['x', 'y', 'z'])    
    scan_pos_hull = ConvexHull(scan_positions[['x', 'y']])
    plot_hull = Polygon(scan_positions.iloc[scan_pos_hull.vertices][['x', 'y']].values)

    # generates paths through all stem points
    if params.verbose:
        print('generating graph, this may take a while...')
    wood_paths = generate_path(chull,
                                skeleton.loc[skeleton.dbh_node].clstr,
                                n_neighbours=128,
                                max_length=params.graph_edge_length)

    # removes paths that are longer for same clstr
    wood_paths = wood_paths.sort_values(['clstr', 'distance'])
    wood_paths = wood_paths.loc[~wood_paths['clstr'].duplicated()]

    if params.verbose:
        print('merging skeleton points with graph')
    stems = pd.merge(skeleton, wood_paths, on='clstr', how='left')

    # give a unique colour to each tree (helps with visualising)
    stems.drop(columns=[c for c in stems.columns if c.startswith('red') or
                        c.startswith('green') or
                        c.startswith('blue')], inplace=True)

    # generate unique RGB for each stem
    unique_stems = stems.t_clstr.unique()
    RGB = pd.DataFrame(data=np.vstack([unique_stems,
                                        np.random.randint(0, 255, size=(3, len(unique_stems)))]).T,
                        columns=['t_clstr', 'red', 'green', 'blue'])
    
    RGB.loc[RGB.t_clstr == params.not_base, :] = [np.nan, 211, 211, 211]
    stems = pd.merge(stems, RGB, on='t_clstr', how='right')

    # read in all "stems" tiles and assign all stem points to a tree
    trees = pd.merge(stem_pc, stems[['clstr', 't_clstr', 'distance', 'red', 'green', 'blue']], on='clstr')
    trees.loc[:, 'cnt'] = trees.groupby('t_clstr').t_clstr.transform('count')
    trees.loc[:, 'height'] = trees.groupby('t_clstr')['n_z'].transform('max')

    trees = trees.loc[trees.cnt > params.min_points_per_tree]
    trees = trees.loc[trees.height > params.min_height_per_tree]

    dbh_node_xy = skeleton[skeleton.dbh_node].groupby('clstr')[['x', 'y']].median()
    in_hull = dbh_node_xy.apply(lambda row: Point(row['x'], row['y']).within(plot_hull), axis=1)
    trees['in_plot'] = trees.t_clstr.map(in_hull).fillna(0).astype(int)
    in_tile_stem_nodes = dbh_node_xy[in_hull].index.tolist()

    print(f'Number of trees found: {len(trees.t_clstr.unique())}')

    # write out all trees
    if not params.add_leaves:
        params.base_I, I = {}, 0
        for i, b in tqdm(enumerate(trees.t_clstr.unique()),
                            total=len(trees.t_clstr.unique()),
                            desc='writing stems to file',
                            disable=False if params.verbose else True):

            if b == params.not_base:
                continue
            params.n = 0
            save_file(os.path.join(params.odir, f'{params.prefix}_T{I}.ply'),
                            trees.loc[trees.t_clstr == b], additional_fields=['label', 'pwood', 'n_z', 'red', 'green', 'blue'])
            params.base_I[b] = I
            I += 1

    if params.add_leaves:

        if params.verbose:
            print('adding leaves to stems, this may take a while...')

        # link stem number to clstr
        stem2tlsctr = trees[['clstr', 't_clstr']].loc[trees.t_clstr != params.not_base].set_index('clstr').to_dict()['t_clstr']
        chull.loc[:, 'stem'] = chull.clstr.map(stem2tlsctr)

        # identify unlabelled woody points to add back to leaves
        unlabelled_wood = chull.loc[[True if np.isnan(s) else False for s in chull.stem]]
        unlabelled_wood = stem_pc.loc[stem_pc.clstr.isin(unlabelled_wood.clstr.to_list() + [-1])]
        unlabelled_wood = unlabelled_wood.loc[unlabelled_wood.n_z >= 2]

        # extract wood points that are attributed to a base and that are the
        # the last clstr of the graph i.e. a tip
        is_tip = wood_paths.set_index('clstr')['is_tip'].to_dict()
        chull = chull.loc[[False if np.isnan(s) else True for s in chull.stem]]
        chull.loc[:, 'is_tip'] = chull.clstr.map(is_tip)
        chull = chull.loc[(chull.is_tip) & (chull.n_z > params.find_stems_height)]
        chull.loc[:, 'xlabel'] = 2

        # process leaf points
        lvs = params.pc.loc[(params.pc.label == 0) & (params.pc.n_z >= 0.5)].copy()
        lvs = lvs.append(unlabelled_wood, ignore_index=True)
        downsampler = PointCloudDownsampler(lvs[['x','y','z']].values, params.resolution)
        lvs = lvs.iloc[downsampler.random()]
        lvs.reset_index(inplace=True)

        # assign cluster labels
        lvs.loc[:, 'clstr'] = np.arange(len(lvs)) + 1 + chull.clstr.max()

        # voxelise
        lvs = voxelise(lvs, length=params.add_leaves_voxel_length)
        lvs_gb = lvs.groupby('VX')[xyz]
        lvs_min = lvs_gb.min()
        lvs_max = lvs_gb.max()
        lvs_med = lvs_gb.median()

        # find faces of leaf voxels and create database 
        cnrs = np.vstack([lvs_min.x, lvs_med.y, lvs_med.z]).T
        clstr = np.tile(np.arange(len(lvs_min.index)) + 1 + chull.clstr.max(), 6)
        VX = np.tile(lvs_min.index, 6)
        cnrs = np.vstack([cnrs, np.vstack([lvs_max.x, lvs_med.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_min.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_max.y, lvs_med.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_med.y, lvs_min.z]).T])
        cnrs = np.vstack([cnrs, np.vstack([lvs_med.x, lvs_med.y, lvs_max.z]).T])
        cnrs = pd.DataFrame(cnrs, columns=['x', 'y', 'z'])
        cnrs.loc[:, 'xlabel'] = 1
        cnrs.loc[:, 'clstr'] = clstr
        cnrs.loc[:, 'VX'] = VX

        # and combine leaves and wood
        branch_and_leaves = cnrs.append(chull[['x', 'y', 'z', 'label', 'stem', 'xlabel', 'clstr']])
        branch_and_leaves.reset_index(inplace=True, drop=True)

        # find neighbouring branch and leaf points - used as entry points
        nn = KDTree(branch_and_leaves[xyz].values.astype('float32'))
        distances, indices = nn.query(branch_and_leaves[xyz].values.astype('float32'), k=2)

        closest_point_to_leaf = indices[:len(lvs), :].flatten()  # only leaf points
        idx = np.isin(closest_point_to_leaf,branch_and_leaves.loc[branch_and_leaves.xlabel == 2].index)
        # points where the branch is closest
        close_branch_points = closest_point_to_leaf[idx]

        # remove all branch points that are not close to leaves
        idx = np.hstack([branch_and_leaves.iloc[:len(cnrs)].index.values, close_branch_points])
        bal = branch_and_leaves.loc[branch_and_leaves.index.isin(np.unique(idx))]

        # generate a leaf paths graph
        leaf_paths = generate_path(bal,
                                    bal.loc[bal.xlabel == 2].clstr.unique(),
                                    max_length=1.0,  # i.e. any leaves which are separated by greater are ignored
                                    n_neighbours=64)

        leaf_paths = leaf_paths.sort_values(['clstr', 'distance'])
        # removes duplicate paths
        leaf_paths = leaf_paths.loc[~leaf_paths['clstr'].duplicated()]
        # removes within cluseter paths
        leaf_paths = leaf_paths.loc[leaf_paths.distance > 0]

        # linking indexs to stem number
        top2stem = branch_and_leaves.loc[branch_and_leaves.xlabel == 2].set_index('clstr')['stem'].to_dict()
        leaf_paths.loc[:, 't_clstr'] = leaf_paths.t_clstr.map(top2stem)

        # linking index to VX number
        index2VX = branch_and_leaves.loc[branch_and_leaves.xlabel == 1].set_index('clstr')['VX'].to_dict()
        leaf_paths.loc[:, 'VX'] = leaf_paths['clstr'].map(index2VX)

        # colour the same as stem
        lvs = pd.merge(lvs, leaf_paths[['VX', 't_clstr', 'distance']], on='VX', how='left')
        
        lvs.loc[:, 'height'] = lvs.groupby('t_clstr')['n_z'].transform('max')
        lvs = lvs.loc[lvs.height > params.min_height_per_tree]
        lvs['in_plot'] = lvs.t_clstr.map(in_hull).fillna(0).astype(int)

        trees = trees[['x', 'y', 'z', 'n_z', 'label', 'pwood', 't_clstr', 'in_plot']]
        trees = trees.append(lvs[['x', 'y', 'z', 'n_z', 'label', 'pwood', 't_clstr', 'in_plot']])


        # and save
        params.base_I, I = {}, 0
        for i, b in tqdm(enumerate(trees['t_clstr'].unique()),
                            total=len(trees['t_clstr'].unique()),
                            desc='writing trees with leaves to file',
                            disable=False if params.verbose else True):
            
            params.n = 0

            # Get the subset of lvs for the current b
            trees_subset = trees.loc[trees.t_clstr == b]

            # Generate random RGB values
            rgb = np.random.randint(0, 256, size=3)

            # Check if label is 1
            trees_subset.loc[trees_subset['label'] == 1, ['red', 'green', 'blue']] = 0

            # For other labels, generate random RGB values
            trees_subset.loc[trees_subset['label'] != 1, ['red', 'green', 'blue']] = np.random.randint(0, 256, size=3)

            save_file(os.path.join(params.odir, f'{params.prefix}_T{I}.ply'),
                            trees_subset, additional_fields=['label', 'pwood', 'n_z', 'red', 'green', 'blue'])
            params.base_I[b] = I
            I += 1

print(f'peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}')
print(f'runtime: {(datetime.datetime.now() - start).seconds}')
