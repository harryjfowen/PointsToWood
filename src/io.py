import sys
import argparse
import numpy as np
import pandas as pd
import os 

'''
Read and write PLY formatted point clouds--------------------------------------------------------------------------------------------
'''

def read_ply(fp, newline=None):

    encoding = 'ISO-8859-1' if sys.version_info > (3, 0) else None
    newline = '\n' if sys.platform == 'win32' and sys.version_info > (3, 0) else None

    with open(fp, encoding=encoding, newline=newline) as ply:
 
        length = 0
        prop = []
        dtype_map = {'uint16':'uint16', 'uint8':'uint8', 'double':'d', 'float64':'f8', 
                     'float32':'f4', 'float': 'f4', 'uchar': 'B', 'int':'i'}
        dtype = []
        fmt = 'binary'
    
        for i, line in enumerate(ply):
            length += len(line)
            split_line = line.split()
            if i == 1 and 'ascii' in line:
                fmt = 'ascii' 
            if 'element vertex' in line: N = int(split_line[2])
            if 'property' in line: 
                if split_line[1] not in dtype_map:
                    # Handle int32 specifically
                    if split_line[1] == 'int32':
                        dtype.append('i4')
                    else:
                        dtype.append('i')  # Default to int
                else:
                    dtype.append(dtype_map[split_line[1]])
                prop.append(split_line[2])
            if 'element face' in line:
                raise Exception('.ply appears to be a mesh')
            if 'end_header' in line: break
    
        ply.seek(length)

        if fmt == 'binary':
            arr = np.fromfile(ply, dtype=','.join(dtype))
        else:
            arr = np.loadtxt(ply)
        df = pd.DataFrame(data=arr)
        df.columns = prop

    return df

def write_ply(output_name, pc, comments=[]):

    cols = ['x', 'y', 'z']
    pc = pc.astype({'x': 'float64', 'y': 'float64', 'z': 'float64'})

    with open(output_name, 'w') as ply:

        ply.write("ply\n")
        ply.write('format binary_little_endian 1.0\n')
        ply.write("comment Author: Phil Wilkes\n")
        for comment in comments:
            ply.write("comment {}\n".format(comment))
        ply.write("obj_info generated with pcd2ply.py\n")
        ply.write("element vertex {}\n".format(len(pc)))
        ply.write("property float64 x\n")
        ply.write("property float64 y\n")
        ply.write("property float64 z\n")
        if 'red' in pc.columns:
            cols += ['red', 'green', 'blue']
            pc[['red', 'green', 'blue']] = pc[['red', 'green', 'blue']].astype('i')
            ply.write("property int red\n")
            ply.write("property int green\n")
            ply.write("property int blue\n")
        for col in pc.columns:
            if col in cols: continue
            try:
                pc[col] = pc[col].astype('float64')
                ply.write("property float64 {}\n".format(col))
                cols += [col]
            except:
                pass
        ply.write("end_header\n")

    with open(output_name, 'ab') as ply:
        ply.write(pc[cols].to_records(index=False).tobytes()) 


'''
Read and write PCD formatted point clouds----------------------------------------------------------------------------------------------------
'''

def read_pcd(fp):

    if (sys.version_info > (3, 0)):
        open_file = open(fp, encoding='ISO-8859-1')
    else:
        open_file = open(fp)

    with open_file as pcd:

        length = 0

        for i, line in enumerate(pcd.readlines()):
            length += len(line)
            if 'WIDTH' in line: N = int(line.split()[1])
            if 'FIELDS' in line: F = line.split()[1:]
            if 'DATA' in line:
                fmt = line.split()[1]
                break

        if fmt == 'binary':
            pcd.seek(length)
            arr = np.fromfile(pcd, dtype='f')

            arr = arr[:N*len(F)].reshape(-1, len(F))
            df = pd.DataFrame(arr, columns=F)

    if fmt == 'ascii':
        df = pd.read_csv(fp, sep=' ', names=F, skiprows=11)

    return df

def write_pcd(df, path, binary=True):

    columns = ['x', 'y', 'z', 'intensity']
    df.rename(columns={'scalar_intensity':'intensity'}, inplace=True)
    if 'intensity' not in df.columns: columns = columns[:3]

    with open(path, 'w') as pcd:

        pcd.write('# .PCD v0.7 - Point Cloud Data file format\n')
        pcd.write('VERSION 0.7\n')
        pcd.write('FIELDS ' + ' '.join(columns + ['\n']))
        pcd.write('SIZE ' + '4 ' * len(columns) + '\n')
        pcd.write('TYPE ' + 'F ' * len(columns) + '\n')
        pcd.write('COUNT ' + '1 ' * len(columns) + '\n')
        pcd.write('WIDTH {}\n'.format(len(df)))
        pcd.write('HEIGHT 1\n')
        pcd.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        pcd.write('POINTS {}\n'.format(len(df)))
        pcd.write('DATA binary\n')

    with open(path, 'ab') as pcd:
        df[columns].values.astype('f4').tofile(pcd)


'''
Read and Write functions-------------------------------------------------------------------------------------------------------------
'''

def load_file(filename, additional_headers=False, verbose=False):
    
    file_extension = os.path.splitext(filename)[1]
    headers = ['x', 'y', 'z']

    if file_extension == '.las' or file_extension == '.laz':

        import laspy

        inFile = laspy.read(filename)
        pc = np.vstack((inFile.x, inFile.y, inFile.z))
        pc = pd.DataFrame(data=pc.T, columns=['x', 'y', 'z'])

    elif file_extension == '.ply':
        pc = read_ply(filename)
        
    elif file_extension == '.pcd':
        pc = read_pcd(filename)
        
    else:
        raise Exception('point cloud format not recognised' + filename)

    original_num_points = len(pc)
    
    if verbose: print(f'read in {filename} with {len(pc)} points')
   
    if additional_headers:
        return pc, [c for c in pc.columns if c not in ['x', 'y', 'z']]
    else: return pc


def save_file(filename, pointcloud, additional_fields=[], verbose=False):
    if verbose:
        print('Saving file:', filename)
        
    cols = ['x', 'y', 'z'] + additional_fields

    if filename.endswith('.las'):
        las = laspy.create(file_version="1.4", point_format=7)
        las.header.offsets = np.min(pointcloud[:, :3], axis=0)
        las.header.scales = [0.001, 0.001, 0.001]

        las.x = pointcloud[:, 0]
        las.y = pointcloud[:, 1]
        las.z = pointcloud[:, 2]

        if len(additional_fields) != 0:
            additional_fields = additional_fields[3:]

            #  The reverse step below just puts the headings in the preferred order. They are backwards without it.
            col_idxs = list(range(3, pointcloud.shape[1]))
            additional_fields.reverse()

            col_idxs.reverse()
            for header, i in zip(additional_fields, col_idxs):
                column = pointcloud[:, i]
                if header in ['red', 'green', 'blue']:
                    setattr(las, header, column)
                else:
                    las.add_extra_dim(laspy.ExtraBytesParams(name=header, type="f8"))
                    setattr(las, header, column)
        las.write(filename)
        if not verbose:
            print("Saved.")

    elif filename.endswith('.csv'):
        pd.DataFrame(pointcloud).to_csv(filename, header=None, index=None, sep=' ')
        if verbose: print("Saved to:", filename)

    elif filename.endswith('.ply'):

        if not isinstance(pointcloud, pd.DataFrame):
            cols = list(set(cols))
            pointcloud = pd.DataFrame(pointcloud, columns=cols)
        
        write_ply(filename, pointcloud[cols])
        if verbose: print("Saved to:", filename)