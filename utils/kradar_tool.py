import numpy as np
from scipy.spatial.transform import Rotation

CLS_NAME2ID = {
    'Empty':0,
    'Static':1,
    'Sedan':2,
    'Bus or Truck':3,
    'Motorcycle':4,
    'Bicycle':5,
    'Pedestrian':6,
    'Pedestrian Group':7,
    'Bicycle Group':8,
}
ID2COLOR = {
            0:[0., 0., 0., 0.],
            1:[0., 0.7, 0, 1.0],
            2:[0.8, 0.3, 0.5, 1.0],
            3:[0.95, 0.6, 0.3, 1.0],
            4:[0.5, 0, 0, 1.0],
            5:[0., 0, 0.5, 1.0],
            6:[0.2, 0.2, 0.2, 1.0],
            7:[0.9, 0.5, 0.6, 1.0],
            8:[0.25, 0.64, 1.0, 1.0],
        }
CLASS2COLOR = {CLS_NAME: ID2COLOR[ID] for CLS_NAME, ID in CLS_NAME2ID.items()}          
def get_label(version, file_path, calib:bool=False):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    list_tuple_objs = []
    header = (lines[0]).rstrip('\n')
    try:
        temp_idx, tstamp = header.split(', ')   
    except: # ? line breaking error for v2.0
        _, header_prime, line0 = header.split('*')
        header = '*' + header_prime
        temp_idx, tstamp = header.split(', ')
        lines.insert(1, '*'+line0)
        lines[0] = header
    rdr, ldr64, camf, ldr128, camr = temp_idx.split('=')[1].split('_')
    tstamp = tstamp.split('=')[1]
    dict_idx = dict(rdr=rdr, ldr64=ldr64, camf=camf,\
                    ldr128=ldr128, camr=camr, tstamp=tstamp)
    if version == 'v1_0':
        for line in lines[1:]:
            # print(line)
            list_vals = line.rstrip('\n').split(', ')
            if len(list_vals) != 11:
                print('* split err in ', file_path)
                continue
            idx_p = int(list_vals[1])
            idx_b4 = int(list_vals[2])
            cls_name = list_vals[3]
            x = float(list_vals[4])
            y = float(list_vals[5])
            z = float(list_vals[6])
            th = np.radians(float(list_vals[7]))
            l = 2*float(list_vals[8])
            w = 2*float(list_vals[9])
            h = 2*float(list_vals[10])
            list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p, idx_b4), 'R'))
    elif version == 'v2_0':
        for line in lines[1:]:
            # print(line)
            list_vals = line.rstrip('\n').split(', ')
            idx_p = int(list_vals[1])
            cls_name = (list_vals[2])
            x = float(list_vals[3])
            y = float(list_vals[4])
            z = float(list_vals[5])
            th = np.radians(float(list_vals[6]))
            l = 2*float(list_vals[7])
            w = 2*float(list_vals[8])
            h = 2*float(list_vals[9])
            list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), 'R'))
    elif version == 'v2_1':
        for line in lines[1:]:
            # print(line)
            list_vals = line.rstrip('\n').split(', ')
            avail = list_vals[1]
            idx_p = int(list_vals[2])
            cls_name = (list_vals[3])
            x = float(list_vals[4])
            y = float(list_vals[5])
            z = float(list_vals[6])
            th = np.radians(float(list_vals[7]))
            l = 2*float(list_vals[8])
            w = 2*float(list_vals[9])
            h = 2*float(list_vals[10])
            list_tuple_objs.append((cls_name, (x, y, z, th, l, w, h), (idx_p), avail))
    
    if calib:
        list_temp = []
        dx, dy, dz = -2.54, 0.3, 0.7 # ! Check that all clips have same calibration between radar & lidar.
        for obj in list_tuple_objs:
            cls_name, (x, y, z, th, l, w, h), trk, avail = obj
            x = x + dx
            y = y + dy
            z = z + dz
            list_temp.append((cls_name, (x, y, z, th, l, w, h), trk, avail))
        list_tuple_objs = list_temp
    
    return dict_idx, list_tuple_objs

def deserialize_list_tuple_objs(list_tuple_objs:list):
    '''Convert list of tuples to a dictionary with class names as keys.'''
    dict_labels = []
    for cls_name, (x, y, z, th, l, w, h), trk, avail in list_tuple_objs:
        dict_labels.append({
            'class_name': cls_name,
            'translation': (x, y, z),
            'rotation': Rotation.from_euler('z', th).as_quat(scalar_first=True), # w, x, y, z
            'size': (l, w, h),
            'track_id': trk,
            'availability': avail
        })
    return dict_labels