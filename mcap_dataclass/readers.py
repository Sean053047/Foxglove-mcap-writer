from abc import ABC, abstractmethod
# import numpy as np
import os
import os.path as osp
import cv2
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path
from pypcd4 import PointCloud
from typing import List
from builtin_interfaces.msg import Time
from scipy.spatial.transform import Rotation
from utils.radarocc_tool import get_occ_grid_info
from utils.kradar_tool import (
    ID2COLOR,
    load_label, 
    deserialize_list_tuple_objs, 
)
from utils.data_class import Calib
IMAGE_SUFFIXES = ['.png', '.jpg', '.bmp']

OPTIONS=dict(
    OCCReader = ('occ_rae', 'occ_xyz'),
    PCDReader = ('rpc'),
    BOX3DReader = ('kradar'),
    CameraReader = ('kradar'),
    CALIBReader = ('kradar'),
    TFStaticReader = ()
)

def from_file(fpth:str):
    suffix = osp.basename(fpth).split('.')[-1]
    with open(fpth, 'r') as file:
        if suffix == 'json':
            data = json.load(file)
        elif suffix in ('yaml', 'yml'):
            data = yaml.safe_load(file)
        elif suffix == 'txt':
            data = [line.strip() for line in file.readlines()]
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    return data
class BaseReader:
    def __init__(self, data_dir:str,  suffix: str, fstem2time:str=None, option=None) -> None:
        self.data_dir = data_dir
        # Initialize suffix related
        self.suffix: str = suffix
        self.fstem2time = fstem2time
        if fstem2time is not None:
            self.fstem2time = from_file(fstem2time)
        self.option = option            
    def __repr__(self):
        return f"{self.__class__.__name__}(data_dir={self.data_dir}, option={self.option})"
    @property
    def all_files(self):
        # ? Check whether the file is symlink
        return sorted( [f.resolve() for f in Path(self.data_dir).iterdir() if f.suffix == self.suffix])
    @abstractmethod
    def load_data(self, fpth: str):
        raise NotImplementedError("load_data do not be implemented.")

    def get_Time(self, stamp:str) -> Time:
        stamp = stamp if self.fstem2time is None else self.fstem2time[stamp]
        if '.' in stamp:
            sec, nsec = map( int , stamp.split('.'))
        else:
            sec, nsec = map( int , (stamp[:-9], stamp[-9:]))
        return Time(sec=sec, nanosec=nsec)
    
    @classmethod
    def deserialize(cls, content:dict):
        return cls(data_dir = content['data_dir'], 
                   suffix = content['suffix'],
                   fstem2time = content.get('fstem2time', None),
                   option = content.get('option', None))
    # Todo: Need to check pcd format. Add use for feather files
class OCCReader(BaseReader):
    def __init__(self, data_dir:str, suffix: str, fstem2time:str=None, option=None ) -> None:
        super().__init__(data_dir, suffix, fstem2time)
        self.occ_info_arr = get_occ_grid_info(XYZ=True)
        self.option = option
    def load_data(self, fpth:str):
        '''format: R, A, E'''
        if self.option == 'occ_rae':
            occ = np.load(fpth, allow_pickle=True)
            semantic = occ[6, :]
            color = np.array([ID2COLOR[ss] for ss in semantic], dtype=np.float32)
            r, a, e = np.split(occ[:3, ...].T, 3, axis=-1)
            x = r * np.cos(np.radians(e)) * np.cos(np.radians(a))
            y = r * np.cos(np.radians(e)) * np.sin(np.radians(a))
            z = r * np.sin(np.radians(e))
            occ_pc_arr = np.concatenate([
                x, y, z, color
            ], axis=1)
            fields = ['x', 'y', 'z','red', 'green', 'blue', 'alpha']
            types = [np.float32]*7
            pc = PointCloud.from_points(occ_pc_arr, fields=fields, types=types)
        elif self.option == 'occ_xyz':
            occ = np.load(fpth, allow_pickle=True)
            semantic = occ[6, :]
            color = np.array([ID2COLOR[ss] for ss in semantic], dtype=np.float32)
            occ_pc_arr = np.concatenate([
                occ[:3, :].T, color
            ], axis=1)
            fields = ['x', 'y', 'z','red', 'green', 'blue', 'alpha']
            types = [np.float32]*7
            pc = PointCloud.from_points(occ_pc_arr, fields=fields, types=types)
        else:   
            occ = np .load(fpth, allow_pickle=True)
            idx_r, idx_a, idx_e = (np.where(occ>0))
            semantic = occ[idx_r, idx_a, idx_e]
            color = np.array([ID2COLOR[ss] for ss in semantic], dtype=np.float32)
            occ_pc_arr = np.concatenate(
                [self.occ_info_arr[idx_r, idx_a, idx_e], color], axis=-1)
            fields = ['x', 'y', 'z','red', 'green', 'blue', 'alpha']
            types = [np.float32]*7
            pc = PointCloud.from_points(occ_pc_arr, fields=fields, types=types)
        return pc

class PCDReader(BaseReader):
    def load_data(self, fpth:str):
        if self.suffix == '.pcd':
            pc = PointCloud.from_path(fpth)
        elif self.suffix == '.npy':
            if self.option == 'rpc':
                pc_arr = np.load(fpth, allow_pickle=True)
                range_mask = pc_arr[:, 5] < 100.0 # ! Temporary added
                pc_arr = pc_arr[range_mask, :]
                
                pw = pc_arr[:, 3:4]
                r = pc_arr[:, 5:6] 
                approx_rcs = r**4 * pw 
                points = np.concatenate([pc_arr[:, :5], np.log1p(pc_arr[:, 3:4]), approx_rcs], axis=1)
                fields = ['x', 'y', 'z', 'pw', 'dop', 'logpw', 'approx_rcs']
                types = [np.float32] * 7
                pc = PointCloud.from_points(points, fields=fields, types=types)
            else:
                # ? This is suitable for RadarOcc .npy file.            
                fields = ['x', 'y', 'z', 'intensity', 'semantic']
                types = [np.float32, np.float32, np.float32, np.float32, np.int32]
                pc_arr = np.load(fpth, allow_pickle=True).T
                if pc_arr.shape[1] < 5:
                    fields.pop(-1)
                    types.pop(-1)
                pc = PointCloud.from_points(pc_arr, fields=fields, types=types)
        else:
            NotImplementedError()
        return pc
class Box3DReader(BaseReader):    
    '''Support option: kradar.v1_0, kradar.v2_0, kradar.v2_1'''
    def load_data(self, fpth:str):
        if 'kradar' in self.option:
            version = self.option.strip().split('.')[-1]
            assert version in ('v1_0', 'v2_0', 'v2_1'), 'Forget to set the version for kradar box reader.'
            _, data = load_label(version, file_path=fpth)
            data = deserialize_list_tuple_objs(data)
        else:
            data = from_file(fpth)
        return data

class CameraReader(BaseReader):
    def load_data(self, fpth:str):
        if self.option == 'kradar':
            image = cv2.imread(str(fpth))
            return image[:, :1280, :]
        else:
            return cv2.imread(str(fpth))
    @property
    def im_shape(self):
        return cv2.imread(str(self.all_files[0])).shape

# Below haven't done yet
class CALIBReader:
    def __init__(self, fpth:str, option:None) -> None:
        self.fpth = fpth
        self.option = option
    
    def load_data(self):
        if self.option == 'kradar':
            calib = Calib.from_file(self.fpth)
            im_w, im_h = calib.img_size
            intrinsic = calib.intrinsic
            distortion = calib.distortion
            projection = np.concatenate([
                calib.intrinsic, np.zeros((3,1))
            ], axis=-1)
            matrices = dict(
                image_width=im_w,
                image_height=im_h,
                intrinsic=intrinsic.flatten().tolist(),
                projection_matrix=projection.flatten().tolist(),
                distortion_model='plumb_bob',
                distortion= distortion.flatten().tolist()
            )
            return matrices
        else:
            return from_file(self.fpth)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(data_dir={self.fpth}, option={self.option})"
    @classmethod
    def deserialize(cls, content:dict):
        return cls(
            fpth = content['fpth'],
            option = content.get('option', None)
        )

class TFStaticReader:
    def __init__(self, fpths:List[str], col_major:bool, option:None) -> None:
        self.fpths = fpths
        self.col_major = col_major
        self.option = option
    def load_data(self):
        tf_list = []
        for fpth in self.fpths:
            extrinsic = from_file(fpth)['extrinsic']
            tf = np.array(extrinsic, dtype=np.float64).reshape((4,4))
            if self.col_major: 
                tf = tf.T
            tf_list.append(tf)
        return tf_list
    def __repr__(self):
        return f"{self.__class__.__name__}(data_dir={self.fpth}, option={self.option})"
    @classmethod
    def deserialize(cls, content:dict):
        fpths = content['fpths']
        if type(fpths) is str:
            fpths = [fpths]
        return cls(
            fpths = fpths,
            col_major = content.get('col_major', False),
            option = content.get('option', None)
        )

# ? Currently, unused
class Box2DReader(BaseReader):    
    def load_data(self, fpth:str):
        return from_file(fpth)

# ? Currently, unused.
class TimePosesReader(BaseReader):
    def __init__(self,
                    fpth:str, suffix:str, option='kradar') -> None:
        self.fpth = fpth
        self.suffix = suffix
    def load_data(self):
        '''Pose: (w, x, y, z) for rotation, (x, y, z) for translation'''
        items = from_file(self.fpth)
        get_Time = lambda stamp: Time(sec=int(stamp[:-9]), nanosec=int(stamp[-9:]))
        timestamps = list()
        poses = list()
        for item in items:
            timestamps.append(get_Time(item['timestamp']))
            pose = dict(
                translation = item['translation'],
                rotation= item['rotation']
            )
            poses.append(pose)
        return timestamps, poses
    @classmethod
    def deserialize(cls, content:dict):
        return cls(
            fpth = content['fpth'],
            suffix = content.get('suffix', '.json')
        )