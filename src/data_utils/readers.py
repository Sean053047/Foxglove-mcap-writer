from abc import ABC, abstractmethod
# import numpy as np
import cv2
import numpy as np
from pathlib import Path
from pypcd4 import PointCloud
from builtin_interfaces.msg import Time
from scipy.spatial.transform import Rotation
class BaseReader(ABC):
    def __init__(self, suffix: str) -> None:
        self.suffix: str = suffix
        
    def load_timestamps(self, fpth: str):
        '''mode: 'ros_time | 'int' '''
        with open(fpth, "r") as txt_file:
            for stamp_str in txt_file.readlines():
                if '.' in stamp_str.strip():
                    integer_part, fractional_part = stamp_str.strip().split('.')
                else:
                    integer_part, fractional_part = stamp_str.strip(), ""
                fractional_part = fractional_part + (9- len(fractional_part)) * '0'
                sec = int(integer_part)
                nanosec = int(fractional_part)
                yield Time(sec=sec, nanosec=nanosec)
                
    @abstractmethod
    def load_data(self, dir: str):
        raise NotImplementedError("load_data do not be implemented.")

class PCDReader(BaseReader):
    def __init__(self, suffix: str = ".pcd") -> None:
        super().__init__(suffix)

    def load_data(self, dir: str):
        fnames = sorted(
            [
                fname
                for fname in Path(dir).iterdir()
                if fname.suffix == self.suffix
            ]
        )
        for fname in fnames:
            pc = PointCloud.from_path(fname)
            pc_info = {
                'fields': pc.fields,
                'types': pc.types
            }
            pc_data = pc.numpy(pc.fields)
            yield fname, pc_info, pc_data
        
class IMGReader(BaseReader):
    def __init__(self, suffix: str = ".png") -> None:
        super().__init__(suffix)
        
    def load_data(self, dir: str):
        fnames = sorted(
            [
                fname
                for fname in Path(dir).iterdir()
                if fname.suffix == self.suffix
            ]
        )
        for fname in fnames:
            yield fname, cv2.imread(str(fname))

class CALIBReader(BaseReader):
    def __init__(self, suffix:str, *args, mode='elan', **kwargs) -> None:
        '''Optional arguments: for kitti, cam_tag:str'''
        super().__init__(suffix)
        if mode =='elan':
            self.get_calib_from_raw = self.__elan_get_calib
        elif mode =='kitti':
            self.get_calib_from_raw = self.__kitti_get_calib(*args, **kwargs)
    @staticmethod
    def __kitti_get_calib(*args, **kwargs):
        cam_tag = kwargs.get('cam_tag', 'P2')
        @staticmethod
        def __calib(data_path:str):
            nonlocal cam_tag
            with open(data_path, 'r') as file:
                raw_calib = file.readlines()
            calib_str = [line for line in raw_calib if cam_tag in line][0]
            cam_calib = np.array(calib_str.strip().split(' ')[1:], dtype=np.float64).reshape(3,4)
            cam_calib[:3, 3] = 0.0
            return cam_calib
            
        return __calib

    # Todo
    @staticmethod
    def __elan_get_calib(*args, **kwargs):
        ...
    
    def load_data(self, data_path:str):
        '''Return Projected matrix'''
        if Path(data_path).is_file():
            fnames = [Path(data_path)]
        elif Path(data_path).is_dir():
            fnames = sorted( [ fname for fname in Path(data_path).iterdir() if fname.suffix == self.suffix])
        else:
            raise AssertionError(f'Failure to load data_path, {data_path}')
        for fname in fnames:
            calib = self.get_calib_from_raw(fname)
            yield fname, calib
    
class TFReader(BaseReader):
    def __init__(self, suffix: str, *args, mode='elan', **kwargs,) -> None:
        '''
        optional argument:\n
        elan mode: key:str\n
        kitti mode: cam_tag:str\n
        '''
        super().__init__(suffix)
        if mode =='elan':
            self.get_tf_from_raw = self.__elan_get_tf(*args, **kwargs)
        elif mode == 'kitti':
            self.get_tf_from_raw = self.__kitti_get_tf(*args, **kwargs)
    @staticmethod
    def __elan_get_tf(*args, **kwargs):
        import json
        key = kwargs.get('key', 'extrinsic')
        def __tf(data_path):
            nonlocal key
            with open(data_path, 'r') as file:
                tf = json.load(file)
                return tf[key]
        return __tf
    
    @staticmethod
    def __kitti_get_tf(*args, **kwargs):
        cam_tag = kwargs.get('cam_tag', 'P2')
        def __tf(data_path):
            nonlocal cam_tag
            with open(data_path, 'r') as file:
                raw_calib = file.readlines()
            calib_str = [line for line in raw_calib if cam_tag in line][0]
            calib = np.array(calib_str.strip().split(' ')[1:], dtype=np.float64).reshape(-1,4)
            tf = np.identity(4)
            tf[:3, 3] = calib[:3, 3]                
            return tf
        return __tf
    
    def load_data(self, data_path:str):
        if Path(data_path).is_file():
            fnames = [Path(data_path)]
        elif Path(data_path).is_dir():
            fnames = sorted( [fname for fname in Path(data_path).iterdir() if fname.suffix == self.suffix])
        else:
            raise AssertionError(f'Failure to load data_path, {data_path}')
        
        for fname in fnames:
            tf = self.get_tf_from_raw(fname)
            
            yield fname, tf
    
class BOX2DReader(BaseReader):
    def __init__(self, suffix:str ='.txt', mode='elan', **kwargs) -> None:
        super().__init__(suffix)
        if mode == 'elan':
            self.get_box2d_from_raw = self.__elan_get_box2d(**kwargs)
    @staticmethod
    def __elan_get_box2d(*args, **kwargs):
        order = kwargs.get('order', 'xy')
        @staticmethod
        def __box2d(*args, **kwargs):
            nonlocal order
            bbox = kwargs.get('box2d', args[0])
            cols = np.sort(bbox[ [0,2]])
            rows = np.sort(bbox[ [1,3]])
            if order == 'xy':
                vertices = [ [c, r] for c in cols for r in rows]
            elif order == 'yx':
                vertices = [ [r, c] for c in cols for r in rows ]
            vertices[3], vertices[2] = vertices[2], vertices[3]
            return vertices
        
    def load_data(self, dir:str, RAW=False):
        fnames = sorted(
            [
                fname for fname in Path(dir).iterdir()
                if fname.suffix == self.suffix
            ]
        )
        for fname in fnames:
            cls_bboxes = list()
            with open(fname, 'r') as f:
                for line in f.readlines():
                    elements = line.strip().split(' ')
                    cls = elements[0]
                    bbox = np.array( elements[1:], dtype=np.float64)
                    if not RAW: 
                        bbox = self.get_box2d_from_raw(bbox)
                    cls_bboxes.append( (cls, bbox) )
                yield fname, cls_bboxes
            
class BOX3DReader(BaseReader):
    def __init__(self, suffix:str ='.txt', mode='elan') -> None:
        super().__init__(suffix)
        if mode == 'elan':
            self.get_box3d_from_raw = self.__elan_get_box3d 
        else:
            self.get_box3d_from_raw = self.__kitti_get_box3d 
            
    @ staticmethod
    def __kitti_get_box3d(*args, **kwargs):
        '''In camera coordinate'''
        box_info = kwargs.get('box3d', args[0])
        observation_angle, rotation_y = box_info[2], box_info[13]
        yaw = rotation_y + np.pi /2 
        if yaw > np.pi: yaw = yaw - 2*np.pi
        elif yaw < -np.pi: yaw = yaw + 2*np.pi
        obj_det = {
            'pos': box_info[10:13],
            'size':box_info[[8,7,9]], # width, Height, length 
            'quat': Rotation.from_euler('y', yaw).as_quat(),
        }
        return obj_det
    
    @ staticmethod
    def __elan_get_box3d(*args, **kwargs):
        ...
    def load_data(self, dir:str, RAW=False):
        fnames = sorted(
            [
                fname for fname in Path(dir).iterdir()
                if fname.suffix == self.suffix
            ]
        )
        for fname in fnames:
            cls_bboxes = list()
            with open(fname, 'r') as f:
                for line in f.readlines():
                    elements = line.strip().split(' ')
                    cls = elements[0]
                    bbox =  np.array(elements[1:], dtype=np.float64)
                    if not RAW:
                        bbox = self.get_box3d_from_raw(bbox)
                    cls_bboxes.append( (cls, bbox) )
                yield fname, cls_bboxes
            
if __name__ == "__main__":
    ...