from abc import ABC, abstractmethod
# import numpy as np
import cv2
import yaml
import json
import numpy as np
from pathlib import Path
from pypcd4 import PointCloud
from builtin_interfaces.msg import Time
from scipy.spatial.transform import Rotation
IMAGE_SUFFIXES = ['.png', '.jpg', '.bmp']

def from_json(fpth:str):
    with open(fpth, 'r') as file:
        data = json.load(file)
    return data

def from_yaml(fpth:str):
        with open(fpth, 'r') as file:
            data = yaml.safe_load(file)
        return data
        
def from_txt(fpth:str):
        with open(fpth, 'r') as file:
            data = [ line.strip() for line in file.readlines()]
        return data

class BaseReader:
    def __init__(self, data_dir:str,  suffix: str, fstem2time:str=None) -> None:
        self.data_dir = data_dir
        # Initialize suffix related
        self.suffix: str = suffix
        self.fstem2time = fstem2time
        if fstem2time is not None:
            fsuffix = Path(fstem2time).suffix.lstrip('.')
            self.fstem2time = globals().get(f"from_{fsuffix}")(fstem2time)
            
    @property
    def all_files(self):
        return sorted( [f for f in Path(self.data_dir).iterdir() if f.suffix == self.suffix])
    
    @abstractmethod
    def load_data(self, fpth: str):
        raise NotImplementedError("load_data do not be implemented.")

    def get_Time(self, stamp:str) -> Time:
        stamp = stamp if self.fstem2time is None else self.fstem2time[stamp]
        sec, nsec = map( int , (stamp[:-9], stamp[-9:]))
        return Time(sec=sec, nanosec=nsec)
    
    @classmethod
    def deserialize(cls, content:dict):
        return cls(data_dir = content['data_dir'], 
                   suffix = content['suffix'],
                   fstem2time = content.get('fstem2time', None))
    
# Todo: Need to check pcd format. Add use for feather files
class PCDReader(BaseReader):
    def load_data(self, fpth:str):
        if self.suffix == '.pcd':
            pc = PointCloud.from_path(fpth)
            return pc
        else:
            NotImplementedError()
    
class Box3DReader(BaseReader):
    def __init__(self, data_dir, suffix, fstem2time = None):
        super().__init__(data_dir, suffix, fstem2time)
        
    def load_data(self, fpth:str):
        if self.suffix == '.json':
            return from_json(fpth)
        elif self.suffix == '.yaml':
            return from_yaml(fpth)
        elif self.suffix == '.text':
            return from_txt(fpth)
        else:
            NotImplementedError()
    
class CameraReader(BaseReader):
    def load_data(self, fpth:str):
        return cv2.imread(str(fpth))
    
    @property
    def im_shape(self):
        return cv2.imread(str(self.all_files[0])).shape

# Todo: Make sure the format for text file.
class Box2DReader(BaseReader):    
    def load_data(self, fpth:str):
        return from_txt(fpth)
# Below haven't done yet
class CALIBReader:
    def __init__(self, fpth:str, suffix:str) -> None:
        self.fpth = fpth
        self.suffix = suffix 
        
    def load_data(self):
        return  from_json(self.fpth)
        
    @classmethod
    def deserialize(cls, content:dict):
        return cls(
            fpth = content['fpth'],
            suffix = content.get('suffix', '.json')
        )
        
class StaticTFReader:
    def __init__(self, fpth:str, suffix: str, row_major:bool) -> None:
        self.fpth = fpth
        self.suffix = suffix
        self.row_major = row_major
        
    def load_data(self):
        tf = np.array(from_json(self.fpth), dtype=np.float64).reshape((4,4))
        if self.row_major:
            return tf
        else:
            return tf.T
        
    @classmethod
    def deserialize(cls, content:dict):
        return cls(
            fpth = content['fpth'],
            suffix = content.get('suffix', '.json'),
            row_major = content.get('row_major')
        )

class TimePosesReader(BaseReader):
    def __init__(self, fpth:str, suffix:str,) -> None:
        self.fpth = fpth
        self.suffix = suffix
    def load_data(self):
        '''Pose: (w, x, y, z) for rotation, (x, y, z) for translation'''
        items = from_json(self.fpth)
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