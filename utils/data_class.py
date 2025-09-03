import numpy as np
from typing import Dict
from scipy.spatial.transform import Rotation as R
from pathlib import Path


class Calib(object):
    NUM2CAM={
        1: 'front0',
        2: 'front1',
        3: 'right0',
        4: 'right1',
        5: 'rear0',
        6: 'rear1',
        7: 'left0',
        8: 'left1',
    }
    def __init__(self, 
                 img_size,
                 intrinsic:np.ndarray,
                 distortion:np.ndarray,
                 ldr2cam:Dict[str, np.ndarray],
                 cam_tag = None):
        '''
        img_size: (width, height)
        ldr2cam: {
            rot:[yaw, pitch, roll],
            trans:[tx, ty, tz]
        }'''
        self.cam_tag = cam_tag
        self.img_size = img_size
        self.intrinsic = intrinsic
        self.distortion = distortion
        self.ldr2cam = ldr2cam
        Rotation = R.from_euler('zyx', self.ldr2cam['rot'], degrees=True)
        # T_ldr2cam: (3, 4)
        self.T_ldr2cam = np.concatenate([Rotation.as_matrix(), self.ldr2cam['trans'].reshape(-1, 1)], dtype=np.float64, axis=1) 
        
    @classmethod
    def deserialize(cls, data):
        intrinsic = np.array([
            [data['fx'], 0.0, data['px']],
            [0.0, data['fy'], data['py']],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        distortion = np.array([
            data['k1'], data['k2'], data['k3'], \
            data['k4'], data['k5']
        ], dtype=np.float64).reshape((-1,1))
        ldr2cam = dict(
            rot= np.array([
                data['yaw_ldr2cam'], data['pitch_ldr2cam'], data['roll_ldr2cam']
            ], dtype=np.float64),
            trans= np.array([
                data['x_ldr2cam'], data['y_ldr2cam'], data['z_ldr2cam']
                ], dtype=np.float64)
        )
        if 'cam_number' in data:
            cam_tag = cls.NUM2CAM[data['cam_number']]
        else:
            cam_tag = data.get('cam_tag', None)
        
        return cls(
            img_size=(  data.get('img_size_w', 1280),   # (img_size_w, img_size_h) only exist in cam_calib/common
                        data.get('img_size_h', 720)),
            intrinsic =  intrinsic,
            distortion = distortion,
            ldr2cam = ldr2cam,
            cam_tag = cam_tag
        ) 
    @classmethod
    def from_file(cls, file_path:str):
        import yaml
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        if 'cam_number' not in data:
            data['cam_tag'] = Path(file_path).stem.split('_')[1]
        
        return cls.deserialize(data)

    def __repr__(self):
        return f"Calib(cam_tag={self.cam_tag})"