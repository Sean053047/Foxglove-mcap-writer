import numpy as np 
import os.path as osp

def get_occ_grid_info(XYZ:bool=False) -> np.ndarray:
    project_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    occ_grid_info = np.load(osp.join(project_dir, 'resource', 'occ_grid_info.pkl'), allow_pickle=True)
    r, a, e = np.meshgrid(
        occ_grid_info['arrRange'], 
        occ_grid_info['arrAzimuth'],
        occ_grid_info['arrElevation'], indexing='ij'
    )
    a = np.radians(a)
    e = np.radians(e)
    if XYZ:
        x = r * np.cos(a) * np.cos(e)
        y = r * np.sin(a) * np.cos(e)
        z = r * np.sin(e)
        return np.stack([x, y, z], axis=-1)
    else:
        return np.stack([r, a, e], axis=-1)
