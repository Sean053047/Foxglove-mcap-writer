import json, yaml
import numpy as np 
from scipy.spatial.transform import Rotation
CLASS2COLOR = {
            'Cyclist': (1.0, 0.588,0.0),
            'Pedestrian':(0.7843, 0.13, 0.0),
            'Person':(0.6274, 0.1176, 0.0),
            'Car': (0.0, 1.0, 0.588),
            'Van': (0.0, 1.0, 0.392),
            'Vehicle': (1.0, 0.588,0.0),
            'Truck': (0.0, 1.0, 0.1961),
            'Tram': (0.0, 0.3922,1.0),
            'Misc': (0.0, 0.1961,1.0),
            'DontCare': (0.7843,0.7843,0.7843),
            'Motorcycle': (0.0, 1.0, 0.588),
            'ScooterRider': (0.0, 1.0, 0.588),
            'Bus': (.8, 0.3, 0.2)}
CLASS2COLOR.update( { k.lower():v for k,v in CLASS2COLOR.items()})

def get_color(cls):
    '''Return: [r, g, b] within range: [0, 1]'''
    global CLASS2COLOR
    
    if cls not in CLASS2COLOR:
        CLASS2COLOR[cls] = np.random.rand(3)
    r, g, b = map(float, CLASS2COLOR[cls])
    return r, g, b
        

def get_box_vertices( bbox, order='xy'):
    cols = np.sort(bbox[ [0,2]])
    rows = np.sort(bbox[ [1,3]])
    
    if order == 'xy':
        vertices = [ [c, r] for c in cols for r in rows]
    elif order == 'yx':
        vertices = [ [r, c] for c in cols for r in rows ]
    vertices[3], vertices[2] = vertices[2], vertices[3]
    return vertices

def get_quat_from_euler(roll, pitch, yaw):
    '''return:  [x,y,z,w]'''
    return Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()

def get_quat_from_matrix(R):
    '''return:  [x,y,z,w]'''
    return Rotation.from_matrix(R).as_quat()
