import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from abc import abstractmethod
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import rosbag2_py
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Quaternion, Pose, Vector3
from foxglove_msgs.msg import (
    PointCloud,
    PackedElementField,
    CompressedImage,
    ImageAnnotations,
    TextAnnotation,
    PointsAnnotation,
    Point2,
    Color,
    CubePrimitive,
    TextPrimitive,
    CameraCalibration,
    SceneUpdate,
    SceneEntity,
    FrameTransforms,
    FrameTransform, 
    PoseInFrame,
    PosesInFrame
)

from utils.utils import ( 
    CLASS2COLOR,
)
'''x:forward, y:left, z:up'''
class ROSBAGWRITER:
    def __init__(self, frame_id, schema, topic) -> None:
        self.frame_id = frame_id
        self.schema = schema
        self.topic = topic
        
    def init_writer(self, writer):
        writer.create_topic(
            rosbag2_py.TopicMetadata(
                name=str(self.topic), type=str(self.schema), serialization_format="cdr"
            )
        )
        return writer

    @abstractmethod
    def write(self, writer, stamp, *args, **kwargs):
        raise NotImplementedError("Not implement write function.")
    
    def write2bag(self, writer, msg, stamp: Time):
        NUM_SEC_DIGIT = 8
        st_indx = max(len(str(stamp.sec))-NUM_SEC_DIGIT, 0)         
        timestamp = int(int(str(stamp.sec)[st_indx:]) * 1e9 + stamp.nanosec)
        writer.write(self.topic, serialize_message(msg), timestamp)
    
    @classmethod
    def deserialize(cls, content:dict):
        return cls(
            frame_id = content['frame_id'],
            schema = content['schema'],
            topic = content['topic']
        )

class PCDWriter(ROSBAGWRITER):
    NP2ROSTYPE = {
        np.int8: 1,  # INT8
        np.uint8: 2,  # UINT8
        np.int16: 3,
        np.uint16: 4,
        np.int32: 5,
        np.uint32: 6,
        np.float32: 7,
        np.float64: 8,
    }
    
    def write(self, mcap_writer, stamp, pc):
        msg = PointCloud(
                timestamp = stamp,
                frame_id = self.frame_id,
                pose=Pose(  position=Point(x=0.0, y=0.0, z=0.0),
                            orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
                )
        )
        lls = ['x', 'y', 'z', 'r','g','b','a']
        fields = [ ll for ll in pc.fields if ll in lls]
        types = [pc.types[i] for i, ll in enumerate(pc.fields) if ll in lls]
        pc_data = pc.numpy(fields)
        N = pc_data.shape[0]
        # Todo: Make it more generalized
        msg.point_stride = pc_data.itemsize * len(fields)
        msg.fields = [
            PackedElementField(name="x", offset=0, type=7),
            PackedElementField(name="y", offset=4, type=7),
            PackedElementField(name="z", offset=8, type=7),
        ]+  ([
            PackedElementField(name="red", offset=12, type=7),
            PackedElementField(name="green", offset=16, type=7),
            PackedElementField(name="blue", offset=20, type=7),
        ] if {'r', 'g', 'b'} <= set(fields) else []
        ) + ([
            PackedElementField(name="alpha", offset=24, type=7),
        ] if 'a' in fields else [])
        msg.data = pc_data.tobytes()
        self.write2bag(mcap_writer, msg, stamp)

    @classmethod
    def deserialize(cls, content: dict):
        return cls(
            frame_id=content.get('frame_id', "/pcd"),
            schema=content.get('schema', "foxglove_msgs/msg/PointCloud"),
            topic=content.get('topic', "/pointcloud"),
        )

class Box3DWriter(ROSBAGWRITER):
    
    def __init__(self, 
                 frame_id = '/pcd', 
                 schema = 'foxglove_msgs/msg/SceneUpdate', 
                 topic = '/box3d',
                 color_offset = [0, 0, 0]
                 ) -> None:
        super().__init__(frame_id, schema, topic)
        self.color_offset = color_offset
    def write(self, writer, stamp, boxes3d):
        
            msg = SceneUpdate()
            scene_entity = SceneEntity(
                timestamp = stamp,
                frame_id = self.frame_id,
                id = self.topic,
            )
            for box3d in boxes3d:
                # Something weird
                cls:str = box3d['class_name']
                position:list = list(map(float, box3d['translation'])) # x, y, z
                quat:list = box3d['rotation'] # rw, rx, ry, rz
                
                size = box3d['size']
                r,g,b = map(float,CLASS2COLOR[cls])
                r = max(min(r + self.color_offset[0], 1.0), 0.0)
                g = max(min(g + self.color_offset[1], 1.0), 0.0)
                b = max(min(b + self.color_offset[2], 1.0), 0.0)
                cube = CubePrimitive(
                    pose  = Pose(    position = Point(x=position[0], y=position[1], z=position[2]),
                                orientation = Quaternion(x = quat[1], y =quat[2], z=quat[3], w=quat[0])
                            ),
                    size  = Vector3(x=size[0], y=size[1], z=size[2]), 
                    color = Color(r=r, g=g, b=b, a=0.5)
                )
                scene_entity.cubes.append(cube)
                # Need to check?
                scene_entity.texts.append(
                    TextPrimitive(
                        pose=Pose(    position = Point(x=position[0], y=position[1], z=position[2]),
                                orientation = Quaternion(x = quat[1], y =quat[2], z=quat[3], w=quat[0])
                            ),
                        font_size=100.0,
                        scale_invariant = False,
                        billboard = False,
                        text = f"class {cls}"
                    )
                )
            msg.entities.append(scene_entity)
            self.write2bag(writer, msg, stamp)
    
    @classmethod
    def deserialize(cls, content: dict):
        return cls( frame_id=content.get('frame_id', '/pcd'),  
                    schema = content.get('schema', 'foxglove_msgs/msg/SceneUpdate'), 
                    topic = content.get('topic', '/box3d'),
                    color_offset = content.get('color_offset', (0,0,0)))

class CameraWriter(ROSBAGWRITER):
    
    def __init__(
        self,
        frame_id="/camera",
        schema="foxglove_msgs/msg/CompressedImage",
        topic="/image",
        suffix = '.png',
    ) -> None:
        super().__init__(frame_id, schema, topic)
        self.suffix = suffix

    def write(self, mcap_writer, stamp, img):
        ret, buffer = cv2.imencode(self.suffix, img)
        assert ret, "Failed to encode image."
        msg = CompressedImage(
            timestamp=stamp,
            frame_id=self.frame_id,
            data=np.array(buffer).tobytes(),
            format=self.suffix.strip("."),
        )
        self.write2bag(mcap_writer, msg, stamp)

    @classmethod
    def deserialize(cls, content:dict):
        return cls(
            frame_id=content.get('frame_id', "/camera"),
            schema=content.get('schema', "foxglove_msgs/msg/CompressedImage"),
            topic=content.get('topic', "/image"),
            suffix = content.get('suffix', '.png')
        )

# Todo: Need to refine
class Box2DWriter(ROSBAGWRITER):
    def __init__(self, 
                 frame_id="/camera", 
                 schema="foxglove_msgs/msg/ImageAnnotations", 
                 topic="/box2d",
                 ) -> None:
        super().__init__(frame_id, schema, topic)
    
    def write(self, writer):
        
        for (_, cls_bboxes), stamp in zip(
            self.data_reader.load_data(self.data_dir),
            self.data_reader.load_timestamps(self.fname2stamp)
        ):
            msg = ImageAnnotations()
            for cls, bbox in cls_bboxes:
                vertices = bbox
                msg.points.append( 
                    PointsAnnotation(
                        timestamp= stamp, 
                        type = PointsAnnotation.LINE_LOOP,
                        points = [Point2(x= float(v[0]), y= float(v[1])) for v in vertices],
                        outline_color = Color(
                            r = CLASS2COLOR[cls][0],
                            g = CLASS2COLOR[cls][1],
                            b = CLASS2COLOR[cls][2],
                            a = CLASS2COLOR[cls][3],
                        ),
                        thickness = 3.0,
                    )
                )
                msg.texts.append( 
                    TextAnnotation(
                        timestamp = stamp, 
                        position = Point2(x=float(vertices[0][0]), y = float(vertices[0][1])),
                        text = cls, 
                        font_size = 30.0,
                        text_color = Color(
                            r = CLASS2COLOR[cls][0],
                            g = CLASS2COLOR[cls][1],
                            b = CLASS2COLOR[cls][2],
                            a = 1.0,
                        )
                    )
                )
            self.write2bag(writer, msg, stamp)
    
    @classmethod
    def deserialize(cls, content: dict):
        return cls(
            frame_id=content.get('frame_id', "/camera"),
            schema=content.get('schema', "foxglove_msgs/msg/ImageAnnotations"),
            topic=content.get('topic', "/box2d"),
        )

class CALIBWriter(ROSBAGWRITER):
    def __init__(self, 
                 frame_id = "/camera", 
                 schema = "foxglove_msgs/msg/CameraCalibration", 
                 topic = "/camera_info",
                ) -> None:
        super().__init__(frame_id, schema, topic)
        
    def write(self, writer, stamp, matrices):
        msg = CameraCalibration(
            timestamp = stamp,
            frame_id = self.frame_id,
            height = int(matrices['image_height']),
            width = int(matrices['image_width']),
            k = matrices['intrinsic'],
            p = matrices['projection_matrix']
        )
        self.write2bag(writer, msg, stamp)
        
    @classmethod
    def deserialize(cls, content: dict):
        return cls(
            parent_frame_id=content.get('parent_frame_id', "/camera"),
            schema=content.get('schema', "foxglove_msgs/msg/CameraCalibration"),
            topic=content.get('topic', "/camera_info"),
        )

class TFStaticWriter(ROSBAGWRITER):
    def __init__(self, 
                 schema = 'foxglove_msgs/msg/FrameTransforms', 
                 parent_frame_id =None,
                 child_frame_id = None,
                 topic = '/tf',
                 ) -> None:
        super().__init__(None, schema, topic)
        del self.frame_id
        self.parent_frame_id = parent_frame_id
        self.child_frame_id = child_frame_id
        
    def write(self, writer, stamp, stf):
            msg = FrameTransforms()
            translate, quat = stf[:3, 3].flatten(), Rotation.from_matrix(stf[:3,:3]).as_quat()
            msg.transforms.append(
                FrameTransform(
                    timestamp = stamp,
                    parent_frame_id = self.parent_frame_id,
                    child_frame_id = self.child_frame_id,
                    translation= Vector3(x= translate[0], y=translate[1], z=translate[2]),
                    rotation = Quaternion(x = quat[0], y =quat[1], z=quat[2], w=quat[3]), 
                )
            )
            self.write2bag(writer, msg, stamp)
            
    @classmethod
    def deserialize(cls, content: dict):
        return cls(
            schema=content.get('schema', "foxglove_msgs/msg/PosesIN"),
            topic=content.get('topic', "/tf"),
            parent_frame_id=content.get('parent_frame_id', None),
            child_frame_id=content.get('child_frame_id', None),
        )

class TimePosesWriter(ROSBAGWRITER):
    def __init__(self, 
                 schema = 'foxglove_msgs/msg/PoseInFrame', 
                 frame_id ='base_link',
                 topic = '/tpf',
                 ) -> None:
        super().__init__(frame_id, schema, topic)
        
    def write(self, writer, stamps, tps):
        '''rotation: (w, x, y, z) for quaternion, translation: (x, y, z)'''
        for stamp, time_pose in zip(stamps, tps):
            msg = PoseInFrame()
            msg.timestamp = stamp
            msg.frame_id = self.frame_id
            
            translation = time_pose['translation']
            quat = time_pose['rotation']
            msg.pose = Pose(
                position=Point(x=translation[0], y=translation[1], z=translation[2]),
                orientation=Quaternion(w=float(quat[0]), x=float(quat[1]), y=float(quat[2]), z=float(quat[3]))
            )
            self.write2bag(writer, msg, stamp)
            
    @classmethod
    def deserialize(cls, content: dict):
        return cls(
            schema=content.get('schema', "foxglove_msgs/msg/PoseInFrame"),
            frame_id=content.get('frame_id', 'base_link'),
            topic=content.get('topic', "/tpf"),
        )