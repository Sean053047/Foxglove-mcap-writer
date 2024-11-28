import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
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
)


from data_utils.readers import (
    PCDReader,
    IMGReader,
    CALIBReader,
    TFReader,
    BOX2DReader,
    BOX3DReader,
)

class ROSBAG_WRITER:
    def __init__(self, data_dir, stamp_fpth, frame_id, schema, topic) -> None:
        self.data_dir = data_dir
        self.stamp_fpth = stamp_fpth
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
    def write(self, writer):
        raise NotImplementedError("Not implement write function.")
    
    def write2bag(self, writer, msg, stamp: Time):
        NUM_SEC_DIGIT = 8
        st_indx = max(len(str(stamp.sec))-NUM_SEC_DIGIT, 0)         
        timestamp = int(int(str(stamp.sec)[st_indx:]) * 1e9 + stamp.nanosec)
        writer.write(self.topic, serialize_message(msg), timestamp)
    @abstractmethod
    def deserialize(self, content:dict):
        raise NotImplementedError("Not implement deserialize()")
class PCDWriter(ROSBAG_WRITER):
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

    def __init__(
        self,
        data_dir: str,
        stamp_fpth: str,
        frame_id="/pcd",
        schema="foxglove_msgs/msg/PointCloud",
        topic="/pointcloud",
        suffix=".pcd",
    ):
        super().__init__(data_dir, stamp_fpth, frame_id, schema, topic)
        self.data_reader = PCDReader(suffix=suffix)

    def write(self, mcap_writer):
        for (fname, pc_info, pc_data), stamp in zip(
            self.data_reader.load_data(self.data_dir),
            self.data_reader.load_timestamps(self.stamp_fpth),
        ):
            msg = PointCloud()
            fields = pc_info["fields"]
            N = pc_data.shape[0]
            msg.timestamp = stamp
            msg.frame_id = self.frame_id
            msg.pose.position = Point(x=0.0, y=0.0, z=0.0)
            msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
            msg.point_stride = pc_data.itemsize * len(fields)
            msg.fields = [
                PackedElementField(name="x", offset=0, type=7),
                PackedElementField(name="y", offset=4, type=7),
                PackedElementField(name="z", offset=8, type=7),
            ]
            msg.data = pc_data.tobytes()
            self.write2bag(mcap_writer, msg, stamp)

    @classmethod
    def deserialize(cls, content: dict):
        return cls(
            data_dir=content['data_dir'],
            stamp_fpth=content['stamp_fpth'],
            frame_id=content.get('frame_id', "/pcd"),
            schema=content.get('schema', "foxglove_msgs/msg/PointCloud"),
            topic=content.get('topic', "/pointcloud"),
            suffix=content.get('suffix', ".pcd"),
        )
class IMGWriter(ROSBAG_WRITER):
    def __init__(
        self,
        data_dir: str,
        stamp_fpth: str,
        frame_id="/camera",
        schema="foxglove_msgs/msg/CompressedImage",
        topic="/image",
        suffix=".png",
    ) -> None:
        super().__init__(data_dir, stamp_fpth, frame_id, schema, topic)
        self.data_reader = IMGReader(suffix=suffix)

    @property
    def img_shape(self):
        for _, img in self.data_reader.load_data(self.data_dir):
            return img.shape

    def write(self, mcap_writer):
        for (_, img), stamp in zip(
            self.data_reader.load_data(self.data_dir),
            self.data_reader.load_timestamps(self.stamp_fpth),
        ):
            ret, buffer = cv2.imencode(self.data_reader.suffix, img)
            if not ret:
                self.get_logger().info(
                    f"[{self.get_name()}] Failed to encode image as {self.data_reader.suffix} format."
                )
            msg = CompressedImage(
                timestamp=stamp,
                frame_id=self.frame_id,
                data=np.array(buffer).tobytes(),
                format=self.data_reader.suffix.strip("."),
            )
            self.write2bag(mcap_writer, msg, stamp)

    @classmethod
    def deserialize(cls, content:dict):
        return cls(
            data_dir=content['data_dir'],
            stamp_fpth=content['stamp_fpth'],
            frame_id=content.get('frame_id', "/camera"),
            schema=content.get('schema', "foxglove_msgs/msg/CompressedImage"),
            topic=content.get('topic', "/image"),
            suffix=content.get('suffix', ".png"),
        )
class Box2DWriter(ROSBAG_WRITER):
    def __init__(self, 
                 data_dir:str, 
                 stamp_fpth:str, 
                 frame_id="/camera", 
                 schema="foxglove_msgs/msg/ImageAnnotations", 
                 topic="/box2d",
                 suffix=".txt") -> None:
        super().__init__(data_dir, stamp_fpth, frame_id, schema, topic)
        self.data_reader = BOX2DReader(suffix=suffix)
    
    def write(self, writer):
        
        for (_, cls_bboxes), stamp in zip(
            self.data_reader.load_data(self.data_dir),
            self.data_reader.load_timestamps(self.stamp_fpth)
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
            data_dir=content['data_dir'],
            stamp_fpth=content['stamp_fpth'],
            frame_id=content.get('frame_id', "/camera"),
            schema=content.get('schema', "foxglove_msgs/msg/ImageAnnotations"),
            topic=content.get('topic', "/box2d"),
            suffix=content.get('suffix', ".txt"),
        )
class Box3DWriter(ROSBAG_WRITER):
    
    def __init__(self, 
                 data_dir:str, 
                 stamp_fpth:str, 
                 frame_id = '/pcd', 
                 schema = 'foxglove_msgs/msg/SceneUpdate', 
                 topic = '/box3d',
                 suffix = '.txt',
                 mode = "elan" ) -> None:
        super().__init__(data_dir, stamp_fpth, frame_id, schema, topic)
        self.data_reader = BOX3DReader(suffix=suffix, mode=mode)
        
        
    def write(self, writer):
        for (fname, cls_bbox3d), stamp in zip(
            self.data_reader.load_data(self.data_dir),
            self.data_reader.load_timestamps(self.stamp_fpth)
        ):
            msg = SceneUpdate()
            scene_entity = SceneEntity(
                timestamp = stamp,
                frame_id = self.frame_id,
                id = self.topic,
            )
            for cls, bbox3d in cls_bbox3d:
                # Something weird
                position, size, quat = bbox3d['pos'], bbox3d['size'], bbox3d['quat']
                
                r,g,b = map(float,class_colors[cls])
                cube = CubePrimitive(
                    pose  = Pose(    position = Point(x=position[0], y=position[1], z=position[2]),
                                orientation = Quaternion(x = quat[0], y =quat[1], z=quat[2], w=quat[3])
                            ),
                    size  = Vector3(x=size[0], y=size[1], z=size[2]), 
                    color = Color(r=r, g=g, b=b, a=0.5)
                )
                scene_entity.cubes.append(cube)
                # Need to check?
                scene_entity.texts.append(
                    TextPrimitive(
                        position=Pose(    position = Point(x=position[0], y=position[1], z=position[2]),
                                orientation = Quaternion(x = quat[0], y =quat[1], z=quat[2], w=quat[3])
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
        return cls(
            data_dir=content['data_dir'],
            stamp_fpth=content['stamp_fpth'],
            frame_id=content.get('frame_id', "/pcd"),
            schema=content.get('schema', "foxglove_msgs/msg/SceneUpdate"),
            topic=content.get('topic', "/box3d"),
            suffix=content.get('suffix', ".txt"),
            mode=content.get('mode', 'elan')
        )
class CALIBWriter(ROSBAG_WRITER):
    def __init__(self, 
                 data_path:str, 
                 stamp_fpth:str, 
                 frame_id = "/camera", 
                 schema = "foxglove_msgs/msg/CameraCalibration", 
                 topic = "/camera_info",
                 suffix = '.json',
                 im_shape = None,
                 mode ='elan',
                 *args, 
                 **kwargs,
                ) -> None:
        super().__init__(None, stamp_fpth, frame_id, schema, topic)
        self.data_path = data_path
        self.data_reader = CALIBReader(suffix=suffix, mode=mode, *args, **kwargs)
        self.im_shape =im_shape
        
    def write(self, writer):        
        for (_, calib), stamp in zip(
            self.data_reader.load_data(self.data_path), 
            self.data_reader.load_timestamps(self.stamp_fpth)
        ):
            msg = CameraCalibration(
                timestamp = stamp,
                frame_id = self.frame_id,
                height = int(self.im_shape[0]),
                width = int(self.im_shape[1]),
                k = calib[:3,:3].flatten(),
                p = calib.flatten()
            )
            self.write2bag(writer, msg, stamp)
    @classmethod
    def deserialize(cls, content: dict):
        import inspect
        arguments = [ param.name for param in \
                        inspect.signature(cls.__init__).parameters.values() ]
        rest_kwargs = { k:v for k,v in content.items() if k not in arguments}
        return cls(
            data_path=content['data_path'],
            stamp_fpth=content['stamp_fpth'],
            frame_id=content.get('frame_id', "/camera"),
            schema=content.get('schema', "foxglove_msgs/msg/CameraCalibration"),
            topic=content.get('topic', "/camera_info"),
            suffix=content.get('suffix', ".json"),
            im_shape=content.get('im_shape', None),
            mode=content.get('mode', 'elan'),
            **rest_kwargs
    )
class TFWriter(ROSBAG_WRITER):
    def __init__(self, 
                 data_path:str, 
                 stamp_fpth:str,  
                 schema = 'foxglove_msgs/msg/FrameTransforms', 
                 topic = '/tf',
                 suffix = '.json',
                 parent_frame_id =None,
                 child_frame_id = None,
                 mode = 'elan',
                 *args,
                 **kwargs,) -> None:
        super().__init__(None, stamp_fpth, None, schema, topic)
        self.data_path = data_path
        self.data_reader = TFReader(suffix=suffix, mode=mode, *args, **kwargs)
        self.parent_frame_id = parent_frame_id
        self.child_frame_id = child_frame_id
        
    def write(self, writer, ):
        for (_, tf), stamp in zip(
            self.data_reader.load_data(self.data_path),
            self.data_reader.load_timestamps(self.stamp_fpth)
        ):
            msg = FrameTransforms()
            translate, quat = tf[:3, 3].flatten(), Rotation.from_matrix(tf[:3,:3]).as_quat()
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
        import inspect
        arguments = [ param.name for param in \
                        inspect.signature(cls.__init__).parameters.values() ]
        rest_kwargs = { k:v for k,v in content.items() if k not in arguments}
        return cls(
            data_path=content['data_path'],
            stamp_fpth=content['stamp_fpth'],
            schema=content.get('schema', "foxglove_msgs/msg/FrameTransforms"),
            topic=content.get('topic', "/tf"),
            suffix=content.get('suffix', ".json"),
            parent_frame_id=content.get('parent_frame_id', None),
            child_frame_id=content.get('child_frame_id', None),
            mode=content.get('mode', 'elan'),
            **rest_kwargs
        )


