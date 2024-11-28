import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_utils.readers import (
    PCDReader, 
    IMGReader, 
    CALIBReader,
    BOX2DReader, 
    BOX3DReader
)
from data_utils.utils import get_box_vertices
import cv2 
import numpy as np 
from scipy.spatial.transform import Rotation as R

from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
import sensor_msgs_py.point_cloud2 as pcd2_utils
from sensor_msgs.msg import PointCloud2 as std_PCD2
from sensor_msgs.msg import CompressedImage as std_IMG
from sensor_msgs.msg import PointField
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Header
from foxglove_msgs.msg import PointCloud as fox_PCD #CompressedImage, CubePrimitive
from foxglove_msgs.msg import PackedElementField
from foxglove_msgs.msg import CompressedImage as fox_IMG
from foxglove_msgs.msg import (
    ImageAnnotations,
    TextAnnotation, 
    Point2,
    CameraCalibration
)

class PCDPublisher(Node):

    NP2ROSTYPE = {
        np.int8     : 1, # INT8
        np.uint8    : 2, # UINT8
        np.int16    : 3, 
        np.uint16   : 4,
        np.int32    : 5,
        np.uint32   : 6,
        np.float32  : 7,
        np.float64  : 8
    }
    def __init__(
        self, data_dir: str, stamp_fpth: str, topic="/pointcloud", option = "std", suffix='.pcd'
    ):
        if option == "foxglove":
            msg_type = fox_PCD     
            publish_callback = self.publish_foxglove
        elif option == "std":
            msg_type = std_PCD2
            publish_callback = self.publish_std
            
        super().__init__("PCD_publisher")
        self.publisher = self.create_publisher(msg_type, topic, 30)
        self.data_dir, self.stamp_fpth = data_dir, stamp_fpth
        self.data_reader = PCDReader(suffix=suffix)
        self.timer = self.create_timer(10, publish_callback)
        
        self.frame_id = "base_link"
    def publish_foxglove(self):
        self.get_logger().info(f"To foxglove studio ...")
        for (fname, pc_info, pc_data), stamp in zip(
                self.data_reader.load_data(self.data_dir),
                self.data_reader.load_timestamps(self.stamp_fpth),
        ):
            import numpy as np 
            np.float32
            msg = fox_PCD()
            fields = pc_info['fields']
            N = pc_data.shape[0]
            self.get_logger().info(f"Publish: {stamp.sec}.{stamp.nanosec:09d}s {fname.name} {N}pts {fields}")    
            msg.timestamp = stamp
            msg.frame_id = self.frame_id
            msg.pose.position = Point(x=0.0, y=0.0, z=0.0)
            msg.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
            msg.point_stride = pc_data.itemsize * len(fields)
            msg.fields = [
                PackedElementField(name='x', offset=0, type=7),
                PackedElementField(name='y', offset=4, type=7),
                PackedElementField(name='z', offset=8, type=7)
            ]
            msg.data = pc_data.tobytes()
            self.publisher.publish(msg)
    
    def publish_std(self):
        for i, ((fname, pc_info, pc_data), stamp) in enumerate(zip(
                self.data_reader.load_data(self.data_dir),
                self.data_reader.load_timestamps(self.stamp_fpth),
        )):
            N = pc_data.shape[0]
            self.get_logger().info(f"Publish: {stamp.sec}.{stamp.nanosec:09d}s {fname.name} | {pc_info['fields']} | {N} points")    
            header = Header(stamp=stamp, frame_id=self.frame_id)
            fields = list()
            accum_offset = 0
            for f, np_type in zip(pc_info['fields'] , pc_info['types']):
                fields.append( PointField(  name=f, 
                                            offset=accum_offset, 
                                            datatype= self.NP2ROSTYPE[np_type],
                                            count=1
                                            )
                            )
                accum_offset = accum_offset + np.dtype(np_type).itemsize
            
            msg = pcd2_utils.create_cloud(header, fields, pc_data)
            self.publisher.publish(msg)
            
class IMGPublisher(Node):
    
    def __init__(self, 
                 data_dir: str, 
                 stamp_fpth: str, 
                 topic="/image", 
                 frame_id = "camera",
                 option="foxglove", suffix='.png'):
        if option == "foxglove":
            msg_type = fox_IMG
            publish_callback = self.publish_foxglove
        elif option == "std":
            msg_type = std_IMG
            publish_callback = self.publish_std
            self.bridge = CvBridge()
        super().__init__("IMG_publisher")
        self.data_dir, self.stamp_fpth = data_dir, stamp_fpth
        self.data_reader = IMGReader(suffix=suffix)
        self.publisher = self.create_publisher(msg_type, topic, 30)
        self.timer = self.create_timer(10, publish_callback)
        self.frame_id = frame_id
        
    @property
    def img_shape(self):
        for _, img in self.data_reader.load_data(self.data_dir):
            return img.shape
        
    def publish_foxglove(self):
        for (fname, img), stamp in zip(
                self.data_reader.load_data(self.data_dir), 
                self.data_reader.load_timestamps(self.stamp_fpth)
        ):
            self.get_logger().info(f"Image Publish: {stamp.sec}.{stamp.nanosec:09d}s| {fname.name} | {img.shape[0]}*{img.shape[1]}")
            ret, buffer = cv2.imencode(self.data_reader.suffix, img)
            if not ret:
                self.get_logger().info(f"[{self.get_name()}] Failed to encode image as {self.data_reader.suffix} format.")    
            msg = fox_IMG(
                timestamp = stamp,
                frame_id = self.frame_id,
                data = np.array(buffer).tobytes(), 
                format = self.data_reader.suffix.strip('.'),
            )
            self.publisher.publish(msg)
    def publish_std(self):
        for (fname, img), stamp in zip(
                self.data_reader.load_data(self.data_dir),
                self.data_reader.load_timestamps(self.stamp_fpth),
        ):
            self.get_logger().info(f"Image publish: {stamp}| {fname.name} | {img.shape[0]}*{img.shape[1]}")    
            msg = self.bridge.cv2_to_compressed_imgmsg(img, self.data_reader.suffix.strip('.'))
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            self.publisher.publish(msg)


class CamCalibrationPublisher(Node):
    def __init__(self, 
                 json_fpth:str, 
                 stamp_fpth:str, 
                 img_shape, 
                 topic='/cam_calib',
                 frame_id ="camera", ):
        super().__init__("CAMCALIB_publisher")
        
        self.height, self.width = img_shape[0], img_shape[1]
        self.stamp_fpth = stamp_fpth
        self.data_reader = CALIBReader(suffix=".json")
        self.data_reader.load_data(json_fpth)
        
        self.frame_id = frame_id
        self.timer = self.create_timer(10, self.publish_callback)
        self.publisher = self.create_publisher(CameraCalibration, topic=topic, qos_profile=30)
        
    def publish_callback(self):
        for stamp in self.data_reader.load_timestamps(self.stamp_fpth):
            print(stamp)
            print(self.frame_id)
            msg = CameraCalibration(
                timestamp = stamp,
                frame_id = self.frame_id,
                width = self.width,
                height = self.height,
                k = self.data_reader.get_data_attr('intrinsic')
            )
            
            self.publisher.publish(msg)
            self.get_logger().info(f"{self.data_reader.get_data_attr('intrinsic')}")
        self.get_logger().info(f"{self.get_name()} finish publish.")

# Todo: Fix the bug
class BOX2DPublisher(Node): 
    def __init__(self, data_dir:str, stamp_fpth: str, topic="/box2d", suffix='.txt'):
        super().__init__("BOX2D_publisher")
        self.publisher = self.create_publisher( ImageAnnotations, topic, 10)
        self.data_dir, self.stamp_fpth = data_dir, stamp_fpth
        self.data_reader = BOX2DReader(suffix=suffix)
        self.timer = self.create_timer(1, self.publish_callback)
        self.frame_id = "camera"
        
    def publish_callback(self):
        msg = ImageAnnotations()
        for (fname, cls_bboxes), stamp in zip(
                self.data_reader.load_data(self.data_dir),
                self.data_reader.load_timestamps(self.stamp_fpth),
        ):
            for cls, bbox in cls_bboxes:
                # print(bbox)
                vertices = get_box_vertices(bbox)
                # vertices.append(vertices[0])
                pts_ann = msg.points.add()
                pts_ann.timestamp = stamp
                pts_ann.type = 2 # Closed polygon
                
                pts_ann.points = [Point2(x= float(v[0]), y= float(v[1])) for v in vertices]
                pts_ann.outline_color.r = CLASS2COLOR[cls][0]
                pts_ann.outline_color.g = CLASS2COLOR[cls][1]
                pts_ann.outline_color.b = CLASS2COLOR[cls][2]
                pts_ann.outline_color.a = CLASS2COLOR[cls][3]
                pts_ann.thickness = 3
                msg.points.append(pts_ann)
            
                txt_ann = TextAnnotation()
                txt_ann.timestamp = stamp
                txt_ann.position.x, txt_ann.position.y = list( map(float, vertices[0]))
                txt_ann.text = "c"
                txt_ann.font_size = 12.0
                txt_ann.text_color.r = CLASS2COLOR[cls][0]
                txt_ann.text_color.g = CLASS2COLOR[cls][1]
                txt_ann.text_color.b = CLASS2COLOR[cls][2]
                txt_ann.text_color.a = 1.0
                msg.texts.append(txt_ann)
                
            self.get_logger().info(f"{self.get_name()}| {stamp.sec}.{stamp.nanosec:09d}s | {len(cls_bboxes)} boxes")
        
        self.get_logger().info(f"ImageAnnotation publish | # boxes: {len(msg.points)}")    
        self.publisher.publish(msg)

# Todo: Finish this code.
class BOX3DPublisher(Node):
    
    def __init__(self, data_dir:str, stamp_fpth: str, topic="/box3d", suffix='.txt'):
        super().__init__("BOX3D_publisher")
        self.publisher = self.create_publisher( ImageAnnotations, topic, 30)
        node_name = "BOX3D_publisher_foxglove"
        
        self.data_dir, self.stamp_fpth = data_dir, stamp_fpth
        self.data_reader = BOX3DReader()

# Todo: Fnish this code 
class TFPublisher(Node):
    def __init__(self, extrinsic_matrix):
        translation = extrinsic_matrix[:3,3]
        rotation = extrinsic_matrix[:3,:3]
        self.tf = translation. R.from_matrix(rotation).as_quat()

if __name__ == "__main__":
    from auto_vehicle.Elan_utils.utils import elan_path_resolve
    main_dir = (
        "/home/sean/Desktop/ROSBag_analyze/Elan_data/city/EVIL_2024-05-09-14-12-16"
    )
    msg_types = ['pcd', 'img', 'box2d']
    category = {
        "pcd": "VLS128",
        "img": "image",
        "box2d":"image_label",
    }
    data_dirs = dict()
    stamp_fpths = dict()
    for type in msg_types:
        data_dirs[type], stamp_fpths[type] = elan_path_resolve(main_dir, category[type])    
        
    
    # with open(stamp_fpths['pcd'], 'r' ) as file:
    #     pcd_timestamps = np.array([float(line.strip()) for line in file.readlines()], dtype=np.float128)
    # with open(stamp_fpths['img'], 'r' ) as file:
    #     img_timestamps = np.array([float(line.strip()) for line in file.readlines()], dtype=np.float128)
    
    rclpy.init(args=None)
    pcd_node = PCDPublisher(data_dir=data_dirs['pcd'], stamp_fpth=stamp_fpths['pcd'])
    # img_node = IMGPublisher(data_dir=data_dirs['img'], stamp_fpth=stamp_fpths['img'])
    # box2d_node = BOX2DPublisher(data_dir=data_dirs['box2d'], stamp_fpth=stamp_fpths['box2d'])
    rclpy.spin_once(pcd_node)
    # rclpy.spin_once(img_node)
    # rclpy.spin_once(box2d_node)
    pcd_node.destroy_node()
    # img_node.destroy_node()
    # box2d_node.destroy_node()
    print("[END] End of publishing.")
    rclpy.shutdown()
