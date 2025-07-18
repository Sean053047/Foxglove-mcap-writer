import argparse
import rosbag2_py
from tqdm import tqdm
from collections import defaultdict
import os.path as osp
from data_utils.writers import (
    ROSBAGWRITER,
    PCDWriter, 
    StaticTFWriter,
    TimePosesWriter,
    CameraWriter,
    CALIBWriter,
    Box2DWriter,
    Box3DWriter,
)
from data_utils.readers import (
    BaseReader,
    PCDReader,
    StaticTFReader,
    TimePosesReader,
    CameraReader,
    CALIBReader,
    Box2DReader,
    Box3DReader   
)

def Initialize_from_yaml(data_root:str, setting_fpth:str):
    import yaml
    with open(setting_fpth, 'r') as file:
        contents =yaml.safe_load(file)
        
    class2writer = {
        'pcd': PCDWriter,
        'static_tf': StaticTFWriter,
        'time_poses': TimePosesWriter,
        'camera': CameraWriter,
        'calib': CALIBWriter,
        'box2d': Box2DWriter,
        'box3d': Box3DWriter
    }
    class2reader = {
        'pcd': PCDReader,
        'static_tf': StaticTFReader,
        'time_poses': TimePosesReader,
        'camera': CameraReader,
        'calib': CALIBReader,
        'box2d': Box2DReader,
        'box3d': Box3DReader
    }
    
    data_IO = list()
    tf_IO = list()
    
    for key, setting_list in contents.items():
        writer_cls = class2writer[key]
        reader_cls = class2reader[key]
        writer_cls : ROSBAGWRITER
        reader_cls : BaseReader    
        if key in ('static_tf', 'calib', 'time_poses'):
            for setting in setting_list:
                setting['fpth']= osp.join(data_root, setting['fpth'])
                tf_IO.append(
                    (
                        reader_cls.deserialize(setting),
                        writer_cls.deserialize(setting)
                    )
                )
        else:
            for setting in setting_list:
                setting['data_dir']= osp.join(data_root, setting['data_dir'])
                data_IO.append(
                    (reader_cls.deserialize(setting), 
                    writer_cls.deserialize(setting))
                )
    return data_IO, tf_IO

def make_args():
    from datetime import datetime
    parser = argparse.ArgumentParser(prog="Write data to mcap.")
    # Load data from yaml
    parser.add_argument('--yaml', type=str, required=True, help="Load data from yaml settings.")
    parser.add_argument('--data-root', type=str, default='./', )
    # Output name of mcap
    parser.add_argument(
        "--mcap",
        type=str,
        help='Output directory',
        default=f"./record_{datetime.now().strftime('%m-%d_%H:%M:%S')}",
    )
    return parser.parse_args()

def main(args):
    # Read Elan data path
    data_IO, tf_IO = Initialize_from_yaml(args.data_root, args.yaml)
    mcap = rosbag2_py.SequentialWriter()
    mcap.open(
        rosbag2_py.StorageOptions(uri=args.mcap, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    # * Iterate over data IO
    frame_time_record = list() # ? Will be used in the below section
    for reader, writer in tqdm(data_IO, total=len(data_IO), desc="Data IO:"):
        writer: ROSBAGWRITER
        reader: BaseReader
        mcap = writer.init_writer(mcap)      
        for i, fpth in tqdm(enumerate(reader.all_files), total=len(reader.all_files), desc=f"Writing {writer.topic} files:"):
            stamp = reader.get_Time(stamp=fpth.stem)
            writer.write(mcap, stamp, reader.load_data(fpth))    
            frame_time_record.append(stamp)
    # * Iterate over tf IO
    for reader, writer in tqdm(tf_IO, total=len(tf_IO), desc="TF IO:"):
        # Write static tf
        mcap = writer.init_writer(mcap)
        if type(writer) is not TimePosesWriter:
            tf = reader.load_data()
            for stamp in frame_time_record:
                writer.write(mcap, stamp, tf)
        else:
            stamps, tps = reader.load_data()
            writer.write(mcap, stamps, tps)
if __name__ == "__main__":
    args = make_args()
    main(args)

    