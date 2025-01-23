import argparse
import rosbag2_py
from tqdm import tqdm
from collections import defaultdict

from data_utils.writers import (
    ROSBAGWRITER,
    PCDWriter, 
    StaticTFWriter,
    CameraWriter,
    CALIBWriter,
    Box2DWriter,
    Box3DWriter,
)
from data_utils.readers import (
    BaseReader,
    PCDReader,
    StaticTFReader,
    CameraReader,
    CALIBReader,
    Box2DReader,
    Box3DReader   
)

def Initialize_from_yaml(fpth:str):
    import yaml
    with open(fpth, 'r') as file:
        contents =yaml.safe_load(file)
        
    class2writer = {
        'pcd': PCDWriter,
        'static_tf': StaticTFWriter,
        'camera': CameraWriter,
        'calib': CALIBWriter,
        'box2d': Box2DWriter,
        'box3d': Box3DWriter
    }
    class2reader = {
        'pcd': PCDReader,
        'static_tf': StaticTFReader,
        'camera': CameraReader,
        'calib': CALIBReader,
        'box2d': Box2DReader,
        'box3d': Box3DReader
    }
    
    data_IO = list()
    static_tf_IO = defaultdict(list)
    
    for key, setting_list in contents.items():
        writer_cls = class2writer[key]
        reader_cls = class2reader[key]
        writer_cls : ROSBAGWRITER
        reader_cls : BaseReader    
        if key in ('static_tf', 'calib'):
            for setting in setting_list:
                static_tf_IO[setting['parent_frame_id']].append(
                    (
                        reader_cls.deserialize(setting),
                        writer_cls.deserialize(setting)
                    )
                )
        else:
            for setting in setting_list:
                data_IO.append(
                    (reader_cls.deserialize(setting), 
                    writer_cls.deserialize(setting))
                )
            
    return data_IO, static_tf_IO

def make_args():
    from datetime import datetime
    parser = argparse.ArgumentParser(prog="Write data to mcap.")
    # Load data from yaml
    parser.add_argument('--yaml', type=str, required=True, help="Load data from yaml settings.")
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
    data_IO, static_tf_IO = Initialize_from_yaml(args.yaml)
    mcap = rosbag2_py.SequentialWriter()
    mcap.open(
        rosbag2_py.StorageOptions(uri=args.mcap, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    for reader, writer in tqdm(data_IO, total=len(data_IO)):
        writer: ROSBAGWRITER
        reader: BaseReader
        mcap = writer.init_writer(mcap)
        # Load if meet condition
        if writer.frame_id in static_tf_IO:
            stfs, stf_writers = list(), list()
            for rw in static_tf_IO[writer.frame_id]:
                stfs.append(rw[0].load_data())
                stf_writers.append(rw[1])
                mcap = rw[1].init_writer(mcap)
        else:
            stf_writers = None
              
        for i, fpth in enumerate(reader.all_files):
            stamp = reader.get_Time(stamp=fpth.stem)
            writer.write(mcap, stamp, reader.load_data(fpth))    
            
            # If static tf exist, write to mcap.
            if stf_writers is not None:
                for stf, ww in zip(stfs, stf_writers):
                    ww.write(mcap, stamp, stf)    
        
        # Remove from dictionary
        if stf_writers is not None:
            del static_tf_IO[writer.frame_id]
            
if __name__ == "__main__":
    args = make_args()
    main(args)

    