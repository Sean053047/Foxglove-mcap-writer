import argparse
import rosbag2_py
from Elan_utils.constant import (
    TF_JSON_PATH,
    CAM_JSON_PATH,
    MSG_SUFFIX,
)
from data_utils.writers import (
    ROSBAG_WRITER,
    PCDWriter, 
    TFWriter,
    IMGWriter,
    CALIBWriter,
    Box2DWriter,
    Box3DWriter,
)

def make_args():
    from datetime import datetime
    parser = argparse.ArgumentParser(prog="Write data to mcap.")
    parser.add_argument(
        "--mcap_path",
        type=str,
        default=f"./record_{datetime.now().strftime('%m-%d_%H:%M:%S')}",
    )
    # Load data from yaml
    parser.add_argument('--yaml_fpth', type=str, default='', help="Load data from yaml settings.")
    return parser.parse_args()


def make_writers_from_yaml(fpth:str):
    import yaml
    with open(fpth, 'r') as file:
        contents =yaml.safe_load(file)
    yaml2class = {
        'pcd_writer': PCDWriter,
        'tf_writer': TFWriter,
        'img_writer': IMGWriter,
        'calib_writer': CALIBWriter,
        'box2d_writer': Box2DWriter,
        'box3d_writer': Box3DWriter
    }
    data_writers = list()
    for key, setting_list in contents.items():
        writer_cls = yaml2class[key]
        writer_cls : ROSBAG_WRITER
        for setting in setting_list:
            data_writers.append(writer_cls.deserialize(setting))

    return data_writers


def main():
    args = make_args()
    # Read Elan data path
    data_writers = make_writers_from_yaml(args.yaml_fpth)
    mcap_writer = rosbag2_py.SequentialWriter()
    mcap_writer.open(
        rosbag2_py.StorageOptions(uri=args.mcap_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    for dw in data_writers:
        mcap_writer = dw.init_writer(mcap_writer)
        dw.write(mcap_writer)

if __name__ == "__main__":
    main()

    