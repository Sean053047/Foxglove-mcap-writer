# Foxglove-mcap-writer

This project aims to write the independent sensor data to mcap format so that let user can visualize the results via [Foxglove](https://foxglove.dev/).

Here, I use the format for ROS2. You can check the detail schema in the [link](https://docs.foxglove.dev/docs/visualization/message-schemas/introduction/).  

## Environment (2025.01.24)

Please ensure successfully install ROS2 and foxglove_msgs. Follow the instructions from websites. Then install the python packages listed in requirements.txt.

```txt
Update:
2025.01.10  Currently, required environment is set in the branch `perception-evaluation` of sdc2.
            Writable foxglove message type:
                1. pcd (PointCloud)
                2. Box3d (SceneUpdate)
                3. image (CompressedImage)
                4. tf (FrameTransforms)
                5. calibration (CameraCalibration)
```

## Usage

```bash
src/write2mcap.py --yaml ${yaml file path} --mcap ${mcap directory}
```  

You can check the reference yaml file at [example.yaml](./example.yaml).

Notice that for message type with `fname2time`. Those `fname2time` file must specify the filename to timestamp dictionary. It's necessary to write mcap bag with timestamp.

## YAML Attribute introduction (2025.01.24)

#### Reader attributes

* data_dir: Directory saving data files.

* fstem2time: Yaml file which specifies all file stems to timestamps. When file stems are timestamps, this attribute can be ignored.  

* suffix: Suffix for loaded data files.

* fpth: Specify the tf file.  

#### Writer attributes  

* frame_id: The dumping frame id for selected data.

* schema: The schema for mcap format. (Currently, only implement for foxglove format. )

* topic: Dumped topic name for selected data.

* parent_frame: Parent frame id.

* child_frame: Child frame id.

* color_offset: Color offset with r,g,b values within the range [0, 1].

`✓`: Required. `-`: No needed. `△`: optional

|Category| data_dir | fstem2time | fpth | suffix | frame_id | schema |　topic | parent_frame | child_frame | color_offset |
|---| --- | --- | --- | --- | --- | -- | -- | --| -- | -- |
| **pcd**      |✓|△|-|✓|✓|✓|✓|-|-|-|
| **box3d**    |✓|△|-|✓|✓|✓|✓|-|-|△|
| **camera**   |✓|△|-|✓|✓|✓|✓|-|-|-|
| **static_tf**|-|-|✓|✓|-|✓|✓|✓|✓|-|
| **calib**    |-|-|✓|✓|-|✓|✓|✓|-|-|

---



Authors: Yang, Yi-Xiang
Email: sadlamb910803@gmail.com

