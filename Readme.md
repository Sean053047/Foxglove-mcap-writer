# Foxglove-mcap-writer

This project aims to write the independent sensor data to mcap format so that let user can visualize the results via [Foxglove](https://foxglove.dev/).

Here, I use the format for ROS2. You can check the detail schema in the [link](https://docs.foxglove.dev/docs/visualization/message-schemas/introduction/).  

## Environment

To setup the environment, use sdc2 nix shell environment. 

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

---

Authors: Yang, Yi-Xiang
Email: sadlamb910803@gmail.com
