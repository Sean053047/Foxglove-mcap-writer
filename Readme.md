# Foxglove-mcap-writer

This project aims to write the independent sensor data to mcap format so that let user can display the results in [Foxglove](https://foxglove.dev/) platform.

Here, I use the format for ROS2. You can check the detail schema in the [link](https://docs.foxglove.dev/docs/visualization/message-schemas/introduction/).  

## Usage

```bash
src/write2mcap.py --mcap_path ${Output directory} --yaml_fpth ${yaml file path}
```  

You can check the formate for yaml file in [example/setting.yaml](./example/setting.yaml)

---

Authors: Yang, Yi-Xiang
Email: sadlamb910803@gmail.com
