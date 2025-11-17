#! /bin/bash

docker run -it --rm \
    -v $(pwd)/../:/workspace/ \
    -v /media/itri_sean/Expansion:/media/itri_sean/Expansion \
    -v /data/K-RadarOcc2:/data/K-RadarOcc2 \
    --name foxglove-mcap-writer \
    foxglove-mcap-writer:latest \
    bash