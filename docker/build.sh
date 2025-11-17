#! /bin/bash

docker build -t foxglove-mcap-writer:latest \
    -f docker/Dockerfile .