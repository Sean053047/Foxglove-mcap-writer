FROM osrf/ros:humble-desktop-full
# Set up environment variables
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=humble

COPY ./requirements.txt /tmp/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-${ROS_DISTRO}-foxglove-msgs \
    ros-${ROS_DISTRO}-rosbag2-storage-mcap \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools 
RUN rosdep update || true

RUN pip install -r /tmp/requirements.txt
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
    
WORKDIR /workspace
CMD ["bash"]