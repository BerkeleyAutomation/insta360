# Lifelong LERF ROS
This repo will contain the ROS2 workspace code for the Lifelong LERF project. Currently, it contains a ROS driver for the Insta360 ONE X2 camera. When the camera is
connected with USB, the 2 fisheye images are published to separate ROS topics. The insta_fisheye node will publish the byte array containing the H264 streaming data. The h264_subscriber node will decode the byte array accordingly to generate an image that is then published. This has been tested on Ubuntu 22.04.

## Installation and Setup
It should just work with the default Ubuntu 22.04 setup, but I could be missing something.
```
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
git clone https://github.com/BerkeleyAutomation/LifelongLERFROS.git src
```

## Running the code
```
colcon build
. install/setup.bash
ros2 launch insta_ros_driver full_launch.launch.py
```
If an Insta360 ONE X2 is plugged in with USB, it should start the livestream, and publish the front and back fisheye images to the /camera/image_raw1 and /camera/image_raw2 topics respectively.