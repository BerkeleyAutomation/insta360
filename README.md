# Lifelong LERF ROS
This repo will contain the ROS2 workspace code for the Lifelong LERF project. Currently, it can stream color and depth images from a Realsense D457 and teleop move a Turtlebot4 with keyboard commands. This has been tested on Ubuntu 22.04.

## Turtlebot Installation and Setup
To setup the Turtlebot to talk to the computer and vice versa, follow the instructions in this link: https://turtlebot.github.io/turtlebot4-user-manual/setup/basic.html.
If Donatello is nearby, follow these instructions to setup and install necessary libraries
```
ssh ubuntu@10.65.87.91
sudo apt-get install ros-humble-realsense2-camera
sudo apt-get install ros-humble-librealsense2* #Should already be on there
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
git clone https://github.com/BerkeleyAutomation/LifelongLERFROS.git src
```

## Computer Installation and Setup
```
sudo apt-get install ros-humble-turtlebot4-navigation
sudo apt-get install ros-humble-navigation2
sudo apt-get install ros-humble-nav2-bringup
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
git clone https://github.com/BerkeleyAutomation/LifelongLERFROS.git src
cd src
source env_setup.bash
```

## Run Camera (Realsense D457/D435)
ON TURTLEBOT
```
ssh ubuntu@10.65.87.91
cd ~/ros2_ws
colcon build
. install/setup.bash
ros2 launch robot_bringup standard_realsense.launch.py
```

ON COMPUTER
```
cd ~/ros2_ws
colcon build
. install/setup.bash
ros2 launch camera_bringup image_visualization.launch.py
```
A color and depth image window will open showing the camera streams.


## Run Droid-SLAM
ON TURTLEBOT (Needs camera images as input)
```
ros2 launch robot_bringup standard_realsense.launch.py
```

ON COMPUTER (Terminal 1)
```
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
ON COMPUTER (Terminal 2)
```
ros2 launch droid_slam_ros droid_slam.launch.py
```
From there, you should get a Viser link and can view Droid-SLAM in action.
## Installation


## Run Navigation 11/18 (Still under development)

Okay, so we got a lot of moving pieces to get this working. We will consolidate soon.
First, make sure the Realsense is plugged into the Fetch and run it.
```
ssh fetch@fetch59.local # password robotics
source ~/ros2_foxy/install/setup.bash
cd ~/ros2_ws
colcon build
. install/setup.bash
ros2 launch realsense2_camera rs_launch.py
```

To verify this works, in another window, ssh into the fetch, source ros2_foxy, and check the frequency of /ros2_camera/color/image_raw. It should be at about 15 Hz.

Next, run the image compression node on the Fetch. We need this because directly subscribing to the full image on the computer causes too much lag, so we subscribe to the compressed image on the computer.
```
ssh fetch@fetch59.local # password robotics
source ~/ros2_foxy/install/setup.bash
cd ~/ros2_ws
colcon build
. install/setup.bash
ros2 run image_compression color_image_compression_node.py
```

To verify this works, in another window, ssh into the fetch, source ros2_foxy, and check the frequency of /imageo_compressedo. It should also be at about 15 Hz.

Next, ssh into the fetch and run the ros2 to ros1 bridge.
```
ssh fetch@fetch59.local
source /opt/ros/noetic/setup.bash
rosparam load ~/ros2_ws/src/image_compression/params/bridge.yaml
source ~/ros2_foxy/install/setup.bash
ros2 run ros1_bridge parameter_bridge
```

Then, run the talker node in ROS2 on the computer and the listener on ROS1 on the Fetch.
```
ros2 run demo_nodes_cpp talker
```

```
ssh fetch@fetch59.local
source /opt/ros/noetic/setup.bash
rosrun roscpp_tutorials listener
```
You should be seeing messages on both the ROS1 and ROS2 ends indicating the bridge is working.

Next, on the computer, run the image uncompression node.
```
cd ~/lifelong_lerf_ws
colcon build
. install/setup.bash
ros2 run camera_bringup realsense_compressed_converter.py
```

To verify this works, in another window, check the frequency of /repub_image_raw. It should also be at about 15 Hz.

Next, on the computer in another window, run RTABMAP and then run rviz and make sure you can visualize the map.
```
cd ~/lifelong_lerf_ws
colcon build
. install/setup.bash
ros2 launch realsense_rtabmap_slam_bringup new_rtabmap.launch.py
```

Next, on computer in another window, run navigation. You should see the window say the words "Creating bond timer..."
```
cd ~/lifelong_lerf_ws
colcon build
. install/setup.bash
ros2 launch realsense_rtabmap_slam_bringup navigation.launch.py
```

To verify navigation is working, echo the following topics on the computer: /cmd_vel and /navigate_to_pose/_action/status

Now we need to verify that the bridge can still work, so kill the chatter topic talker, and then rerun it, and make sure the listener still works.

Now that you have verified this, you can permanently kill the talker.

In another window, run the twist to string conversion on the computer.
```
cd ~/lifelong_lerf_ws
colcon build
. install/setup.bash
ros2 run realsense_rtabmap_slam_bringup twist_to_string.py
```

Then, on the fetch, run the corresponding string to twist conversion.
```
ssh fetch@fetch59.local
source /opt/ros/noetic/setup.bash
cd lifelong_lerf_fetch_ws
catkin_make
source devel/setup.bash
rosrun nuc_bridge string_to_twist.py
```

Now, you should put a goal down in RVIZ, and it should navigate to the goal!!! You can verify that you reached the goal when the /navigate_to_pose/_action/status has a status 4 as opposed to staus 2. Status 6 means that the goal was aborted


