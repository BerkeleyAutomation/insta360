# Copyright 2018 Lucas Walter
# All rights reserved.
#
# Software License Agreement (BSD License 2.0)
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Lucas Walter nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import argparse
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


#### LAUNCHES ONLY THE 3 SIDE ARUDCAMS NOT THE FRONT REALSENSE

def generate_launch_description():
    ld = LaunchDescription()

    usb_cam_dir = get_package_share_directory('camera_bringup')

    # get path to params file
    params_path = os.path.join(
        usb_cam_dir,
        'config')
    params2_path = os.path.join(
        params_path,
        'camLeft_params.yaml'
    )

    print(params_path)
    

    # ld.add_action(Node(
    #     package='usb_cam', executable='usb_cam_node_exe', output='screen',
    #     name='camera1',
    #     # namespace=ns,
    #     parameters=[params1_path],
    #     remappings=[('/image_raw', 'camBack/image_raw'), ('/image_raw/compressed', 'camBack/image_raw/compressed')]
    #     ))
    ld.add_action(Node(
        package='usb_cam', executable='usb_cam_node_exe', output='screen',
        name='camera2',
        # namespace=ns,
        parameters=[params2_path],
        remappings=[('/image_raw', 'camLeft/image_raw'), ('/image_raw/compressed', 'camLeft/image_raw/compressed'), ('/camera_info', 'camLeft/camera_info')]
        ))
    # ld.add_action(Node(
    #     package='usb_cam', executable='usb_cam_node_exe', output='screen',
    #     name='camera3',
    #     # namespace=ns,
    #     parameters=[params3_path],
    #     remappings=[('/image_raw', 'camRight/image_raw'), ('/image_raw/compressed', 'camRight/image_raw/compressed')]
    #     ))
    
    # ld.add_action(Node(
    #     package='usb_cam', executable='show_image.py', output='screen',
    #     # namespace=ns,
    #     # arguments=[image_manip_dir + "/data/mosaic.jpg"])
    #     # remappings=[('image_in', 'image_raw')]
    #     ))

    return ld