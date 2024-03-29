#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import numpy as np
import os
import time

class ManualSplatCollectionNode(Node):
    def __init__(self):
        super().__init__('manual_splat_collection_node')
        self.bridge = CvBridge()

        self.realsense_color_image_ = None
        self.realsense_depth_image_ = None
        self.left_zed_left_color_image_ = None
        self.left_zed_right_color_image_ = None
        self.left_zed_depth_image_ = None
        self.i_ = 0
        os.makedirs('manual_splat_directory', exist_ok=True)
        os.makedirs('manual_splat_directory/realsense_compressed_image', exist_ok=True)
        os.makedirs('manual_splat_directory/realsense_depth_image', exist_ok=True)
        os.makedirs('manual_splat_directory/left_zed_left_image', exist_ok=True)
        os.makedirs('manual_splat_directory/left_zed_right_image', exist_ok=True)
        os.makedirs('manual_splat_directory/left_zed_depth_image', exist_ok=True)

        self.realsense_color_sub = Subscriber(
            self, CompressedImage, '/repub_realsense_color/compressed')
        self.realsense_depth_sub = Subscriber(
            self, Image, '/repub_realsense_depth')
        self.left_zed_left_color_sub = Subscriber(
            self, CompressedImage, '/repub_left_zed_left_color/compressed')
        self.left_zed_right_color_sub = Subscriber(
            self, CompressedImage, '/repub_left_zed_right_color/compressed')
        self.left_zed_depth_sub = Subscriber(
            self, Image, '/repub_left_zed_depth')
        self.keyboard_sub = self.create_subscription(String,'/save_images',self.keyboard_callback,10)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.realsense_color_sub,self.realsense_depth_sub,self.left_zed_left_color_sub,self.left_zed_right_color_sub,self.left_zed_depth_sub], 10, 0.25)
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info('Press Enter to save images')

    def sync_callback(self,realsense_color_msg,realsense_depth_msg,left_zed_left_msg,left_zed_right_msg,left_zed_depth_msg):
        self.realsense_color_image_,self.realsense_depth_image_,self.left_zed_left_color_image_,self.left_zed_right_color_image_,self.left_zed_depth_image_ = realsense_color_msg,realsense_depth_msg,left_zed_left_msg,left_zed_right_msg,left_zed_depth_msg

    def keyboard_callback(self,msg):
        start_time = time.time()
        if self.realsense_color_image_ is not None and self.realsense_depth_image_ is not None and self.left_zed_left_color_image_ is not None and self.left_zed_right_color_image_ is not None and self.left_zed_depth_image_ is not None:
            image_str = 'image' + str(self.i_).zfill(6) 
            cv2.imwrite('manual_splat_directory/realsense_compressed_image/' + image_str + '.png',self.bridge.compressed_imgmsg_to_cv2(self.realsense_color_image_))
            cv2.imwrite('manual_splat_directory/left_zed_left_image/' + image_str + '.png',self.bridge.compressed_imgmsg_to_cv2(self.left_zed_left_color_image_))
            cv2.imwrite('manual_splat_directory/left_zed_right_image/' + image_str + '.png',self.bridge.compressed_imgmsg_to_cv2(self.left_zed_right_color_image_))
            np.save('manual_splat_directory/realsense_depth_image/' + image_str,self.bridge.imgmsg_to_cv2(self.realsense_depth_image_))
            np.save('manual_splat_directory/left_zed_depth_image/' + image_str,self.bridge.imgmsg_to_cv2(self.left_zed_depth_image_))
            print("Saved image " + str(image_str))
            self.i_ += 1
        else:
            print("Null images")

def main(args=None):
    rclpy.init(args=args)
    node = ManualSplatCollectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
