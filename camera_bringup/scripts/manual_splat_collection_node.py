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

        self.color_image_ = None
        self.left_image_ = None
        self.right_image_ = None
        self.depth_image_ = None
        self.i_ = 0
        os.makedirs('manual_splat_directory', exist_ok=True)
        os.makedirs('manual_splat_directory/compressed_image', exist_ok=True)
        os.makedirs('manual_splat_directory/left_image', exist_ok=True)
        os.makedirs('manual_splat_directory/right_image', exist_ok=True)
        os.makedirs('manual_splat_directory/depth_image', exist_ok=True)

        #self.color_sub = self.create_subscription(CompressedImage,'/repub_realsense_color/compressed',self.image_callback,10)
        #self.left_sub = self.create_subscription(CompressedImage,'/repub_left/compressed',self.left_callback,10)

        self.color_sub = Subscriber(
            self, CompressedImage, '/repub_realsense_color/compressed')
        self.left_sub = Subscriber(
            self, CompressedImage, '/repub_left/compressed')
        self.right_sub = Subscriber(
            self, CompressedImage, '/repub_right/compressed')
        self.depth_sub = Subscriber(
            self, Image, '/repub_realsense_depth')
        self.keyboard_sub = self.create_subscription(String,'/save_images',self.keyboard_callback,10)
        
        self.ts = ApproximateTimeSynchronizer(
            [self.color_sub,self.left_sub,self.right_sub,self.depth_sub], 10, 0.25)
        self.ts.registerCallback(self.sync_callback)

        self.get_logger().info('Press Enter to save images')

    def sync_callback(self,color_msg,left_msg,right_msg,depth_msg):
        self.color_image_,self.left_image_,self.right_image_,self.depth_image_ = color_msg,left_msg,right_msg,depth_msg

    def left_callback(self, color_msg):
        print("LEFT HERE")
        import pdb
        pdb.set_trace()

    def image_callback(self, color_msg):
        print("IN HERE")
        import pdb
        pdb.set_trace()
        #

    def keyboard_callback(self,msg):
        start_time = time.time()
        if self.color_image_ is not None and self.left_image_ is not None and self.right_image_ is not None and self.depth_image_ is not None:
            image_str = 'image' + str(self.i_).zfill(6) 
            cv2.imwrite('manual_splat_directory/compressed_image/' + image_str + '.png',self.bridge.compressed_imgmsg_to_cv2(self.color_image_))
            cv2.imwrite('manual_splat_directory/left_image/' + image_str + '.png',self.bridge.compressed_imgmsg_to_cv2(self.left_image_))
            cv2.imwrite('manual_splat_directory/right_image/' + image_str + '.png',self.bridge.compressed_imgmsg_to_cv2(self.right_image_))
            np.save('manual_splat_directory/depth_image/' + image_str,self.bridge.imgmsg_to_cv2(self.depth_image_))
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
