#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import time
import numpy as np

class ImageSubscriberNode(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()

        self.color_sub = self.create_subscription(CompressedImage, '/repub_compressed_image_synced', self.color_callback, 30)
        #self.color_pub = self.create_publisher(Image,'/repub_compressed_image',10)
        #self.left_sub = self.create_subscription(CompressedImage, '/repub_left_image_synced', self.left_callback, 10)
        #self.left_pub = self.create_publisher(Image,'/repub_left_image',10)
        #self.right_sub = self.create_subscription(CompressedImage, '/repub_right_image_synced', self.right_callback, 10)
        #self.right_pub = self.create_publisher(Image,'/repub_right_image',10)
        # self.depth_sub = self.create_subscription(Image,'/repub_depth_image_synced',self.depth_callback,10)

    def color_callback(self, msg):
        start_time = time.time()
        cv2_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        cv2.imshow('compressed_image',cv2_image)
        cv2.waitKey(1)
        print("Time taken: " + str(time.time() - start_time))
        #image_msg = self.bridge.cv2_to_imgmsg(cv2_image,encoding='bgr8')
        #self.color_pub.publish(image_msg)

    def left_callback(self, msg):
        cv2_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        cv2.imshow('left_image',cv2_image)
        cv2.waitKey(1)

    def right_callback(self, msg):
        cv2_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        cv2.imshow('right_image',cv2_image)
        cv2.waitKey(1)

    def depth_callback(self, msg):
        cv2_image = self.bridge.imgmsg_to_cv2(msg)
        min_depth = np.min(cv2_image)
        max_depth = np.max(cv2_image)
        normalized_depth_image = (cv2_image - min_depth) / (max_depth - min_depth)
        normalized_depth_image = (normalized_depth_image * 255).astype(np.uint8)
        cv2.imshow('depth_image',normalized_depth_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriberNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
