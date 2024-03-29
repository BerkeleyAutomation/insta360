#!/usr/bin/env python3

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', 10)
        self.depth_image_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)

        # Path to the folders containing images and depth images
        self.image_folder_path = "/home/kushtimusprime/legs_ws/2024_03_02_week_data/d455_kitchen/images"
        self.depth_folder_path = "/home/kushtimusprime/legs_ws/2024_03_02_week_data/d455_kitchen/depth_images"

        self.image_files = sorted(os.listdir(self.image_folder_path))
        self.depth_files = sorted(os.listdir(self.depth_folder_path))

        self.timer = self.create_timer(5.0, self.publish_images)

    def publish_images(self):
        for image_file, depth_file in zip(self.image_files, self.depth_files):
            time.sleep(0.5)
            image_path = os.path.join(self.image_folder_path, image_file)
            depth_path = os.path.join(self.depth_folder_path, depth_file)

            # Read image
            image = cv2.imread(image_path)
            if image is None:
                self.get_logger().error("Failed to read image from path: %s" % image_path)
                continue

            # Convert image to ROS format
            image_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")

            # Read depth image
            depth_image = np.load(depth_path)
            if depth_image is None:
                self.get_logger().error("Failed to read depth image from path: %s" % depth_path)
                continue
            
            print("depth image data type is:", depth_image.dtype)
            # Convert depth image to ROS format
            depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="passthrough")

            # Set depth image encoding
            depth_image_msg.encoding = "UINT16"  # Assuming depth image is float32

            # Publish images
            self.image_pub.publish(image_msg)
            self.depth_image_pub.publish(depth_image_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

