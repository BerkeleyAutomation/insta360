#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import time

class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_rgb = self.create_publisher(CompressedImage, '/repub_compressed_image_synced', 10)
        self.publisher_depth = self.create_publisher(Image, '/repub_depth_image_synced', 10)
        self.cv_bridge = CvBridge()
        self.publish_images()

    def publish_images(self):
        rgb_folder = '/home/kushtimusprime/legs_ws/manual_splat_good_directory/compressed_image'
        depth_folder = '/home/kushtimusprime/legs_ws/manual_splat_good_directory/depth_image'

        rgb_files = sorted(os.listdir(rgb_folder))
        depth_files = sorted(os.listdir(depth_folder))
        i = 0
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            rgb_image_path = os.path.join(rgb_folder, rgb_file)
            depth_image_path = os.path.join(depth_folder, depth_file)

            rgb_image = cv2.imread(rgb_image_path)
            depth_image = np.load(depth_image_path)
            current_time = self.get_clock().now().to_msg()
            rgb_msg = self.cv_bridge.cv2_to_compressed_imgmsg(rgb_image)
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_image, encoding='passthrough')
            rgb_msg.header.stamp = current_time
            depth_msg.header.stamp = current_time
            self.publisher_rgb.publish(rgb_msg)
            self.publisher_depth.publish(depth_msg)
            self.get_logger().info(f"Published {rgb_file} and {depth_file}")
            i += 1
            time.sleep(2)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
