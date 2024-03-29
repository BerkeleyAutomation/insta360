#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageVisualization(Node):
    def __init__(self):
        super().__init__('image_visualization')
        # self.depth_subscriber_ = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.image_callback, 10)
        self.color_subscriber_ = self.create_subscription(CompressedImage, '/ros2_camera/color/image_raw/compressed', self.compressed_image_callback, 1)
        self.depth_subscriber_ = self.create_subscription(Image,'/ros2_camera/depth/image_rect_raw',self.image_callback,1)
        #self.subscription = self.create_subscription(CompressedImage,'/camera/color/image_raw/compressed',self.compressed_image_callback,10)
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().info(f"Error converting image: {e}")
            return
        cv2.imshow("Depth Image", cv_image)
        cv2.waitKey(1)  # This is necessary for the OpenCV window to update

    def compressed_image_callback(self,msg):
        image_np = self.bridge.compressed_imgmsg_to_cv2(msg,'bgr8')
        cv2.imshow("Color image",image_np)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_visualization = ImageVisualization()
    rclpy.spin(image_visualization)
    image_visualization.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
