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
        #self.subscription = self.create_subscription(Image, '/new_raw', self.image_callback, 10)
        self.subscription = self.create_subscription(CompressedImage,'/image_raw/compressed',self.compressed_image_callback,10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().info(f"Error converting image: {e}")
            return

        cv2.imshow("Received Image", cv_image)
        cv2.waitKey(2)  # This is necessary for the OpenCV window to update

    def compressed_image_callback(self,msg):
        image_np = self.bridge.compressed_imgmsg_to_cv2(msg,'bgr8')
        print(image_np.shape)
        cv2.imshow("Received image",image_np)
        cv2.waitKey(2)

def main(args=None):
    rclpy.init(args=args)
    image_visualization = ImageVisualization()
    rclpy.spin(image_visualization)
    image_visualization.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
