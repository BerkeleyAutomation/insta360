#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.realsense_subscription = self.create_subscription(
            CompressedImage,
            '/repub_realsense_color/compressed',
            self.realsense_image_callback,
            10)
        self.left_zed_subscription = self.create_subscription(
            CompressedImage,
            '/repub_left_zed_left_color/compressed',
            self.left_zed_image_callback,
            10)
        self.right_zed_subscription = self.create_subscription(
            CompressedImage,
            '/repub_right_zed_left_color/compressed',
            self.right_zed_image_callback,
            10)
        self.bridge = CvBridge()
        self.i_ = 1
    def realsense_image_callback(self, msg):
        print(self.i_)
        self.i_ += 1
        # Convert compressed image to OpenCV image
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        
        # Display the image using cv2.imshow()
        cv2.imshow("Realsense Image", cv_image)
        cv2.waitKey(1)  # You might need to adjust the waitKey value

    def left_zed_image_callback(self, msg):
        print(self.i_)
        self.i_ += 1
        # Convert compressed image to OpenCV image
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        
        # Display the image using cv2.imshow()
        cv2.imshow("Left Zed Image", cv_image)
        cv2.waitKey(1)  # You might need to adjust the waitKey value

    def right_zed_image_callback(self, msg):
        print(self.i_)
        self.i_ += 1
        # Convert compressed image to OpenCV image
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        
        # Display the image using cv2.imshow()
        cv2.imshow("Right Zed Image", cv_image)
        cv2.waitKey(1)  # You might need to adjust the waitKey value
        
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
