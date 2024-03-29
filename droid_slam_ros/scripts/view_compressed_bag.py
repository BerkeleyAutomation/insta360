#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/repub_compressed_image_synced',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()
        self.i_ = 1
    def image_callback(self, msg):
        print(self.i_)
        self.i_ += 1
        # Convert compressed image to OpenCV image
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg)
        
        # Display the image using cv2.imshow()
        cv2.imshow("Image", cv_image)
        cv2.waitKey(1)  # You might need to adjust the waitKey value
        
def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
