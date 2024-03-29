#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import os
import glob
import cv2
import torch
from std_msgs.msg import String
from sensor_msgs.msg import Image,CompressedImage
import sys
from cv_bridge import CvBridge
import torch.nn.functional as F
import time

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.realsense_image_publisher_ = self.create_publisher(CompressedImage, '/repub_realsense_color/compressed', 10)
        self.realsense_depth_publisher_ = self.create_publisher(Image, '/repub_realsense_depth', 10)
        self.left_zed_left_image_publisher_ = self.create_publisher(CompressedImage, '/repub_left_zed_left_color/compressed', 10)
        self.left_zed_depth_image_publisher_ = self.create_publisher(Image,'/repub_left_zed_depth',10)
        self.cv_bridge_ = CvBridge()
        file_location = os.path.dirname(os.path.realpath(__file__))
        d455_folder = file_location+'/../../../../2024_03_02_week_data/d455_zed_kitchen'
        left_zed_image_list = sorted(glob.glob(os.path.join(d455_folder, 'tri_rgb', '*.png')))[::1]
        left_zed_depth_list = sorted(glob.glob(os.path.join(d455_folder, 'tri_depth', '*.npy')))[::1]
        realsense_image_list = sorted(glob.glob(os.path.join(d455_folder, 'realsense_compressed_image', '*.png')))[::1]
        realsense_depth_list = sorted(glob.glob(os.path.join(d455_folder, 'realsense_depth_image', '*.npy')))[::1]
        total_image_stream = []
        for t, (left_zed_image_file, left_zed_depth_file,realsense_image_file,realsense_depth_file) in enumerate(zip(left_zed_image_list, left_zed_depth_list,realsense_image_list,realsense_depth_list)):
            left_zed_image = cv2.imread(left_zed_image_file)
            realsense_image = cv2.imread(realsense_image_file)
            
            left_zed_depth_image = np.load(left_zed_depth_file)
            realsense_depth_image = np.load(realsense_depth_file)

            total_image_stream.append((t,left_zed_image,left_zed_depth_image,realsense_image,realsense_depth_image))
        for (t,left_zed_image,left_zed_depth_image,realsense_image,realsense_depth_image) in total_image_stream:
            left_zed_image_rosmsg = self.cv_bridge_.cv2_to_compressed_imgmsg(left_zed_image)
            left_zed_depth_image_rosmsg = self.cv_bridge_.cv2_to_imgmsg(left_zed_depth_image)
            realsense_image_rosmsg = self.cv_bridge_.cv2_to_compressed_imgmsg(realsense_image)
            realsense_depth_image_rosmsg = self.cv_bridge_.cv2_to_imgmsg(realsense_depth_image)
            # Get current time
            current_time = self.get_clock().now()
            
            # Set the timestamp of the image message
            left_zed_image_rosmsg.header.stamp = current_time.to_msg()
            left_zed_depth_image_rosmsg.header.stamp = current_time.to_msg()
            realsense_image_rosmsg.header.stamp = current_time.to_msg()
            realsense_depth_image_rosmsg.header.stamp = current_time.to_msg()
            self.left_zed_left_image_publisher_.publish(left_zed_image_rosmsg)
            self.left_zed_depth_image_publisher_.publish(left_zed_depth_image_rosmsg)
            self.realsense_image_publisher_.publish(realsense_image_rosmsg)
            self.realsense_depth_publisher_.publish(realsense_depth_image_rosmsg)
            print("T: " + str(t))
            time.sleep(0.2)
        print("DONE")

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

    def image_stream(self,datapath, use_depth=False, stride=1):
        """ image generator """

        fx, fy, cx, cy = np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()
        left_zed_image_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::stride]
        left_zed_depth_list = sorted(glob.glob(os.path.join(datapath, 'depth', '*.png')))[::stride]
        total_image_stream = []
        for t, (image_file, depth_file) in enumerate(zip(left_zed_image_list, left_zed_depth_list)):
            
            image = cv2.imread(image_file)
            # image = image[61:404, 102:661,:]
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 5000.0
            # image = cv2.resize(image,(424,240))
            
            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
            h1 = int(h0 * ((384) / (h0)))
            w1 = int(w0 * (( 512) / (w0)))
            h1 = 480
            w1 = 848
            image = cv2.resize(image, (w1, h1))
            image = image[:h1-h1%8, :w1-w1%8]
            image = torch.as_tensor(image).permute(2, 0, 1)
            
            depth = torch.as_tensor(depth)
            depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
            depth = depth[:h1-h1%8, :w1-w1%8]

            intrinsics = torch.as_tensor([fx, fy, cx, cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)
            if use_depth:
                total_image_stream.append((t,image[None],depth,intrinsics))
                #yield t, image[None], depth, intrinsics
            else:
                total_image_stream.append((t,image[None],intrinsics))
                #yield t, image[None], intrinsics
        return total_image_stream


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MinimalPublisher()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()