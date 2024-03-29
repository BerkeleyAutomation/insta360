#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import os
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from std_msgs.msg import Bool
import glob
from geometry_msgs.msg import Pose,Point,Quaternion
import matplotlib.pyplot as plt

from lifelong_msgs.msg import ImagePose, ImagePoses
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation as R
import cv2
import time
from cv_bridge import CvBridge
import argparse
import torch
import message_filters
from tf2_ros import TransformBroadcaster
import sys
file_location = os.path.dirname(os.path.realpath(__file__))
ultimate_reference_filepath = file_location + '/../../../../src/droid_slam_ros/config/ultimate_reference.png'
sys.path.append(file_location+'/../../share/droid_slam_ros/droid_slam')
from droid import Droid
import droid_backends
from lietorch import SE3
from sensor_msgs.msg import PointCloud2, PointField
import torch.nn.functional as F
from std_msgs.msg import String

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.cv_bridge_ = CvBridge()
        self.bundle_adjust_subscriber = self.create_subscription(String,'/bundle_adjust',self.bundleAdjustCallback,1)
        self.publisher = self.create_publisher(ImagePoses, '/camera/color/imagepose',50)
        self.cam_params = {"w": 848,
            "h": 480,
            "fl_x": 428.2534484863281,
            "fl_y": 427.8413391113281,
            "cx": 422.375732421875,
            "cy": 236.3165283203125,
            "k1": -0.05663116276264191,
            "k2": 0.06359340995550156,
            "k3": -0.020485159009695053,
            "camera_model": "OPENCV",}

        # self.rgb_sub = message_filters.Subscriber(self, CompressedImage, '/repub_compressed_image_synced')
        self.rgb_sub = message_filters.Subscriber(self, CompressedImage, '/repub_compressed_image_synced')
        self.depth_sub = message_filters.Subscriber(self,Image,'/repub_depth_image_synced')
        self.realsense_publisher = self.create_publisher(ImagePoses, '/sim_realsense',50)
        self.last_pose = None
        self.t_ = 0
        self.image_counter = 0
        self.stride = 1
        self.stored_camera_stream_ = []
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 50, 3.0)
        self.ts.registerCallback(self.image_callback_uncompressed)
        #self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.cam_info_sub], 50, 0.2)
        #self.ts.registerCallback(self.fullImageCallback)
        parser = argparse.ArgumentParser()
        parser.add_argument("--datapath",default="/home/kushtimusprime/legs_ws/src/droid_slam_ros/datasets/ETH3D-SLAM/training/sfm_house_loop")
        parser.add_argument("--weights", default="droid.pth")
        parser.add_argument("--buffer", type=int, default=512)
        parser.add_argument("--image_size", default=[240, 320])
        parser.add_argument("--disable_vis", action="store_true")
        parser.add_argument("--upsample", action="store_true")
        parser.add_argument("--beta", type=float, default=0.5)
        parser.add_argument("--filter_thresh", type=float, default=2.0)
        parser.add_argument("--warmup", type=int, default=8)
        parser.add_argument("--keyframe_thresh", type=float, default=3.5)
        parser.add_argument("--frontend_thresh", type=float, default=16.0)
        parser.add_argument("--frontend_window", type=int, default=16)
        parser.add_argument("--frontend_radius", type=int, default=1)
        parser.add_argument("--frontend_nms", type=int, default=0)

        parser.add_argument("--stereo", action="store_true")
        parser.add_argument("--depth", action="store_true",default=True)

        parser.add_argument("--backend_thresh", type=float, default=22.0)
        parser.add_argument("--backend_radius", type=int, default=2)
        parser.add_argument("--backend_nms", type=int, default=3)

        self.droid_args_ = parser.parse_args()
        torch.multiprocessing.set_start_method('spawn')
        print("Running evaluation on {}".format(self.droid_args_.datapath))
        print(self.droid_args_)

        self.cam_transform = np.diag([1, -1, -1, 1])
        self.tf_broadcaster = TransformBroadcaster(self)
        self.uncompressed_counter_ = 1
        self.sim_realsense_sub = self.create_subscription(ImagePoses,'/sim_realsense',self.sim_realsense_callback,50)

    def xyzquat2mat(self,vec):
        xyz = vec[:3]
        quat = vec[3:]
        matrix = np.eye(4)
        try:
            rotation = R.from_quat(quat)
        except ValueError:
            import pdb; pdb.set_trace()
        # rotation = R.from_quat(quat[3] + quat[:4])
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = xyz
        matrix = np.linalg.inv(matrix) @ self.cam_transform
        return matrix
    
    def sim_realsense_callback(self,full_msg):
        print("UNCOMPRESSED COUNTER: " + str(self.uncompressed_counter_))
        self.uncompressed_counter_ += 1
        msg = full_msg.image_poses[0]
        start_time = time.time()
        print("sim realsense callback",self.image_counter)
        cv_image_original = self.cv_bridge_.compressed_imgmsg_to_cv2(msg.img)
        depth_image_original = self.cv_bridge_.imgmsg_to_cv2(msg.depth)
        if(depth_image_original.dtype == np.uint16):
            depth_image_original = depth_image_original / 1000.0
        # cv_image_original = cv2.resize(cv_image_original,(zed_w,zed_h))
        new_image = cv_image_original
        new_depth = depth_image_original
        if 'cx' not in self.cam_params:
            print("Not recieved intr yet, skipping frame")
            return
        # if(new_depth.dtype == np.uint16):
        #     new_depth = new_depth / 1000.0
        # height, width, _ = new_image.shape
        # height = height - height % 32
        # width = width - width % 32
        # new_image = new_image[:height, :width, :3]
        # cv_image_original = new_image
        h0, w0, _ = new_image.shape
        h1 = h0
        w1 = w0
        new_image = cv2.resize(new_image, (w1, h1))
        new_image = new_image[:h1-h1%8, :w1-w1%8]
        new_image = torch.as_tensor(new_image).permute(2, 0, 1)
        
        new_depth = torch.as_tensor(new_depth)
        new_depth = F.interpolate(new_depth[None,None], (h1, w1)).squeeze()
        new_depth = new_depth[:h1-h1%8, :w1-w1%8]

        intrinsics = torch.as_tensor([self.cam_params['fl_x'],self.cam_params['fl_y'],self.cam_params['cx'],self.cam_params['cy']])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)
        
        image = new_image[None]
        depth = new_depth

        start_time = time.time()
        #if not args.disable_vis:
        #    self.show_image(image[0])

        if self.t_ == 0:
            self.droid_args_.image_size = [image.shape[2], image.shape[3]]
            self.droid = Droid(self.droid_args_)
        
        self.droid.track(self.t_, image, depth, intrinsics=intrinsics)
        
        image_poses = ImagePoses()
        if(self.droid.video.counter.value == self.image_counter):
            if(self.last_pose is not None):
                ros_transform = TransformStamped()
                ros_transform.header.stamp = self.get_clock().now().to_msg()
                ros_transform.header.frame_id = "map_droid"
                ros_transform.child_frame_id = "droid_optical_frame"
                last_pose_arr = self.last_pose
                xyz = last_pose_arr[:3]
                quat = last_pose_arr[3:]
                matrix = np.eye(4)
                rotation = R.from_quat(quat)
                matrix[:3, :3] = rotation.as_matrix()
                matrix[:3, 3] = xyz
                map_to_droid_link = matrix
                map_to_base = np.linalg.inv(map_to_droid_link)# @ tilt_tf
                alt_position = map_to_base[:3,3]
                alt_rotation_matrix = map_to_base[:3,:3]
                alt_quaternion = R.from_matrix(alt_rotation_matrix).as_quat()

                ros_transform.transform.translation.x = alt_position[0]
                ros_transform.transform.translation.y = alt_position[1]
                ros_transform.transform.translation.z = alt_position[2]

                ros_transform.transform.rotation.x = alt_quaternion[0]
                ros_transform.transform.rotation.y = alt_quaternion[1]
                ros_transform.transform.rotation.z = alt_quaternion[2]
                ros_transform.transform.rotation.w = alt_quaternion[3]
                self.tf_broadcaster.sendTransform(ros_transform)
            return
        
        self.image_counter += 1
        pose = self.droid.video.poses[self.droid.video.counter.value-1].cpu().numpy()
        disp = self.droid.video.disps[self.droid.video.counter.value-1].cpu().numpy()
        dirty_index = torch.where(self.droid.video.dirty.clone())[0]
        if(len(dirty_index) > 0):
            self.droid.video.dirty[dirty_index] = False
            print("Dirty index")
            print(dirty_index, dirty_index.shape)
            # convert poses to 4x4 matrix
            poses = torch.index_select(self.droid.video.poses, 0, dirty_index)
            disps = torch.index_select(self.droid.video.disps, 0, dirty_index)
            Ps = SE3(poses).inv().matrix().cpu().numpy()

            images = torch.index_select(self.droid.video.images, 0, dirty_index)
            images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
            points = droid_backends.iproj(SE3(poses).inv().data, disps, self.droid.video.intrinsics[0]).cpu()
            i = len(dirty_index) - 1
            filter_thresh = 0.005
            thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
    
            count = droid_backends.depth_filter(
                self.droid.video.poses, self.droid.video.disps, self.droid.video.intrinsics[0], dirty_index, thresh)

            count = count.cpu()
            disps = disps.cpu()
            masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))
            mask = masks[i].reshape(-1)
            pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
            clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
            pts_array = pts.flatten().tolist()
            clr_array = clr.flatten().tolist()
            image_poses.points = pts_array
            image_poses.colors = clr_array
            image_poses.mask_idxs = torch.where(mask)[0].cpu().numpy().flatten().tolist()
            print("Saved points")
        self.last_pose = pose
        ros_transform = TransformStamped()
        ros_transform.header.stamp = self.get_clock().now().to_msg()
        ros_transform.header.frame_id = "map_droid"
        ros_transform.child_frame_id = "droid_optical_frame"
        last_pose_arr = self.last_pose
        xyz = last_pose_arr[:3]
        quat = last_pose_arr[3:]
        matrix = np.eye(4)
        rotation = R.from_quat(quat)
        matrix[:3, :3] = rotation.as_matrix()
        matrix[:3, 3] = xyz
        map_to_droid_link = matrix
        map_to_base = np.linalg.inv(map_to_droid_link)# @ tilt_tf
        alt_position = map_to_base[:3,3]
        alt_rotation_matrix = map_to_base[:3,:3]
        alt_quaternion = R.from_matrix(alt_rotation_matrix).as_quat()

        ros_transform.transform.translation.x = alt_position[0]
        ros_transform.transform.translation.y = alt_position[1]
        ros_transform.transform.translation.z = alt_position[2]

        ros_transform.transform.rotation.x = alt_quaternion[0]
        ros_transform.transform.rotation.y = alt_quaternion[1]
        ros_transform.transform.rotation.z = alt_quaternion[2]
        ros_transform.transform.rotation.w = alt_quaternion[3]
        self.tf_broadcaster.sendTransform(ros_transform)
        print("Adding droid keyframe...")

        h0, w0, _ = cv_image_original.shape
        #h1 = h0
        #w1 = w0
        #cv_image_original = cv2.resize(cv_image_original, (w1, h1))
        
        #depth_image_original = torch.as_tensor(depth_image_original)
        #depth_image_original = F.interpolate(depth_image_original[None,None], (h1, w1),mode='bilinear',antialias=True).squeeze()
        #depth_image_original = depth_image_original.cpu().detach().numpy()

        intrinsics = torch.as_tensor([self.cam_params['fl_x'], self.cam_params['fl_y'], self.cam_params['cx'], self.cam_params['cy']])
        #intrinsics[0::2] *= (w1 / w0)
        #intrinsics[1::2] *= (h1 / h0)

        image_pose_msg = ImagePose()
        image_pose_msg.img = self.cv_bridge_.cv2_to_compressed_imgmsg(cv_image_original)
        image_pose_msg.depth = self.cv_bridge_.cv2_to_imgmsg(depth_image_original)
        image_pose_msg.w = w0
        image_pose_msg.h = h0
        # image_pose_msg.fl_x = intrinsics[0].item()
        # image_pose_msg.fl_y = intrinsics[1].item()
        # image_pose_msg.cx = intrinsics[2].item()
        # image_pose_msg.cy = intrinsics[3].item()

        # image_pose_msg.w = self.cam_params['w']
        # image_pose_msg.h = self.cam_params['h']
        image_pose_msg.fl_x = self.cam_params['fl_x']
        image_pose_msg.fl_y = self.cam_params['fl_y']
        image_pose_msg.cx = self.cam_params['cx']
        image_pose_msg.cy = self.cam_params['cy']
        
        image_pose_msg.k1 = self.cam_params['k1']
        image_pose_msg.k2 = self.cam_params['k2']
        image_pose_msg.k3 = self.cam_params['k3']
        posemat = self.xyzquat2mat(pose)
        pose = posemat[:3,3]
        orient = R.from_matrix(posemat[:3,:3]).as_quat()
        print("Big T: " + str(self.t_))
        # self.stored_camera_stream_.append([self.image_counter,torch.from_numpy(cv_image_original),original_intrinsics])
        self.stored_camera_stream_.append([self.t_,image,intrinsics])
        # if(self.image_counter % 149 == 0): #Kitchen
        # #if(self.image_counter % 77 == 0): # Hallway
        #     new_poses = self.droid.globalBundleAdjust(self.stored_camera_stream_)
        #     new_poses_list = new_poses.flatten().tolist()
        #     image_poses.got_prev_poses = True
        #     image_poses.prev_poses = new_poses_list
        # else:
        #     image_poses.got_prev_poses = False

        image_pose_msg.pose = Pose(position=Point(x=pose[0],y=pose[1],z=pose[2]),orientation=Quaternion(x=orient[0],y=orient[1],z=orient[2],w=orient[3]))  # Replace with your Pose message
        self.first_image_ = False
        image_poses.image_poses = [image_pose_msg]
        self.publisher.publish(image_poses)
        self.t_ += 1
        
    def bundleAdjustCallback(self,msg):
        print("Starting global bundle adjust")
        image_poses = ImagePoses()
        new_poses = self.droid.globalBundleAdjust(self.stored_camera_stream_)
        new_poses_list = new_poses.flatten().tolist()
        blank_image_pose = ImagePose()
        image_poses.image_poses = [blank_image_pose]
        image_poses.points = []
        image_poses.colors = []
        image_poses.mask_idxs = []
        image_poses.got_prev_poses = True
        image_poses.prev_poses = new_poses_list
        self.publisher.publish(image_poses)

    def image_callback_uncompressed(self, img_msg, depth_msg):
        realsense_sim_msg = ImagePose()
        # cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding='bgr8')  # Convert ROS Image message to OpenCV image
        # cv_img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        realsense_sim_msg.img = img_msg
        realsense_sim_msg.depth = depth_msg
        imageposes_msg = ImagePoses()
        imageposes_msg.image_poses = [realsense_sim_msg]
        self.realsense_publisher.publish(imageposes_msg)
        return
  
def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()