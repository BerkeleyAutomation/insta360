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
sys.path.append(file_location+'/../../share/droid_slam_ros/droid_slam')
from droid import Droid
import droid_backends
from lietorch import SE3
from sensor_msgs.msg import PointCloud2, PointField
import torch.nn.functional as F

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.cv_bridge_ = CvBridge()
        self.publisher = self.create_publisher(ImagePoses, '/camera/color/imagepose',1)
        self.cam_params = {}
        self.left_cam_params = {
            "w": 848,
            "h": 480,
            "fl_x": 502.22156413,
            "fl_y": 503.07629405,
            "cx": 423.59392771,
            "cy":  244.60180021,
            "k1": 0.20101611,
            "k2": -0.47480968,
            "k3": -0.00121715,
            "camera_model": "OPENCV",
        }
        self.right_cam_params = {
            "w": 848,
            "h": 480,
            "fl_x": 505.11569948,
            "fl_y": 507.41553284,
            "cx": 422.20778592,
            "cy":  246.2358567,
            "k1": 1.82171279e-01,
            "k2": -4.25860120e-01,
            "k3": 1.32861466e-04,
            "camera_model": "OPENCV",
        }
        # self.rgb_sub = message_filters.Subscriber(self, CompressedImage, '/repub_compressed_image_synced')
        self.rgb_sub = message_filters.Subscriber(self, Image, '/repub_compressed_image_synced')
        self.cam_info_sub = self.create_subscription(CameraInfo,'/ros2_camera/color/camera_info',self.cam_intr_cb,1)
        self.depth_sub = message_filters.Subscriber(self,Image,'/ros2_camera/depth/image_rect_raw')
        self.done_sub_ = self.create_subscription(Bool,'/loop_done',self.doneCallback,10)
        self.realsense_publisher = self.create_publisher(ImagePoses, '/sim_realsense',1)


        r_ang = -70.7099538 * np.pi / 180
        l_ang = -r_ang

        self.droid_nerf_frame_to_left_nerf_frame_ = np.array([[np.cos(l_ang), 0, np.sin(l_ang), -0.050404],
                                    [0, 1, 0, 0.01016],
                                    [-np.sin(l_ang), 0, np.cos(l_ang), 0.043395],
                                    [0, 0, 0, 1]])
                                    
        self.droid_nerf_frame_to_right_nerf_frame_ = np.array([[np.cos(r_ang), 0, np.sin(r_ang), 0.027404],
                                    [0, 1, 0, 0.01016],
                                    [-np.sin(r_ang), 0, np.cos(r_ang), 0.043395],
                                    [0, 0, 0, 1]])
        # self.droid_nerf_frame_to_left_nerf_frame_ = np.array([[0.00, 0.000 ,1.000 ,-0.055],
        #                                                             [0.000,  1.000 ,0.000 , 0.000],
        #                                                             [-1.000 , 0.0 , 0.0 ,0.041],
        #                                                             [0,0,0,1]])
        # self.droid_nerf_frame_to_right_nerf_frame_ = np.array([[0.00, 0.000 ,-1.000 ,0.032],
        #                                                             [0.000,  1.000 ,-0.000 , 0.000],
        #                                                             [1.000 , 0.0 , 0.0 ,0.041],
        #                                                             [0,0,0,1]])
        self.tstamps = []
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_pose = None
        self.t_ = 0
        self.iter_times = []
        self.image_counter = 0
        self.stride = 1
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 1, 3.0)
        self.ts.registerCallback(self.image_callback_uncompressed)
        #self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.cam_info_sub], 10, 0.2)
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
        self.prev_time = time.time()
        self.curr_time = time.time()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.sim_realsense_sub = self.create_subscription(ImagePoses,'/sim_realsense',self.sim_realsense_callback,1)

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
        msg = full_msg.image_poses[0]
        start_time = time.time()
        print("sim realsense callback",self.image_counter)
        cv_image_original = self.cv_bridge_.compressed_imgmsg_to_cv2(msg.img)
        depth_image_original = self.cv_bridge_.imgmsg_to_cv2(msg.depth)
        new_image = cv_image_original
        new_depth = depth_image_original
        if 'cx' not in self.cam_params:
            print("Not recieved intr yet, skipping frame")
            return
        if(new_depth.dtype == np.uint16):
            new_depth = new_depth / 1000.0
        h0, w0, _ = new_image.shape
        h1 = h0
        w1 = w0
        new_depth = cv2.resize(new_depth,(w1,h1))
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
        self.t_ += 1
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
            pts = points[i]
            clr = images[i]
            pts_array = pts.float().flatten().tolist()
            clr_array = clr.float().flatten().tolist()
            image_poses.points = pts_array
            image_poses.colors = clr_array
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
        image_pose_msg = ImagePose()
        image_pose_msg.img = self.cv_bridge_.cv2_to_compressed_imgmsg(cv_image_original)
        image_pose_msg.depth = self.cv_bridge_.cv2_to_imgmsg(depth_image_original)
        image_pose_msg.w = self.cam_params['w']
        image_pose_msg.h = self.cam_params['h']
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

        if not (-0.001 < pose[0] < 0.001 and -0.001 < pose[1] < 0.001 and -0.001 < pose[2] < 0.001): # if pose is not [0,0,0] then publish
            image_pose_msg.pose = Pose(position=Point(x=pose[0],y=pose[1],z=pose[2]),orientation=Quaternion(x=orient[0],y=orient[1],z=orient[2],w=orient[3]))  # Replace with your Pose message
            
            image_poses.image_poses = [image_pose_msg]
            self.publisher.publish(image_poses)
        self.curr_time = time.time()
        self.prev_time = self.curr_time

    def left_cam_intr_cb(self,left_msg):
        if 'w' in self.left_cam_params:
            return
        left_d_list = left_msg.d.tolist()
        if(len(left_d_list) > 0):
            left_k1 = left_msg.d[0]
            left_k2 = left_msg.d[1]
            left_k3 = left_msg.d[4]
        else:
            left_k1 = 0.0
            left_k2 = 0.0
            left_k3 = 0.0
        left_K = left_msg.k.reshape(3,3)
        left_params = {
            "w": left_msg.width,
            "h": left_msg.height,
            "fl_x": left_K[0,0],
            "fl_y": left_K[1,1],
            "cx": left_K[0,2],
            "cy": left_K[1,2],
            "k1": left_k1,
            "k2": left_k2,
            "k3": left_k3,
            "camera_model": "OPENCV",
        }
        self.left_cam_params |= left_params
        

    def right_cam_intr_cb(self,right_msg):
        if 'w' in self.right_cam_params:
            pass
        right_d_list = right_msg.d.tolist()
        if(len(right_d_list) > 0):
            right_k1 = right_msg.d[0]
            right_k2 = right_msg.d[1]
            right_k3 = right_msg.d[4]
        else:
            right_k1 = 0.0
            right_k2 = 0.0
            right_k3 = 0.0
        
        right_K = right_msg.k.reshape(3,3)
        right_params = {
            "w": right_msg.width,
            "h": right_msg.height,
            "fl_x": right_K[0,0],
            "fl_y": right_K[1,1],
            "cx": right_K[0,2],
            "cy": right_K[1,2],
            "k1": right_k1,
            "k2": right_k2,
            "k3": right_k3,
            "camera_model": "OPENCV",
        }
        self.right_cam_params |= right_params

    def cam_intr_cb(self,msg):
        if 'w' in self.cam_params:
            return
        d_list = msg.d.tolist()
        
        
        if(len(d_list) > 0):
            k1 = msg.d[0]
            k2 = msg.d[1]
            k3 = msg.d[4]
        else:
            k1 = 0.0
            k2 = 0.0
            k3 = 0.0

        

        
        K = msg.k.reshape(3,3)
        
        params = {
            "w": msg.width,
            "h": msg.height,
            "fl_x": K[0,0],
            "fl_y": K[1,1],
            "cx": K[0,2],
            "cy": K[1,2],
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "camera_model": "OPENCV",
        }
        
        
        self.frames = []
        self.cam_params |= params
        

    def image_callback_uncompressed(self, img_msg, depth_msg):
        realsense_sim_msg = ImagePose()
        cv_image = self.cv_bridge_.imgmsg_to_cv2(img_msg)  # Convert ROS Image message to OpenCV image
        img_msg = self.cv_bridge_.cv2_to_compressed_imgmsg(cv_image)

        realsense_sim_msg.img = img_msg
        realsense_sim_msg.depth = depth_msg
        imageposes_msg = ImagePoses()
        imageposes_msg.image_poses = [realsense_sim_msg]
        self.realsense_publisher.publish(imageposes_msg)
        return

    def doneCallback(self,msg):
        traj_est = self.droid.terminate(self.ros_image_stream(self.droid_args_.datapath, use_depth=False, stride=self.stride))
        print("#"*20 + " Results...")
    
        import evo
        from evo.core.trajectory import PoseTrajectory3D
        from evo.tools import file_interface
        from evo.core import sync
        import evo.main_ape as main_ape
        from evo.core.metrics import PoseRelation

        image_path = os.path.join(self.droid_args_.datapath, 'rgb')
        images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))[::self.stride]
        tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

        traj_est = PoseTrajectory3D(
            positions_xyz=traj_est[:,:3],
            orientations_quat_wxyz=traj_est[:,3:],
            timestamps=np.array(tstamps))

        gt_file = os.path.join(self.droid_args_.datapath, 'groundtruth.txt')
        traj_ref = file_interface.read_tum_trajectory_file(gt_file)

        traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

        result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
            pose_relation=PoseRelation.translation_part, align=True, correct_scale=False)

        print(result.stats)
        print("Did prime image stream")

    def ros_image_stream(self,datapath,use_depth=False,stride=1):
        fx, fy, cx, cy = np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()
        image_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::stride]
        depth_list = sorted(glob.glob(os.path.join(datapath, 'depth', '*.png')))[::stride]

        for t, (image_file, depth_file) in enumerate(zip(image_list, depth_list)):
            
            image = cv2.imread(image_file)
            depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 5000.0
            ros_image = self.cv_bridge_.cv2_to_imgmsg(image,encoding="passthrough")
            ros_image.header.frame_id = "droid_optical_frame"
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_depth = self.cv_bridge_.cv2_to_imgmsg(depth,encoding="passthrough")
            ros_depth.header.stamp = ros_image.header.stamp
            ros_depth.header.frame_id = "droid_optical_frame"
            ros_camera_info = CameraInfo()
            ros_camera_info.header.stamp = ros_image.header.stamp
            ros_camera_info.header.frame_id = "droid_optical_frame"
            ros_camera_info.k = [fx,0.0,cx,0.0,fy,cy,0.0,0.0,1.0]
            new_image = self.cv_bridge_.imgmsg_to_cv2(ros_image)
            new_depth = self.cv_bridge_.imgmsg_to_cv2(ros_depth)
            new_fx,_,new_cx,_,new_fy,new_cy,_,_,_ = ros_camera_info.k

            h0, w0, _ = new_image.shape
            h1 = h0
            w1 = w0
            new_image = cv2.resize(new_image, (w1, h1))
            new_image = new_image[:h1-h1%8, :w1-w1%8]
            new_image = torch.as_tensor(new_image).permute(2, 0, 1)
            
            new_depth = torch.as_tensor(new_depth)
            new_depth = F.interpolate(new_depth[None,None], (h1, w1)).squeeze()
            new_depth = new_depth[:h1-h1%8, :w1-w1%8]

            intrinsics = torch.as_tensor([new_fx, new_fy, new_cx, new_cy])
            intrinsics[0::2] *= (w1 / w0)
            intrinsics[1::2] *= (h1 / h0)
            if use_depth:
                yield t, new_image[None], new_depth, intrinsics

            else:
                yield t, new_image[None], intrinsics

    def fullImageCallback(self,ros_image,ros_depth,ros_camera_info):
        new_image = self.cv_bridge_.imgmsg_to_cv2(ros_image)
        new_depth = self.cv_bridge_.imgmsg_to_cv2(ros_depth)
        if(new_depth.dtype == np.uint16):
            new_depth = new_depth / 1000.0
        new_fx,_,new_cx,_,new_fy,new_cy,_,_,_ = ros_camera_info.k
        h0, w0, _ = new_image.shape
        h1 = h0
        w1 = w0
        new_depth = cv2.resize(new_depth,(w1,h1))
        new_image = cv2.resize(new_image, (w1, h1))
        new_image = new_image[:h1-h1%8, :w1-w1%8]
        new_image = torch.as_tensor(new_image).permute(2, 0, 1)
        
        new_depth = torch.as_tensor(new_depth)
        new_depth = F.interpolate(new_depth[None,None], (h1, w1)).squeeze()
        new_depth = new_depth[:h1-h1%8, :w1-w1%8]

        intrinsics = torch.as_tensor([new_fx, new_fy, new_cx, new_cy])
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
        self.t_ += 1
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
        
                # matrix_quat = R.from_matrix(matrix[:3,:3]).as_quat()
                # matrix_pos = matrix[:3,3]
                # t.transform.translation.x = matrix_pos[0]
                # t.transform.translation.y = matrix_pos[1]
                # t.transform.translation.z = matrix_pos[2]

                # t.transform.rotation.x = matrix_quat[0]
                # t.transform.rotation.y = matrix_quat[1]
                # t.transform.rotation.z = matrix_quat[2]
                # t.transform.rotation.w = matrix_quat[3]

                # print('matrix pos', matrix_pos)
                self.tf_broadcaster.sendTransform(ros_transform)
            return
        self.image_counter += 1
        pose = self.droid.video.poses[self.droid.video.counter.value-1].cpu().numpy()
        self.last_pose = pose
        



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