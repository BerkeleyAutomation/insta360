import os
import sys
file_location = os.path.dirname(os.path.realpath(__file__))
droid_slam_location = file_location + '/../droid_slam'
sys.path.append(droid_slam_location)

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import time
import torch.nn.functional as F
from droid import Droid
import matplotlib.pyplot as plt


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(datapath, use_depth=False, stride=1):
    """ image generator """
    
    fx, fy, cx, cy = np.loadtxt(os.path.join(datapath, 'calibration.txt')).tolist()
    image_list = sorted(glob.glob(os.path.join(datapath, 'rgb', '*.png')))[::stride]
    depth_list = sorted(glob.glob(os.path.join(datapath, 'depth', '*.npy')))[::stride]
    total_image_stream = []
    for t, (image_file, depth_file) in enumerate(zip(image_list, depth_list)):
        
        image = cv2.imread(image_file)
        # image = image[61:404, 102:661,:]
        
        depth = np.load(depth_file)
        if(depth.dtype == np.uint16):
            depth = depth / 1000.0
        # image = cv2.resize(image,(424,240))
        height, width, _ = image.shape
        height = height - height % 32
        width = width - width % 32
        image = image[:height, :width, :3]
        h0, w0, _ = image.shape
        h1 = 456
        w1 = 855
        print("h1:", h1)
        print("w1:", w1)
        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)
        
        depth = torch.as_tensor(depth)
        depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
        depth = depth[:h1-h1%8, :w1-w1%8]

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)
        # cv2.imwrite('teehee.png',image.permute(1,2,0).cpu().detach().numpy())
        # cv2.imwrite('teehee2.png',np.round(((depth.cpu().detach().numpy() - np.min(depth.cpu().detach().numpy())) / (np.max(depth.cpu().detach().numpy())- np.min(depth.cpu().detach().numpy()))) * 255).astype(np.uint8))
        # import pdb
        # pdb.set_trace()
        if use_depth:
            total_image_stream.append((t,image[None],depth,intrinsics))
            #yield t, image[None], depth, intrinsics
        else:
            total_image_stream.append((t,image[None],intrinsics))
            #yield t, image[None], intrinsics
    return total_image_stream

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
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
    parser.add_argument("--depth", action="store_true")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')

    print("Running evaluation on {}".format(args.datapath))
    print(args)

    # this can usually be set to 2-3 except for "camera_shake" scenes
    # set to 2 for test scenes
    stride = 1

    tstamps = []
    iter_times = []
    total_image_stream = image_stream(args.datapath,use_depth=True,stride=stride)
    
    for (t, image, depth, intrinsics) in tqdm(total_image_stream):
        start_time = time.time()
        if not args.disable_vis:
            show_image(image[0])
        if t == 0:
            args.image_size = [image.shape[2], image.shape[3]]
            print("the image size is", args.image_size)
            droid_obj = Droid(args)
        
        droid_obj.track(t, image, depth, intrinsics=intrinsics)
        print("t: " + str(t))
        end_time = time.time()
        # if(t == 80):
        #     import pdb
        #     pdb.set_trace()
        #     print("Global bundle adjust")
        #     droid.globalBundleAdjust()
        #     import pdb
        #     pdb.set_trace()
        iter_times.append(end_time - start_time)
    
    traj_est = droid_obj.terminate(image_stream(args.datapath, use_depth=False, stride=stride))

    iter_time_np = np.array(iter_times)
    print("Mean time: " + str(np.mean(iter_time_np)))

