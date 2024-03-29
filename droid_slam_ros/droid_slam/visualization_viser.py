import torch
import cv2
import lietorch
import droid_backends
import time
import argparse
import numpy as np
import open3d as o3d

from lietorch import SE3
import geom.projective_ops as pops

import viser
import viser.transforms as vtf
import time

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

VISER_ALREADY_ADDED_INDS = set()

def white_balance(img):
    # from https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

@torch.no_grad()
def update_visualization(server: viser.ViserServer, video, dirty_index: torch.Tensor):
    video.dirty[dirty_index] = False
    # convert poses to 4x4 matrix
    poses = torch.index_select(video.poses, 0, dirty_index)
    disps = torch.index_select(video.disps, 0, dirty_index)
    Ps = SE3(poses).inv().matrix().cpu().numpy()

    images = torch.index_select(video.images, 0, dirty_index)
    images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
    points = droid_backends.iproj(SE3(poses).inv().data, disps, video.intrinsics[0]).cpu()

    thresh = droid_visualization.filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))
    
    count = droid_backends.depth_filter(
        video.poses, video.disps, video.intrinsics[0], dirty_index, thresh)

    count = count.cpu()
    disps = disps.cpu()
    masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

    for i in range(len(dirty_index)):
        pose = Ps[i]
        ix = dirty_index[i].item()

        ### add camera actor ###
        # camera_iter is whatever is the latest pred for the camera
        position = pose[:3, 3]
        camera_ix = server.add_camera_frustum(
            f'/camera_iter/camera_{ix}',
            fov=90,
            aspect=images[0].shape[1] / images[0].shape[0],
            scale=0.01,
            position=position,
            wxyz=vtf.SO3.from_matrix(pose[:3,:3]).wxyz
        )
        droid_visualization.cameras[ix] = camera_ix
        mask = masks[i].reshape(-1)
        pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
        clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
        
        ## add point actor ###
        # ... and pointcloud_iter is also the latest pred.
        point_actor_ix = server.add_point_cloud(
            f'/pointcloud_iter/pointcloud_{ix}',
            points=pts,
            colors=clr,
            point_size=0.01,
        )
        droid_visualization.points[ix] = point_actor_ix

        # I would just make a viser button that says "export current camera poses"
        # as well as the pointcloud. (and maybe the images are also cached somewhere...?)
        # Then, you can put the poses into a transforms.json + images file, 
        # Then this would probably be the easeiest way to verify
        #   ... if the BA poses are actually good for splats,
        #   ... and i guess you can also run colmap on the images and compare. but this is lower priority
        
        if ix not in VISER_ALREADY_ADDED_INDS:
            # *_first means the first ever prediction for the camera.
            server.add_camera_frustum(
                f'/camera_first/camera_{ix}',
                fov=90,
                aspect=images[0].shape[1] / images[0].shape[0],
                scale=0.01,
                position=position,
                wxyz=vtf.SO3.from_matrix(pose[:3,:3]).wxyz
            )
            server.add_point_cloud(
                f'/pointcloud_first/pointcloud_{ix}',
                points=pts,
                colors=clr,
                point_size=0.01,
            )
            VISER_ALREADY_ADDED_INDS.add(ix)


def droid_visualization(video, device="cuda:0"):
    """ DROID visualization frontend """

    torch.cuda.set_device(device)
    droid_visualization.video = video
    droid_visualization.cameras = {}
    droid_visualization.points = {}
    droid_visualization.warmup = 8
    droid_visualization.scale = 1.0
    droid_visualization.ix = 0

    droid_visualization.filter_thresh = 0.005

    # NOTE(cmk) removed the increase/decrease filter key callback
    # from the original visualization, because we never used it.
    # but we can add it back again via creating viser buttons.

    server = viser.ViserServer()

    while True:
        with video.get_lock():
            t = video.counter.value 
            dirty_index, = torch.where(video.dirty.clone())
            dirty_index = dirty_index

        if len(dirty_index) == 0:
            time.sleep(0.1)
            continue
        
        update_visualization(server, video, dirty_index)