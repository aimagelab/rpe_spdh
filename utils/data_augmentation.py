import random

import numpy as np

from utils.depth_utils import pointcloud2depthmap, hole_filling


def augment_3d(depth_intrinsic, points, depth16_img, joints_3D_Z):
    if len(points) > 0:
        points_mean = points.mean(axis=0)
        points -= points_mean
        joints_3D_Z -= points_mean

        p_rot = random.random()
        if p_rot < 1/3:
            # rotation around x axis
            angle = (random.random() * 2 - 1) * 5
            a = angle / 180 * np.pi
            rot = np.asarray([
                [1, 0, 0],
                [0, np.cos(a), -np.sin(a)],
                [0, np.sin(a), np.cos(a)],
            ])  # x axis
        elif p_rot >= 2/3:
            # rotation around y axis
            angle = (random.random() * 2 - 1) * 5
            a = angle / 180 * np.pi
            rot = np.asarray([
                [np.cos(a), 0, np.sin(a)],
                [0, 1, 0],
                [-np.sin(a), 0, np.cos(a)],
            ])  # y axis
        else:
            # rotation around z axis
            angle = (random.random() * 2 - 1) * 0
            a = angle / 180 * np.pi
            rot = np.asarray([
                [np.cos(a), -np.sin(a), 0],
                [np.sin(a), np.cos(a), 0],
                [0, 0, 1],
            ])  # z axis
        points = points @ rot.T
        joints_3D_Z = joints_3D_Z @ rot.T

        p_tr = random.random()
        if p_tr < 1/3:
            # translation over x axis
            tr_x = (random.random() * 2 - 1) * 0.08
            tr = np.array([tr_x, 0, 0])
        elif p_tr >= 2/3:
            # translation over y axis
            tr_y = (random.random() * 2 - 1) * 0
            tr = np.array([0, tr_y, 0])
        else:
            # translation over z axis
            tr_z = (random.random() * 2 - 1) * 0.08
            tr = np.array([0, 0, tr_z])
        points = points + tr
        joints_3D_Z = joints_3D_Z + tr

        # from pointcloud to depthmap
        points += points_mean
        joints_3D_Z += points_mean
        depth16_img = pointcloud2depthmap(points, depth16_img.shape[1], depth16_img.shape[0],
                                          fx=depth_intrinsic[0, 0],
                                          fy=depth_intrinsic[1, 1],
                                          cx=depth_intrinsic[0, 2],
                                          cy=depth_intrinsic[1, 2]).astype(depth16_img.dtype)
        depth16_img = hole_filling(depth16_img, kernel_size=2)

    return depth16_img, joints_3D_Z