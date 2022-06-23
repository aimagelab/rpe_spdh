import json
from collections import defaultdict
from pathlib import Path

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.data_augmentation import augment_3d
from utils.depth_utils import depthmap2pointcloud
from utils.depth_utils import depthmap2points
from utils.preprocessing import apply_depth_normalization_16bit_image
from utils.utils import heatmap_from_kpoints_array, gkern

JOINTS_NAMES = [
    'head',
    'right_upper_shoulder',
    'right_lower_shoulder',
    'right_upper_elbow',
    'right_lower_elbow',
    'right_upper_forearm',
    'right_lower_forearm',
    'right_wrist',
    'left_upper_shoulder',
    'left_lower_shoulder',
    'left_upper_elbow',
    'left_lower_elbow',
    'left_upper_forearm',
    'left_lower_forearm',
    'left_wrist',
    'base',
    # 'r_gripper_r_finger',
    # 'r_gripper_l_finger',
    # 'l_gripper_r_finger',
    # 'l_gripper_l_finger',
]


def init_worker(worker_id):
    np.random.seed(torch.initial_seed() % 2 ** 32)


class BaxterJointsSynthDataset(Dataset):
    def __init__(self, dataset_dir: Path, run: list, init_mode: str = 'train', img_type: str = 'D',
                 img_size: tuple = (384, 216), sigma: int = 4., norm_type: str = 'mean_std',
                 network_input: str = 'D', network_output: str = 'H', network_task: str = '3d_RPE',
                 depth_range: tuple = (500, 3380, 15), depth_range_type: str = 'normal', aug_type: str = '3d',
                 aug_mode: bool = True, noise: bool = False, demo: bool = False):
        """Load Baxter robot synthetic dataset

        Parameters
        ----------
        dataset_dir: Path
            Dataset path.
        run: list
            Synthetic run to load (0, 1, or both).
        init_mode: str
            Loading mode (train -> train/val set, test -> test set).
        img_type: str
            Type of image to load (RGB, D or RGBD).
        img_size: str
            Image dimensions to which resize dataset images.
        sigma: int
            Variance of ground truth gaussian heatmaps.
        norm_type: str
            Type of normalization (min_max, mean_std supported).
        network_input: str
            Input type of the network (D, RGB, RGBD, XYZ).
        network_task: str
            Task the network should solve (2d or 3d robot pose estimation).
        ref_frame: str
            Reference frame of gt joints (right or left handed).
        surrounding_removals: bool
            Activate or deactivate removal of surrounding walls/objects.
        aug_type: str
            Type of data augmentation on the input (2d or 3d).
        aug_mode: bool
            Activate or deactivate augmentation during training.
        noise: bool
            If true, adds random pepper noise to data augmentation during training.
        demo: bool
            Useful for loading a portion of the dataset when debugging.
        """
        assert init_mode in ['train', 'test']

        self.dataset_dir = Path(dataset_dir)
        self.run = run
        self._mode = init_mode
        self.img_type = img_type
        self.img_size = img_size
        self.sigma = sigma
        self.norm_type = norm_type
        self.network_input = network_input
        self.network_output = network_output
        self.depth_range = depth_range
        self.depth_range_type = depth_range_type
        self.network_task = network_task
        self.aug_type = aug_type
        self._aug_mode = aug_mode
        self.aug_mode = aug_mode
        self.noise = noise
        self.demo = demo
        self.data = self.load_data()

    def __len__(self):
        return len(self.data[self.mode])

    def __getitem__(self, idx):
        sample = self.data[self.mode][idx].copy()

        # image loading (depth and/or RGB)
        depth16_img = cv2.imread(sample['depth16_file'], cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth16_img = cv2.resize(depth16_img, tuple(self.img_size), interpolation=cv2.INTER_NEAREST)
        # RGB and depth scale are the same since they are aligned
        scale_x, scale_y = depth16_img.shape[1] / 1920, depth16_img.shape[0] / 1080
        # image size divided by 32 should be an even value (for SH network)
        depth16_img = depth16_img[12:-12, :]
        if "RGB" in self.img_type:
            rgb_img = cv2.imread(sample['rgb_file'], cv2.IMREAD_COLOR).astype(np.float32)
            rgb_img = cv2.resize(rgb_img, tuple(self.img_size), interpolation=cv2.INTER_NEAREST)
            rgb_img = rgb_img[12:-12, ...]

        # adapt depth image to "padding" depth range type
        if self.depth_range_type == 'padding':
            Z_min, Z_max, dZ = self.depth_range
            new_img_h = (Z_max - Z_min) // dZ
            padding = int(np.abs(depth16_img.shape[0] - new_img_h) // 2)
            depth16_img = cv2.copyMakeBorder(depth16_img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, 0)

        # initialize intrinsic matrix
        camera_params = json.load(open(str(self.dataset_dir / "camera_params.json"), 'r'))
        fx, fy = camera_params['rgb']['fx'] * scale_x, camera_params['rgb']['fy'] * scale_y
        cx, cy = depth16_img.shape[1] / 2, depth16_img.shape[0] / 2  # principal point is at the center of the image
        intrinsic = np.asarray([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])

        # 2D and 3D joints loading
        joints_2D = np.zeros((sample['joints'].shape[0], 2))
        joints_3D_Z = np.zeros((sample['joints'].shape[0], 3))
        for n, joint in enumerate(sample['joints']):
            # point3d has left handed reference frame (Y-axis upwards)
            point3d = np.asarray([joint[1], joint[2], -joint[0]])  # [x, y, z] -> [y, z, -x] as in ROS synchronization
            # 5.5cm translation between depth and RGB cameras aligned
            point3d[0] = point3d[0] + camera_params['tx_depth_to_rgb']
            joints_3D_Z[n] = point3d.copy()
            point3d[1] *= -1  # invert Y axis direction for 2D reprojection
            u, v, w = (intrinsic @ point3d[:, None]).T.squeeze(0)
            u = u / w
            v = v / w
            joints_2D[n] = [u, v]

        # apply 2d or 3d augmentation
        if self.aug_mode:
            if self.aug_type == '2d':
                if self.noise:
                    seq = iaa.Sequential([
                        iaa.Pepper(p=(0.1, 0.15)),
                        iaa.Affine(
                            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                            rotate=(-10, 10),
                        )
                    ]).to_deterministic()
                else:
                    seq = iaa.Sequential([
                        iaa.Affine(
                            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                            rotate=(-10, 10),
                        )
                    ]).to_deterministic()
                kps_tmp = ia.KeypointsOnImage([ia.Keypoint(x, y) for x, y in joints_2D], shape=depth16_img.shape)
                depth16_img, joints_2D = seq.augment(image=depth16_img, keypoints=kps_tmp)
                joints_2D = joints_2D.to_xy_array()
                if "RGB" in self.img_type:
                    rgb_img = seq.augment(image=rgb_img)
            else:
                points = depthmap2pointcloud(depth16_img / 1000, fx=fx, fy=fy, cx=intrinsic[0, 2],
                                             cy=intrinsic[1, 2])
                points = points[points[:, 2] != 0]
                depth16_img, joints_3D_Z = augment_3d(intrinsic, points, depth16_img, joints_3D_Z)
                depth16_img = depth16_img * 1000
                if self.noise:
                    seq = iaa.Sequential([
                        iaa.Pepper(p=(0.1, 0.15)),
                    ]).to_deterministic()
                    depth16_img = seq.augment(image=depth16_img)
                point3d = joints_3D_Z.copy()
                point3d[..., 1] *= -1  # invert Y axis direction for 2D reprojection
                joints_2D = (intrinsic @ point3d[..., None]).squeeze(-1)
                joints_2D = (joints_2D / joints_2D[:, -1:])[:, :2]

        # create visible depth map
        depth16_img_vis = ((depth16_img * 255) / depth16_img.max()).astype(np.uint8)[..., None]
        depth16_img_vis = np.concatenate([depth16_img_vis, depth16_img_vis, depth16_img_vis], -1)

        '''
        DEBUG = False
        if DEBUG:
            import matplotlib.pyplot as plt
            depth16_img_vis_copy = depth16_img_vis.copy()
            if "RGB" in self.img_type:
                rgb_img_copy = rgb_img.copy()
                for n, joint_2D in enumerate(joints_2D):
                    print(JOINTS_NAMES[n])
                    cv2.circle(rgb_img_copy, (int(joint_2D[0]), int(joint_2D[1])), 10, (255, 0, 0), -1)
                plt.imshow(rgb_img_copy[..., ::-1].astype(np.uint8))
                plt.show()
            for n, joint_2D in enumerate(joints_2D):
                print(JOINTS_NAMES[n])
                cv2.circle(depth16_img_vis_copy, (int(joint_2D[0]), int(joint_2D[1])), 2, (255, 0, 0), -1)
            plt.imshow(depth16_img_vis_copy[..., ::-1].astype(np.uint8))
            plt.show()
        '''

        # get 3D joints with Z value from depth map or from ground truth values
        z_values = depth16_img.copy() / 1000  # depth values from millimeters to meters
        joints_2D_homo = np.ones((joints_2D.shape[0], 3))
        joints_2D_homo[:, :2] = joints_2D
        XY_rw = (np.linalg.inv(intrinsic) @ joints_2D_homo[..., None]).squeeze(-1)[:, :2]
        joints_3D_depth = np.ones((joints_2D.shape[0], 3), dtype=np.float32)
        joints_3D_depth[:, :2] = XY_rw
        depth_coords = []
        for joint in joints_2D:
            x, y = joint[0], joint[1]
            if x < 0:
                x = 0
            if x > z_values.shape[1] - 1:
                x = z_values.shape[1] - 1
            if y < 0:
                y = 0
            if y > z_values.shape[0] - 1:
                y = z_values.shape[0] - 1
            depth_coords.append([x, y])
        depth_coords = np.array(depth_coords)
        z = z_values[depth_coords[:, 1].astype(int), depth_coords[:, 0].astype(int)]
        joints_3D_depth *= z[..., None]
        if self.aug_type == '2d':
            joints_3D_Z_ = np.ones((joints_2D.shape[0], 3), dtype=np.float32)
            joints_3D_Z_[:, :2] = XY_rw
            z = joints_3D_Z[:, -1:]
            joints_3D_Z_ *= z
            joints_3D_Z = joints_3D_Z_
            joints_3D_Z[..., 1] *= -1

        # compute XYZ image
        if "XYZ" in self.network_input:
            xyz_img = depthmap2points(depth16_img / 1000, fx=fx, fy=fy, cx=intrinsic[0, 2], cy=intrinsic[1, 2])
            xyz_img[..., 0] = xyz_img[..., 0] / 3.
            xyz_img[..., 1] = xyz_img[..., 1] / 2.
            xyz_img[..., 2] = xyz_img[..., 2] / 5.
            xyz_img[depth16_img == 0] = 0

        # depth map and keypoints normalization
        depth16_img = apply_depth_normalization_16bit_image(depth16_img, self.norm_type)
        if self.network_task == '2d_RPE':
            joints_2D[:, 0] = joints_2D[:, 0] / depth16_img.shape[1]
            joints_2D[:, 1] = joints_2D[:, 1] / depth16_img.shape[0]
        else:
            Z_min, Z_max, dZ = self.depth_range
            sigma_mm = 50

            # UV heatmaps
            x, y = np.meshgrid(np.arange(depth16_img.shape[1]), np.arange(depth16_img.shape[0]))
            heatmaps_uv = np.zeros((joints_2D.shape[0], depth16_img.shape[0], depth16_img.shape[1]))
            for n, (p, P) in enumerate(zip(joints_2D, joints_3D_Z)):
                P = P.copy() * 1000
                # compute distances (px) from point
                dst = np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2)
                # convert sigma from mm to pixel
                sigma_pixel = sigma_mm * intrinsic[0, 0] / P[2]
                # compute the heatmap
                mu = 0.
                heatmaps_uv[n] = np.exp(-((dst - mu) ** 2 / (2.0 * sigma_pixel ** 2)))

            # UZ heatmaps
            heatmaps_uz = np.zeros((joints_2D.shape[0], depth16_img.shape[0], depth16_img.shape[1]))
            for n, P in enumerate(joints_3D_Z):
                P = P.copy() * 1000
                # compute X for each value of x (u) at each slice z
                x = np.arange(depth16_img.shape[1])  # [u]
                x_unsqueezed = x[None, :]  # [1, u]
                z = np.arange(Z_min, Z_max, dZ)  # [Z / dZ]
                z_unsqueezed = z[:, None]  # [Z / dZ, 1]
                fx, _, cx = intrinsic[0]
                X = (x_unsqueezed - cx) * z_unsqueezed / fx  # [Z / dZ, u]
                # compute distances (mm) from point
                dst_mm = np.sqrt((X - P[0]) ** 2 + (z_unsqueezed - P[2]) ** 2)  # [Z / dZ, u]
                # compute heatmap
                heatmaps_uz[n] = np.exp(-(dst_mm ** 2 / (2.0 * sigma_mm ** 2)))  # [Z / dZ, u]

            heatmaps_25d = np.concatenate((heatmaps_uv, heatmaps_uz), axis=0)

        # keypoint to heatmap transform
        if self.network_task == '2d_RPE':
            heatmaps = heatmap_from_kpoints_array(kpoints_array=joints_2D, shape=depth16_img.shape[:2],
                                                  sigma=self.sigma)

        # mean and std for 2D and 3D joints
        stats = np.atleast_1d(np.load(str(self.dataset_dir / "mean_std_stats.npy"), allow_pickle=True))[0]

        output = {
            'depth16': torch.from_numpy(depth16_img[None, ...]),
            'depth16vis': depth16_img_vis,
            'z_values': z_values,
            'K_depth': intrinsic,

            'joints_2D_depth': joints_2D,
            'joints_2D_depth_mean': stats['joints2d_mean'],
            'joints_2D_depth_std': stats['joints2d_std'],

            'joints_3D_Z': joints_3D_Z,
            'joints_3D_Z_mean': stats['joints3d_mean'],
            'joints_3D_Z_std': stats['joints3d_std']
        }
        if "RGB" in self.network_input:
            output['rgb'] = torch.from_numpy(((rgb_img.astype(np.float32) - 128.) / 128.).transpose(2, 0, 1))
            output['rgb_vis'] = rgb_img.copy()
            output['K_rgb'] = intrinsic
        if self.network_task == '2d_RPE':
            output['joints_3D_depth'] = joints_3D_depth
            output['heatmap_depth'] = torch.from_numpy(heatmaps.astype(np.float32).transpose(2, 0, 1))
            if "RGB" in self.network_input:
                output['heatmap_rgb'] = torch.from_numpy(heatmaps.astype(np.float32).transpose(2, 0, 1))
        else:
            output['heatmap_25d'] = torch.from_numpy(heatmaps_25d.astype(np.float32))
        if "XYZ" in self.network_input:
            output['xyz_img'] = torch.from_numpy(xyz_img.transpose(2, 0, 1))
        if "3DH" in self.network_output:
            h_depth, h_height, h_width = 64, 48, 96
            factor_x, factor_y = h_width / 384, h_height / 192
            sigma = 4
            g = 17
            gaussian_patch = gkern(w=g, h=g, d=g, center=(g // 2, g // 2, g // 2), s=sigma,
                                   device='cpu')
            heatmap_3d = torch.zeros((len(joints_2D), h_depth, h_height, h_width)).to('cpu')
            for n, joint2d in enumerate(joints_2D):
                joint = joint2d.copy()
                joint[0] *= factor_x
                joint[1] *= factor_y
                cam_dist = np.sqrt(joints_3D_Z[n, 0] ** 2 + joints_3D_Z[n, 1] ** 2 + joints_3D_Z[n, 2] ** 2)
                cam_dist = cam_dist * ((h_depth - 1) / 5.)

                center = [
                    int(round(joint[0])),
                    int(round(joint[1])),
                    int(round(cam_dist))
                ]

                center = center[::-1]

                xa, ya, za = max(0, center[2] - g // 2), \
                             max(0, center[1] - g // 2), \
                             max(0, center[0] - g // 2)
                xb, yb, zb = min(center[2] + g // 2, h_width - 1), \
                             min(center[1] + g // 2, h_height - 1), \
                             min(center[0] + g // 2, h_depth - 1)
                hg, wg, dg = (yb - ya) + 1, (xb - xa) + 1, (zb - za) + 1

                gxa, gya, gza = 0, 0, 0
                gxb, gyb, gzb = g - 1, g - 1, g - 1

                if center[2] - g // 2 < 0:
                    gxa = -(center[2] - g // 2)
                if center[1] - g // 2 < 0:
                    gya = -(center[1] - g // 2)
                if center[0] - g // 2 < 0:
                    gza = -(center[0] - g // 2)
                if center[2] + g // 2 > (h_width - 1):
                    gxb = wg - 1
                if center[1] + g // 2 > (h_height - 1):
                    gyb = hg - 1
                if center[0] + g // 2 > (h_depth - 1):
                    gzb = dg - 1

                heatmap_3d[n, za:zb + 1, ya:yb + 1, xa:xb + 1] = torch.max(
                    torch.cat(tuple([
                        heatmap_3d[n, za:zb + 1, ya:yb + 1, xa:xb + 1].unsqueeze(0),
                        gaussian_patch[gza:gzb + 1, gya:gyb + 1, gxa:gxb + 1].unsqueeze(0)
                    ])), 0)[0]
            output["heatmap_3d"] = heatmap_3d

        return output

    def train(self):
        self.mode = 'train'
        if self._aug_mode:
            self.aug_mode = True

    def eval(self):
        self.mode = 'val'
        self.aug_mode = False

    def test(self):
        self.mode = 'test'
        self.aug_mode = False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        assert value in ['train', 'val', 'test']
        self._mode = value

    def load_data(self):
        if self._mode == 'train':
            splits = ['train', 'val']
        else:
            splits = ['test']
        data = defaultdict(list)
        for split in splits:
            iter = 0
            for run in self.run:
                with (self.dataset_dir / 'splits' / f'{split}_rgb_run_{run}.txt').open('r') as fd:
                    rgb_files = fd.read().splitlines()
                if self.demo:
                    rgb_files = rgb_files[:800]
                for rgb_file in tqdm(rgb_files, f'Loading {split} set run {run}'):
                    info = rgb_file.split('/')
                    camera = info[-4]
                    seq = info[-3]
                    sub_seq = info[-2]
                    img_name = Path(rgb_file).stem

                    # camera, depth, joints and picknplace file and data
                    depth8_file = Path(rgb_file.replace('rgb', 'depth8_registered'))
                    depth8_img_name = f"{img_name.replace('color', 'depth')}.png"
                    depth8_file = str(depth8_file.parents[0] / f'{depth8_img_name}')
                    depth16_file = Path(rgb_file.replace('rgb', 'depth16_registered'))
                    depth16_img_name = f"{img_name.replace('color', 'depth')}_16bit.png"
                    depth16_file = str(depth16_file.parents[0] / f'{depth16_img_name}')

                    # camera_file = str(self.dataset_dir / f'cameras_run_{run}' / f'{camera}' / f'{seq}.json')
                    # with open(camera_file, 'r') as fd:
                    #     camera_pos_dict = json.load(fd)
                    # camera_position_dict = camera_pos_dict['position']
                    # camera_orientation_dict = camera_pos_dict['orientation']
                    # camera_pos = np.array([camera_position_dict['x'], camera_position_dict['y'],
                    #                        camera_position_dict['z'], camera_orientation_dict['x'],
                    #                        camera_orientation_dict['y'], camera_orientation_dict['z'],
                    #                        camera_orientation_dict['w']], dtype=np.float32)

                    joints_json_name = f"{img_name.replace('_color', '')}.json"
                    joints_file = str(self.dataset_dir / f'joints_run_{run}' / f'{seq}' / f'{sub_seq}' /
                                      f'{joints_json_name}')
                    with open(joints_file, 'r') as fd:
                        joints_pos_dict = json.load(fd)['joints'][f'{camera}']
                    joints_pos = np.zeros((len(JOINTS_NAMES), 7), dtype=np.float32)
                    for n, (k, joint_pos_dict) in enumerate(joints_pos_dict.items()):
                        if k in JOINTS_NAMES:
                            idx = JOINTS_NAMES.index(k)
                            joints_pos[idx] = [joint_pos_dict['translation.x'], joint_pos_dict['translation.y'],
                                               joint_pos_dict['translation.z'], joint_pos_dict['rotation.x'],
                                               joint_pos_dict['rotation.y'], joint_pos_dict['rotation.z'],
                                               joint_pos_dict['rotation.w']]

                    # picknplace_file = str(self.dataset_dir / f'picknplace_run_{run}' / f'{seq}' / f'{sub_seq}.json')
                    # with open(picknplace_file, 'r') as fd:
                    #     picknplace_pos_dict = json.load(fd)
                    # pick_position_dict = picknplace_pos_dict['pick']['position']
                    # pick_orientation_dict = picknplace_pos_dict['pick']['orientation']
                    # pick_pos = np.array([pick_position_dict['x'], pick_position_dict['y'], pick_position_dict['z'],
                    #                      pick_orientation_dict['w'], pick_orientation_dict['x'],
                    #                      pick_orientation_dict['y'], pick_orientation_dict['z']], dtype=np.float32)
                    # place_position_dict = picknplace_pos_dict['place']['position']
                    # place_orientation_dict = picknplace_pos_dict['place']['orientation']
                    # place_pos = np.array([place_position_dict['x'], place_position_dict['y'], place_position_dict['z'],
                    #                       place_orientation_dict['x'], place_orientation_dict['y'],
                    #                       place_orientation_dict['z'], place_orientation_dict['w']], dtype=np.float32)

                    iter += 1

                    sample = {
                        'rgb_file': rgb_file,
                        'depth8_file': depth8_file,
                        'depth16_file': depth16_file,
                        'joints': joints_pos,             # [tx, ty, tz, qw, qx, qy, qz]

                        # 'camera': camera_pos,           # [tx, ty, tz, qw, qx, qy, qz]
                        # 'pick': pick_pos,               # [tx, ty, tz, qw, qx, qy, qz]
                        # 'place': place_pos              # [tx, ty, tz, qw, qx, qy, qz]
                    }

                    data[split].append(sample)

        return data
