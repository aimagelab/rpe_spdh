import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel as DP

from datasets.BaxterRealDataLoader import BaxterJointsRealDataset, JOINTS_NAMES as real_joints_name
from datasets.BaxterSynthDataLoader import BaxterJointsSynthDataset, JOINTS_NAMES as synth_joints_name
from networks.HRNet.model import HRNet
from networks.stacked_hourglass.model import HourglassNet
from utils.configer import Configer


class RPETrainer:
    def __init__(self, cfg, DEBUG):
        cfg_dict = {}
        for k, v in cfg.args.items():
            cfg_dict[k] = v
        for k, v in cfg.params.items():
            cfg_dict[k] = v
        self.cfg = cfg_dict
        self.DEBUG = DEBUG

        # set device and workers
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        if self.DEBUG:
            self.num_workers = 0
        else:
            self.num_workers = 4

        self.test()

    def init_worker(self, worker_id):
        np.random.seed(torch.initial_seed() % 2 ** 32)

    def init_dataset(self):
        if self.cfg["dataset"] == "synthetic":
            self.JOINTS_NAMES = synth_joints_name
        else:
            self.JOINTS_NAMES = real_joints_name
        if "depth_range" in self.cfg["data"].keys():
            depth_range = self.cfg["data"]["depth_range"]
        else:
            depth_range = [500, 3380, 15]
        if self.cfg["dataset"] == "synthetic":
            self.dataset = BaxterJointsSynthDataset(dataset_dir=self.cfg["data"]["data_path"],
                                                    run=self.cfg["data"]["run"],
                                                    init_mode="test", noise=self.cfg['noise'],
                                                    img_type=self.cfg["data"]["type"],
                                                    img_size=tuple(self.cfg["data"]["image_size"]),
                                                    sigma=self.cfg["data"]["sigma"],
                                                    norm_type=self.cfg["data"]["norm_type"],
                                                    network_input=self.cfg["network"]["input_type"],
                                                    network_output=self.cfg["network"]["output_type"],
                                                    network_task=self.cfg["network"]["task"],
                                                    depth_range=depth_range,
                                                    depth_range_type=self.cfg["data"]["depth_range_type"],
                                                    aug_type=self.cfg["data"]["aug_type"],
                                                    aug_mode=False,
                                                    demo=self.cfg['demo'])
        else:
            self.dataset = BaxterJointsRealDataset(dataset_dir=self.cfg["data"]["data_path"],
                                                   init_mode="test",
                                                   img_type=self.cfg["data"]["type"],
                                                   img_size=tuple(self.cfg["data"]["image_size"]),
                                                   sigma=self.cfg["data"]["sigma"],
                                                   norm_type=self.cfg["data"]["norm_type"],
                                                   network_input=self.cfg["network"]["input_type"],
                                                   network_output=self.cfg["network"]["output_type"],
                                                   network_task=self.cfg["network"]["task"],
                                                   depth_range=depth_range,
                                                   depth_range_type=self.cfg["data"]["depth_range_type"],
                                                   demo=self.cfg['demo'])
        self.dataloader = {
            "test": torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=False,
                                                num_workers=self.num_workers, worker_init_fn=self.init_worker,
                                                drop_last=False)
        }

    def init_model(self):
        if self.cfg["network"]["input_type"] == "D":
            inchannels = 1
        elif self.cfg["network"]["input_type"] == "XYZ":
            inchannels = 3
        elif self.cfg["network"]["input_type"] == "RGB":
            inchannels = 3
        elif self.cfg["network"]["input_type"] == "RGBD":
            inchannels = 4
        else:
            raise ValueError

        if self.cfg["network"]["name"] == "stacked_hourglass":
            self.model = DP(HourglassNet(inchannels=inchannels,
                                         num_stacks=self.cfg["network"]["stacks"],
                                         num_blocks=self.cfg["network"]["blocks"],
                                         num_joints=self.cfg["data"]["n_joints"]),
                            device_ids=[0]
                            )
        elif self.cfg["network"]["name"] == "hrnet":
            self.model = DP(HRNet(inchannels=inchannels,
                                  c=self.cfg["network"]["conv_width"],
                                  num_joints=self.cfg["data"]["n_joints"],
                                  bn_momentum=0.1),
                            device_ids=[0]
                            )
        else:
            raise ValueError
        self.model.to(self.device)

        self.starting_epoch = 0
        self.global_iter = 0
        self.avg_AP_scores = []

        if self.cfg["resume"] != "":
            self.load_model_weights(self.cfg["resume"])

    def load_model_weights(self, weights_dir):
        weights_dir = Path(weights_dir)
        if weights_dir.is_dir():
            weights_dir = weights_dir / 'checkpoint.pth'

        print(f'restoring checkpoint {str(weights_dir)}')

        self.checkpoint = torch.load(str(weights_dir), map_location=self.device)

        self.starting_epoch = self.checkpoint["epoch"]
        if "global_iter" in self.checkpoint:
            self.global_iter = self.checkpoint["global_iter"]
        if "avg_AP" in self.checkpoint:
            self.avg_AP_scores = self.checkpoint["avg_AP"]

        ret = self.model.load_state_dict(self.checkpoint["model"], strict=True)
        print(f'restored "{self.cfg["network"]["name"]}" model. Key errors:')
        print(ret)

        for k, v in self.checkpoint["config"].items():
            not_change = ["results_dir", "resume", "data", "demo", "dataset", "device", "phase", "metrics", "network"]
            if k not in not_change:
                self.cfg[k] = v

    def run_epoch(self, phase):
        phase_str = 'Test'

        print(f'#### {phase_str} ####')

        test_joints3D_avg_precision = []
        ADD = []
        ADD_no_sqrt = []
        thresholds = np.arange(0.02, 0.11, 0.01)

        for iter, batch in enumerate(self.dataloader[phase]):
            start_iter = time.time()
            start_iter_datetime = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))

            if self.cfg["network"]["input_type"] == 'D':
                input_tensor = batch['depth16'].to(self.device).float()
            elif self.cfg["network"]["input_type"] == 'XYZ':
                input_tensor = batch['xyz_img'].to(self.device).float()
            else:
                raise ValueError
            heatmaps_gt = batch['heatmap_25d'].to(self.device).float()
            joints_3d_gt = batch['joints_3D_Z'].numpy()[0]
            input_K = batch['K_depth'].to(self.device).float()

            with torch.no_grad():
                outputs = self.model(input_tensor)
                b, c, h, w = heatmaps_gt.size()
                if self.cfg["network"]["name"] == "stacked_hourglass":
                    heatmap_pred = outputs['heatmaps']
                    heatmap_pred = (F.interpolate(heatmap_pred[-1], (h, w), mode='bicubic',
                                                  align_corners=False) + 1) / 2.
                else:
                    heatmap_pred = outputs
                    heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic',
                                                  align_corners=False) + 1) / 2.

                B, C, H, W = heatmap_pred.shape
                joints_3d_pred = np.ones((B, C // 2, 3))
                joints_3d_pred[:, :, :2] = np.array([np.array(
                    [np.unravel_index(np.argmax(heatmap_pred[b, i, :, :].detach().cpu().numpy(),
                                                axis=None), heatmap_pred[b, i, :, :].shape)
                     for i in range(0, c // 2)])
                    for b in range(B)])[..., ::-1]
                # add Z coordinate from UZ heatmap
                z = np.array([np.array(
                    [np.unravel_index(np.argmax(heatmap_pred[b, i, :, :].detach().cpu().numpy(),
                                                axis=None), heatmap_pred[b, i, :, :].shape)
                     for i in range(c // 2, c)])
                    for b in range(heatmap_pred.shape[0])])[..., :1]
                Z_min, _, dZ = self.cfg["data"]["depth_range"]
                z = ((z * dZ) + Z_min) / 1000
                z = z.squeeze(-1)

                # convert 2D predicted joints to 3D coordinate multiplying by inverse intrinsic matrix
                inv_intrinsic = torch.inverse(input_K).unsqueeze(1).repeat(1,
                                                                           joints_3d_pred.shape[1],
                                                                           1,
                                                                           1).detach().cpu().numpy()
                joints_3d_pred = (inv_intrinsic @ joints_3d_pred[:, :, :, None]).squeeze(-1)
                joints_3d_pred *= z[..., None]
                joints_3d_pred[:, :, 1] *= -1  # invert Y axis for left-handed reference frame
                joints_3d_pred = joints_3d_pred[0]

                joints_pred, joints_gt = joints_3d_pred[None, ...], joints_3d_gt[None, ...]
                dist_3D = np.sqrt(np.sum((joints_pred - joints_gt) ** 2, axis=-1))
                dist_3D_no_sqrt = np.sum(np.abs(joints_pred - joints_gt), axis=-1)
                ADD.append(dist_3D.mean())
                ADD_no_sqrt.append(dist_3D_no_sqrt.mean())
                joints3D_avg_precision = []
                for thres in thresholds:
                    avg_AP = np.sum(dist_3D < thres, axis=-1) / joints_pred.shape[1]
                    joints3D_avg_precision.append(avg_AP)
                test_joints3D_avg_precision.append(joints3D_avg_precision)

            end_iter = time.time()
            print(f'[{end_iter - start_iter:.2f}s/it] - '
                  f'[from {start_iter_datetime} to {str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))}]')

            self.global_iter += 1

        ADD_avg = np.mean(ADD)
        ADD_std = np.std(ADD)
        ADD_no_sqrt_avg = np.mean(ADD_no_sqrt)
        ADD_no_sqrt_std = np.std(ADD_no_sqrt)
        print(f"ADD:\n{ADD_avg:.5f} +- {ADD_std:.5f} mm")
        print(f"ADD no sqrt:\n{ADD_no_sqrt_avg:.5f} +- {ADD_no_sqrt_std:.5f} mm")

        avg_avg_precision = np.round(np.mean(test_joints3D_avg_precision, axis=0) * 100, 2)
        avg_precision_score_str = ''
        for thres, val in zip(np.arange(0.02, 0.11, 0.01), avg_avg_precision):
            avg_precision_score_str += f'THRESHOLD {thres:.2f} | AP = {val[0]}%\n'
        print(f'Mean avg precision:\n{avg_precision_score_str}')

        return

    def test(self):
        self.init_dataset()
        self.init_model()

        self.global_iter = 0

        self.model.eval()
        self.dataset.test()
        self.run_epoch(phase="test")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='path to training configuration')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint for resuming training')
    parser.add_argument('--demo', action='store_true', help='load first 100 samples when debugging')
    args = parser.parse_args()

    cfg = Configer(args)

    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    DEBUG = (sys.gettrace() is not None)

    RPETrainer(cfg=cfg, DEBUG=DEBUG)
