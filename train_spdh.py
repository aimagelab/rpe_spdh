import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel as DP
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from datasets.BaxterSynthDataLoader import BaxterJointsSynthDataset, JOINTS_NAMES as synth_joints_name
from networks.HRNet.model import HRNet
from networks.pose_machines.pose_machines import PoseMachine
from networks.stacked_hourglass.model import HourglassNet
from utils.configer import Configer
from utils.metrics import PCK_pixel as PCK
from utils.utils import to_colormap, get_keypoint_barplot, random_blend_grid


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

        if self.cfg["phase"] == "train":
            net_info = self.cfg["config"].split('/')[-1][:-11]
            new_res_dir = self.cfg["results_dir"].parts[-1] + f'_{net_info}'
            self.cfg["results_dir"] = self.cfg["results_dir"].parents[0] / new_res_dir
            self.train()
        else:
            raise ValueError

    def init_logger(self):
        if not self.cfg["results_dir"].is_dir():
            self.cfg["results_dir"].mkdir(parents=True, exist_ok=True)
        self.log_writer = SummaryWriter(str(self.cfg["results_dir"]))

        # log experiment setting
        param_str = ''
        for key, value in self.cfg.items():
            if isinstance(value, dict):
                for key1, value1 in value.items():
                    param_str += f'{key}.{key1}: {value1}  \n'
            else:
                param_str += f'{key}: {value}  \n'
        self.log_writer.add_text('Experiment setting', param_str)

    def init_worker(self, worker_id):
        np.random.seed(torch.initial_seed() % 2 ** 32)

    def init_dataset(self):
        if self.cfg["dataset"] == "synthetic":
            self.JOINTS_NAMES = synth_joints_name
        if self.cfg["phase"] == "train":
            self.dataset = BaxterJointsSynthDataset(dataset_dir=self.cfg["data"]["data_path"],
                                                    run=self.cfg["data"]["run"],
                                                    init_mode="train", noise=self.cfg['noise'],
                                                    img_type=self.cfg["data"]["type"],
                                                    img_size=tuple(self.cfg["data"]["image_size"]),
                                                    sigma=self.cfg["data"]["sigma"],
                                                    norm_type=self.cfg["data"]["norm_type"],
                                                    network_input=self.cfg["network"]["input_type"],
                                                    network_task=self.cfg["network"]["task"],
                                                    depth_range=self.cfg["data"]["depth_range"],
                                                    depth_range_type=self.cfg["data"]["depth_range_type"],
                                                    aug_type=self.cfg["data"]["aug_type"],
                                                    aug_mode=self.cfg["data"]["aug"],
                                                    demo=self.cfg['demo'])
            self.dataloader = {
                "train": torch.utils.data.DataLoader(self.dataset, batch_size=self.cfg["data"]["batch_size"],
                                                     shuffle=True, num_workers=self.num_workers,
                                                     worker_init_fn=self.init_worker, drop_last=True),
                "val": torch.utils.data.DataLoader(self.dataset, batch_size=self.cfg["data"]["batch_size"],
                                                   shuffle=False, num_workers=self.num_workers,
                                                   worker_init_fn=self.init_worker, drop_last=True)
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
        elif self.cfg["network"]["name"] == "fast_pose_machines":
            self.model = DP(PoseMachine(params=self.cfg["network_params"]),
                            device_ids=[0])
        else:
            raise ValueError
        self.model.to(self.device)

        self.starting_epoch = 0
        self.global_iter = 0
        self.avg_AP_scores = []

        if self.cfg["resume"] != "":
            self.load_model_weights(self.cfg["resume"])

    def init_criterion_and_optimizer(self):
        self.criterion = torch.nn.MSELoss()
        if self.cfg["phase"] == "train":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["solver"]["lr"])
            self.scheduler = MultiStepLR(self.optimizer, cfg["solver"]["decay_steps"], gamma=0.1)

            if self.cfg["resume"] != "":
                self.load_optimizer_weights()
                self.load_scheduler_weights()

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
            if self.cfg["phase"] == 'train':
                not_change = ["results_dir", "resume", "demo", "phase"]
            else:
                not_change = ["results_dir", "resume", "demo", "dataset", "device", "phase", "metrics", "network"]
            if k not in not_change:
                self.cfg[k] = v

    def load_optimizer_weights(self):
        self.optimizer.load_state_dict(self.checkpoint["optimizer"])
        print(f'restore AdamW optimizer')

    def load_scheduler_weights(self):
        self.scheduler.load_state_dict(self.checkpoint["scheduler"])
        print(f'restore AdamW scheduler')

    def save_weights(self, save_dir, type: str = 'best'):
        if type == 'best':
            save_dir = save_dir / 'best_checkpoint.pth'
        elif type == 'latest':
            save_dir = save_dir / 'latest_checkpoint.pth'
        else:
            save_dir = save_dir / f'epoch_{(self.epoch + 1):03}_checkpoint.pth'

        save_dict = {
            'epoch': self.epoch,
            'global_iter': self.global_iter,
            'avg_AP': self.avg_AP_scores,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.cfg
        }

        torch.save(save_dict, str(save_dir))

    def log_and_visualize(self, phase, phase_str, batch_losses,
                          UV_avg_PCKs, UV_PCK_scores, UZ_avg_PCKs, UZ_PCK_scores,
                          avg_precision_scores, precision_scores,
                          gt_results=None, pred_results=None,
                          true_blends_UV=None, pred_blends_UV=None,
                          true_blends_UZ=None, pred_blends_UZ=None):
        # averaged loss and metrics
        mean_epoch_loss = np.mean(batch_losses)

        UV_avg_avg_PCK = np.round(np.mean(UV_avg_PCKs) * 100, 2)
        UV_avg_PCK_scores = np.round(np.mean(UV_PCK_scores, axis=0) * 100, 2)
        UV_avg_PCK_scores_str = ''
        for n, PCK_score in enumerate(UV_avg_PCK_scores):
            UV_avg_PCK_scores_str += f'{self.JOINTS_NAMES[n]}: {PCK_score}% | '
            count = n + 1
            if count > 1 and count % 4 == 0 and count != len(UV_avg_PCK_scores):
                UV_avg_PCK_scores_str += '\n'
        UZ_avg_avg_PCK = np.round(np.mean(UZ_avg_PCKs) * 100, 2)
        UZ_avg_PCK_scores = np.round(np.mean(UZ_PCK_scores, axis=0) * 100, 2)
        UZ_avg_PCK_scores_str = ''
        for n, PCK_score in enumerate(UZ_avg_PCK_scores):
            UZ_avg_PCK_scores_str += f'{self.JOINTS_NAMES[n]}: {PCK_score}% | '
            count = n + 1
            if count > 1 and count % 4 == 0 and count != len(UZ_avg_PCK_scores):
                UZ_avg_PCK_scores_str += '\n'

        avg_avg_precision = np.round(np.mean(avg_precision_scores, axis=0) * 100, 2)
        if phase == 'val':
            self.avg_AP_scores.append(avg_avg_precision[0])
        avg_precision_score_str = ''
        for thres, val in zip([0.02, 0.06, 0.1], avg_avg_precision):
            avg_precision_score_str += f'THRESHOLD {thres:.2f} | AP = {val}%\n'
        avg_joint_precision = np.round(np.mean(precision_scores, axis=0) * 100, 2)
        precision_score_str = ''
        for thres, val in zip([0.02, 0.06, 0.1], avg_joint_precision):
            precision_score_str += f'THRESHOLD {thres:.2f}:\n'
            for n, precision in enumerate(val):
                precision_score_str += f'{self.JOINTS_NAMES[n]} = {precision}%\n'
        print(f'Loss: {mean_epoch_loss}')
        print(f'UV PCK score: {UV_avg_avg_PCK}%')
        print(f'UV PCK scores per joint:\n{UV_avg_PCK_scores_str}')
        print(f'UZ PCK score: {UZ_avg_avg_PCK}%')
        print(f'UZ PCK scores per joint:\n{UZ_avg_PCK_scores_str}')
        print(f'Mean avg precision:\n{avg_precision_score_str}')
        print(f'Average precision per joint:\n{precision_score_str}')

        # log scalars, AP barplot and joints 2d projection on depth image
        self.log_writer.add_scalar(f'{phase_str}/MSE Loss', mean_epoch_loss, self.global_iter)
        self.log_writer.add_scalar(f'{phase_str}/UV PCK', UV_avg_avg_PCK, self.global_iter)
        self.log_writer.add_scalar(f'{phase_str}/UZ PCK', UZ_avg_avg_PCK, self.global_iter)
        self.log_writer.add_scalar(f'{phase_str}/AP', avg_avg_precision[0], self.global_iter)
        self.log_writer.add_scalar(f'{phase_str}/AP 6cm', avg_avg_precision[1], self.global_iter)
        self.log_writer.add_scalar(f'{phase_str}/AP 10cm', avg_avg_precision[2], self.global_iter)
        # plot and visualize per keypoint precision histogram
        for curr_avg_joint_precision, thres_str in zip(avg_joint_precision, ['', ' 6cm', ' 10cm']):
            if self.cfg["network"]["name"] in ["transformer", "competitor_r", "competitor_h", "uniformer"]:
                num_joints = self.cfg["data"]["n_joints"]
            else:
                num_joints = self.cfg["data"]["n_joints"] // 2
            precision_barplot = get_keypoint_barplot(np.arange(num_joints, dtype=int),
                                                     curr_avg_joint_precision, metric='Precision')
            precision_barplot = torch.from_numpy(precision_barplot.astype(np.float32) / 255.).permute(2, 0, 1)
            self.log_writer.add_image(tag=f'{phase_str}/Precision bar plot{thres_str}',
                                      img_tensor=precision_barplot,
                                      global_step=self.global_iter)

        if gt_results is not None and pred_results is not None:
            results_grid = random_blend_grid(gt_results, pred_results)
            results_grid = np.concatenate(results_grid, axis=1)
            results_grid = torch.from_numpy(results_grid)
            self.log_writer.add_image(tag=f'{phase_str}/Joints Prediction', img_tensor=results_grid,
                                      global_step=self.global_iter)

        # visualize pred and gt UV heatmaps
        if true_blends_UV is not None and pred_blends_UV is not None:
            eval_grid = random_blend_grid(true_blends_UV, pred_blends_UV)
            eval_grid = np.concatenate(eval_grid, axis=1)
            eval_grid = torch.from_numpy(eval_grid)
            self.log_writer.add_image(tag=f'{phase_str}/Joints UV heatmaps', img_tensor=eval_grid,
                                      global_step=self.global_iter)

        # visualize pred and gt UZ heatmaps
        if true_blends_UZ is not None and pred_blends_UZ is not None:
            eval_grid = random_blend_grid(true_blends_UZ, pred_blends_UZ)
            eval_grid = np.concatenate(eval_grid, axis=1)
            eval_grid = torch.from_numpy(eval_grid)
            self.log_writer.add_image(tag=f'{phase_str}/Joints UZ heatmaps', img_tensor=eval_grid,
                                      global_step=self.global_iter)

        return mean_epoch_loss, UV_avg_avg_PCK, UZ_avg_avg_PCK, avg_avg_precision

    def run_iteration(self, phase, input_tensor, heatmap_gt, joints_3d_gt, input_K, input_fx, input_fy):
        outputs = self.model(input_tensor)

        b, c, h, w = heatmap_gt.size()
        if self.cfg["network"]["name"] == "stacked_hourglass":
            heatmap_pred = outputs['heatmaps']
            curr_loss = self.criterion((F.interpolate(heatmap_pred[0], (h, w), mode='bicubic',
                                                      align_corners=False) + 1) / 2., heatmap_gt)
            for j in range(1, len(heatmap_pred)):
                curr_loss += self.criterion((F.interpolate(heatmap_pred[j], (h, w), mode='bicubic',
                                                           align_corners=False) + 1) / 2., heatmap_gt)
        elif self.cfg["network"]["name"] == "fast_pose_machines":
            heatmap_pred = outputs[0]
            curr_loss = self.criterion((F.interpolate(heatmap_pred, (h, w), mode='bicubic',
                                                      align_corners=False) + 1) / 2., heatmap_gt)
        elif self.cfg["network"]["name"] == "hrnet":
            heatmap_pred = outputs
            curr_loss = self.criterion((F.interpolate(heatmap_pred, (h, w), mode='bicubic',
                                                      align_corners=False) + 1) / 2., heatmap_gt)
        else:
            raise ValueError

        if phase == "train":
            self.optimizer.zero_grad()
            curr_loss.backward()
            self.optimizer.step()

        if self.cfg["network"]["name"] == "stacked_hourglass":
            heatmap_pred = (F.interpolate(heatmap_pred[-1], (h, w), mode='bicubic', align_corners=False) + 1) / 2.
        else:
            heatmap_pred = (F.interpolate(heatmap_pred, (h, w), mode='bicubic', align_corners=False) + 1) / 2.

        uv_avg_pck, uv_pck_scores = PCK(heatmap_pred[:, :(c // 2)], heatmap_gt[:, :(c // 2)],
                                        pixel_thres=self.cfg["metrics"]["PCK_pixel_thres"],
                                        conf_thres=self.cfg["metrics"]["PCK_conf_thres"])
        uz_avg_pck, uz_pck_scores = PCK(heatmap_pred[:, (c // 2):], heatmap_gt[:, (c // 2):],
                                        pixel_thres=self.cfg["metrics"]["PCK_pixel_thres"],
                                        conf_thres=self.cfg["metrics"]["PCK_conf_thres"])

        B, C, H, W = heatmap_pred.shape
        joints_3d_pred = np.ones((B, C // 2, 3))
        joints_3d_pred[:, :, :2] = np.array([np.array(
            [np.unravel_index(np.argmax(heatmap_pred[b, i, :, :].detach().cpu().numpy(),
                                        axis=None), heatmap_pred[b, i, :, :].shape)
             for i in range(0, c // 2)])
            for b in range(heatmap_pred.shape[0])])[..., ::-1]
        # add Z coordinate from UZ heatmap
        z = np.array([np.array(
            [np.unravel_index(np.argmax(heatmap_pred[b, i, :, :].detach().cpu().numpy(),
                                        axis=None), heatmap_pred[b, i, :, :].shape)
             for i in range(c // 2, c)])
            for b in range(heatmap_pred.shape[0])])[..., :1]
        Z_min, _, dZ = self.cfg["data"]["depth_range"]
        z = ((z * dZ) + Z_min) / 1000
        # convert 2D predicted joints to 3D coordinate multiplying by inverse intrinsic matrix
        inv_intrinsic = torch.inverse(input_K).unsqueeze(1).repeat(1,
                                                                   joints_3d_pred.shape[1],
                                                                   1,
                                                                   1).detach().cpu().numpy()
        joints_3d_pred = (inv_intrinsic @ joints_3d_pred[:, :, :, None]).squeeze(-1)
        joints_3d_pred *= z
        joints_3d_pred[:, :, 1] *= -1  # invert Y axis for left-handed reference frame

        thresholds = [0.02, 0.06, 0.1]
        joints3D_avg_precision, joints3D_precision = [], []
        dist_3D = np.sqrt(np.sum((joints_3d_pred - joints_3d_gt.cpu().numpy()) ** 2, axis=-1))
        for thres in thresholds:
            num_joints = heatmap_pred.shape[1] // 2
            avg_AP = np.mean(np.sum(dist_3D < thres, axis=-1) / num_joints)
            joints_AP = np.sum(dist_3D.transpose() < thres, axis=-1) / B
            joints3D_avg_precision.append(avg_AP)
            joints3D_precision.append(joints_AP)

        return heatmap_pred, joints_3d_pred, curr_loss, uv_avg_pck, uv_pck_scores, uz_avg_pck, uz_pck_scores, \
               joints3D_avg_precision, joints3D_precision

    def run_epoch(self, phase):
        if phase == 'train':
            phase_str = 'Train'

        batch_losses = []
        UV_avg_PCKs = []
        UV_PCK_scores = []
        UZ_avg_PCKs = []
        UZ_PCK_scores = []
        avg_precision_scores = []
        precision_scores = []
        if phase == 'train':
            self.validation_iteration = len(self.dataloader[phase]) // 4
        print(f'#### {phase_str} ####')

        if phase == 'train':
            epoch_losses = []
            epoch_UV_avg_PCKs = []
            epoch_UZ_avg_PCKs = []
            epoch_avg_precision_scores = []
            epoch_eval_losses = []
            epoch_eval_UV_avg_PCKs = []
            epoch_eval_UZ_avg_PCKs = []
            epoch_eval_avg_precision_scores = []
            gt_results = []
            pred_results = []
            true_blends_UV = []
            pred_blends_UV = []
            true_blends_UZ = []
            pred_blends_UZ = []
            sampling_ranges = [range(self.validation_iteration)]
            for i in range(3):
                sampling_ranges.append(
                    range(self.validation_iteration * (2 + (i - 1)), self.validation_iteration * (2 + i))
                )
            random_samples = [np.random.choice(sampling_range, 4, replace=False) for sampling_range in sampling_ranges]
            random_samples = np.sort(np.asarray(random_samples).flatten())

        for self.iter, batch in enumerate(self.dataloader[phase]):
            start_iter = time.time()
            start_iter_datetime = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))

            joints_3d = batch['joints_3D_Z'].to(self.device).float()
            heatmaps_gt = batch['heatmap_25d'].to(self.device).float()
            input_K = batch['K_depth'].to(self.device).float()
            input_fx = batch['K_depth'].to(self.device).float()[:, 0, 0]
            input_fy = batch['K_depth'].to(self.device).float()[:, 1, 1]

            if self.cfg["network"]["input_type"] == 'D':
                input_tensor = batch['depth16'].to(self.device).float()
            elif self.cfg["network"]["input_type"] == 'RGB':
                input_tensor = batch['rgb'].to(self.device).float()
            elif self.cfg["network"]["input_type"] == 'RGBD':
                input_tensor = torch.cat([batch['rgb'].to(self.device).float(),
                                          batch['depth16'].to(self.device).float()], dim=1)
            elif self.cfg["network"]["input_type"] == 'XYZ':
                input_tensor = batch['xyz_img'].to(self.device).float()
            else:
                raise ValueError

            if phase != 'train':
                with torch.no_grad():
                    heatmaps_pred, joints_3d_pred, curr_loss, uv_avg_pck, uv_pck_scores, \
                    uz_avg_pck, uz_pck_scores, \
                    joints3D_avg_precision, joints3D_precision = self.run_iteration(phase,
                                                                                    input_tensor,
                                                                                    heatmaps_gt,
                                                                                    joints_3d,
                                                                                    input_K,
                                                                                    input_fx,
                                                                                    input_fy)
            else:
                heatmaps_pred, joints_3d_pred, curr_loss, uv_avg_pck, uv_pck_scores, \
                uz_avg_pck, uz_pck_scores, \
                joints3D_avg_precision, joints3D_precision = self.run_iteration(phase,
                                                                                input_tensor,
                                                                                heatmaps_gt,
                                                                                joints_3d,
                                                                                input_K,
                                                                                input_fx,
                                                                                input_fy)

            batch_losses.append(curr_loss.item())
            UV_avg_PCKs.append(uv_avg_pck)
            UV_PCK_scores.append(uv_pck_scores)
            UZ_avg_PCKs.append(uz_avg_pck)
            UZ_PCK_scores.append(uz_pck_scores)
            avg_precision_scores.append(joints3D_avg_precision)
            precision_scores.append(joints3D_precision)

            if phase == 'train' and self.iter in random_samples:
                gt_images = batch['depth16vis'].clone().numpy()[:8]
                pred_images = gt_images.copy()
                blend_images = gt_images.copy()

                K = batch['K_depth'].clone().numpy()[:8]
                joints_3d_gt = joints_3d.clone().cpu().numpy()[:8]
                joints_3d_pred = joints_3d_pred.copy()[:8]
                joints_3d_gt[..., 1] *= -1
                joints_2d_gt = (K[:, None, :, :] @ joints_3d_gt[..., None]).squeeze(-1)
                joints_2d_gt = joints_2d_gt / joints_2d_gt[..., 2:]
                joints_3d_pred[..., 1] *= -1
                joints_2d_pred = (K[:, None, :, :] @ joints_3d_pred[..., None]).squeeze(-1)
                joints_2d_pred = (joints_2d_pred / (joints_2d_pred[..., 2:] + 1e-9))
                for b in range(len(gt_images)):
                    for joint_2d_gt, joint_2d_pred in zip(joints_2d_gt[b, ...], joints_2d_pred[b, ...]):
                        cv2.circle(gt_images[b],
                                   (int(joint_2d_gt[0]), int(joint_2d_gt[1])), 2, (255, 0, 0), -1)
                        cv2.circle(pred_images[b],
                                   (int(joint_2d_pred[0]), int(joint_2d_pred[1])), 2, (0, 255, 0), -1)
                gt_images = np.stack(gt_images).transpose(0, 3, 1, 2).astype(float) / 255.
                pred_images = np.stack(pred_images).transpose(0, 3, 1, 2).astype(float) / 255.
                gt_results.append(gt_images)
                pred_results.append(pred_images)

                c = heatmaps_pred.shape[1]
                blend_images = np.stack(blend_images).transpose(0, 3, 1, 2).astype(float) / 255.
                pred_blend_uv = 0.5 * blend_images + 0.5 * to_colormap(heatmaps_pred[:8, :(c // 2)], self.device)
                pred_blend_uz = to_colormap(heatmaps_pred[:8, (c // 2):], self.device)
                true_blend_uv = 0.5 * blend_images + 0.5 * to_colormap(heatmaps_gt[:8, :(c // 2)], self.device)
                true_blend_uz = to_colormap(heatmaps_gt[:8, (c // 2):], self.device)
                true_blends_UV.append(true_blend_uv)
                true_blends_UZ.append(true_blend_uz)
                pred_blends_UV.append(pred_blend_uv)
                pred_blends_UZ.append(pred_blend_uz)

            end_iter = time.time()
            print(f'[{end_iter - start_iter:.2f}s/it] - '
                  f'[from {start_iter_datetime} to {str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))}]')

            # validation every 1/4 of the number of current epoch iterations
            if phase == 'train':
                if self.iter > 0 and self.iter % self.validation_iteration == 0:
                    # log training results
                    iter_mean_loss, iter_UV_PCK, iter_UZ_PCK, iter_AP = self.log_and_visualize(phase, phase_str,
                                                                                               batch_losses,
                                                                                               UV_avg_PCKs,
                                                                                               UV_PCK_scores,
                                                                                               UZ_avg_PCKs,
                                                                                               UZ_PCK_scores,
                                                                                               avg_precision_scores,
                                                                                               precision_scores,
                                                                                               gt_results,
                                                                                               pred_results,
                                                                                               true_blends_UV,
                                                                                               pred_blends_UV,
                                                                                               true_blends_UZ,
                                                                                               pred_blends_UZ)
                    if self.iter > 0:
                        epoch_losses.append(iter_mean_loss)
                        epoch_UV_avg_PCKs.append(iter_UV_PCK)
                        epoch_UZ_avg_PCKs.append(iter_UZ_PCK)
                        epoch_avg_precision_scores.append(iter_AP)
                    # set validation parameters
                    phase = 'val'
                    phase_str = 'Validation'
                    print(f'#### {phase_str} ####')
                    self.model.eval()
                    self.dataset.eval()
                    batch_losses = []
                    UV_avg_PCKs = []
                    UV_PCK_scores = []
                    UZ_avg_PCKs = []
                    UZ_PCK_scores = []
                    avg_precision_scores = []
                    precision_scores = []
                    gt_results = []
                    pred_results = []
                    true_blends_UV = []
                    pred_blends_UV = []
                    true_blends_UZ = []
                    pred_blends_UZ = []
                    sampling_range = range(len(self.dataloader[phase]))
                    random_sample = np.random.choice(sampling_range, 4, replace=False)

                    # validation iterations
                    for i, batch in enumerate(self.dataloader[phase]):
                        start_iter = time.time()
                        start_iter_datetime = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))

                        joints_3d = batch['joints_3D_Z'].to(self.device).float()
                        heatmaps_gt = batch['heatmap_25d'].to(self.device).float()
                        input_K = batch['K_depth'].to(self.device).float()
                        input_fx = batch['K_depth'].to(self.device).float()[:, 0, 0]
                        input_fy = batch['K_depth'].to(self.device).float()[:, 1, 1]

                        if self.cfg["network"]["input_type"] == 'D':
                            input_tensor = batch['depth16'].to(self.device).float()
                        elif self.cfg["network"]["input_type"] == 'RGB':
                            input_tensor = batch['rgb'].to(self.device).float()
                        elif self.cfg["network"]["input_type"] == 'RGBD':
                            input_tensor = torch.cat([batch['rgb'].to(self.device).float(),
                                                      batch['depth16'].to(self.device).float()], dim=1)
                        elif self.cfg["network"]["input_type"] == 'XYZ':
                            input_tensor = batch['xyz_img'].to(self.device).float()
                        else:
                            raise ValueError

                        with torch.no_grad():
                            heatmaps_pred, joints_3d_pred, curr_loss, uv_avg_pck, uv_pck_scores, \
                            uz_avg_pck, uz_pck_scores, \
                            joints3D_avg_precision, joints3D_precision = self.run_iteration(phase,
                                                                                            input_tensor,
                                                                                            heatmaps_gt,
                                                                                            joints_3d,
                                                                                            input_K,
                                                                                            input_fx,
                                                                                            input_fy)

                        batch_losses.append(curr_loss.item())
                        UV_avg_PCKs.append(uv_avg_pck)
                        UV_PCK_scores.append(uv_pck_scores)
                        UZ_avg_PCKs.append(uz_avg_pck)
                        UZ_PCK_scores.append(uz_pck_scores)
                        avg_precision_scores.append(joints3D_avg_precision)
                        precision_scores.append(joints3D_precision)

                        if i in random_sample:
                            gt_images = batch['depth16vis'].clone().numpy()[:8]
                            pred_images = gt_images.copy()
                            blend_images = gt_images.copy()

                            K = batch['K_depth'].clone().numpy()[:8]
                            joints_3d_gt = joints_3d.clone().cpu().numpy()[:8]
                            joints_3d_pred = joints_3d_pred.copy()[:8]
                            joints_3d_gt[..., 1] *= -1
                            joints_2d_gt = (K[:, None, :, :] @ joints_3d_gt[..., None]).squeeze(-1)
                            joints_2d_gt = joints_2d_gt / joints_2d_gt[..., 2:]
                            joints_3d_pred[..., 1] *= -1
                            joints_2d_pred = (K[:, None, :, :] @ joints_3d_pred[..., None]).squeeze(-1)
                            joints_2d_pred = joints_2d_pred / (joints_2d_pred[..., 2:] + 1e-9)
                            for b in range(len(gt_images)):
                                for joint_2d_gt, joint_2d_pred in zip(joints_2d_gt[b, ...], joints_2d_pred[b, ...]):
                                    cv2.circle(gt_images[b],
                                               (int(joint_2d_gt[0]), int(joint_2d_gt[1])), 2, (255, 0, 0), -1)
                                    cv2.circle(pred_images[b],
                                               (int(joint_2d_pred[0]), int(joint_2d_pred[1])), 2, (0, 255, 0), -1)
                            gt_images = np.stack(gt_images).transpose(0, 3, 1, 2).astype(float) / 255.
                            pred_images = np.stack(pred_images).transpose(0, 3, 1, 2).astype(float) / 255.
                            gt_results.append(gt_images)
                            pred_results.append(pred_images)

                            c = heatmaps_pred.shape[1]
                            blend_images = np.stack(blend_images).transpose(0, 3, 1, 2).astype(float) / 255.
                            pred_blend_uv = 0.5 * blend_images + 0.5 * to_colormap(heatmaps_pred[:8, :(c // 2)], self.device)
                            pred_blend_uz = to_colormap(heatmaps_pred[:8, (c // 2):], self.device)
                            true_blend_uv = 0.5 * blend_images + 0.5 * to_colormap(heatmaps_gt[:8, :(c // 2)], self.device)
                            true_blend_uz = to_colormap(heatmaps_gt[:8, (c // 2):], self.device)
                            true_blends_UV.append(true_blend_uv)
                            true_blends_UZ.append(true_blend_uz)
                            pred_blends_UV.append(pred_blend_uv)
                            pred_blends_UZ.append(pred_blend_uz)

                        end_iter = time.time()
                        print(f'[{end_iter - start_iter:.2f}s/it] - '
                              f'[from {start_iter_datetime} to {str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))}]')

                    # log validation results
                    iter_mean_loss, iter_UV_PCK, iter_UZ_PCK, iter_AP = self.log_and_visualize(phase, phase_str,
                                                                                               batch_losses,
                                                                                               UV_avg_PCKs,
                                                                                               UV_PCK_scores,
                                                                                               UZ_avg_PCKs,
                                                                                               UZ_PCK_scores,
                                                                                               avg_precision_scores,
                                                                                               precision_scores,
                                                                                               gt_results,
                                                                                               pred_results,
                                                                                               true_blends_UV,
                                                                                               pred_blends_UV,
                                                                                               true_blends_UZ,
                                                                                               pred_blends_UZ)

                    if self.iter > 0:
                        epoch_eval_losses.append(iter_mean_loss)
                        epoch_eval_UV_avg_PCKs.append(iter_UV_PCK)
                        epoch_eval_UZ_avg_PCKs.append(iter_UZ_PCK)
                        epoch_eval_avg_precision_scores.append(iter_AP)

                    # save best weights if current avg_precision is the best
                    if len(self.avg_AP_scores) > 1:
                        if self.avg_AP_scores[-1] > np.max(self.avg_AP_scores[:-1]):
                            self.save_weights(self.cfg["results_dir"], type='best')

                    # reset training parameters
                    batch_losses = []
                    UV_avg_PCKs = []
                    UV_PCK_scores = []
                    UZ_avg_PCKs = []
                    UZ_PCK_scores = []
                    avg_precision_scores = []
                    precision_scores = []
                    gt_results, pred_results = [], []
                    true_blends_UV = []
                    pred_blends_UV = []
                    true_blends_UZ = []
                    pred_blends_UZ = []
                    phase = 'train'
                    phase_str = 'Train'
                    print(f'#### {phase_str} ####')
                    self.model.train()
                    self.dataset.train()

            self.global_iter += 1

        if phase == 'train':
            self.log_writer.add_scalar(f'{phase_str}/Epoch MSE Loss', np.mean(epoch_losses), self.epoch + 1)
            self.log_writer.add_scalar(f'{phase_str}/Epoch UV PCK',
                                       np.mean(epoch_UV_avg_PCKs, axis=0), self.epoch + 1)
            self.log_writer.add_scalar(f'{phase_str}/Epoch UZ PCK',
                                       np.mean(epoch_UZ_avg_PCKs, axis=0), self.epoch + 1)
            self.log_writer.add_scalar(f'{phase_str}/Epoch AP',
                                       np.mean(epoch_avg_precision_scores, axis=0)[0], self.epoch + 1)
            self.log_writer.add_scalar(f'{phase_str}/Epoch AP 6cm',
                                       np.mean(epoch_avg_precision_scores, axis=0)[1], self.epoch + 1)
            self.log_writer.add_scalar(f'{phase_str}/Epoch AP 10cm',
                                       np.mean(epoch_avg_precision_scores, axis=0)[2], self.epoch + 1)
            self.log_writer.add_scalar(f'{phase_str}/Epoch LR', self.scheduler.optimizer.param_groups[0]['lr'],
                                       self.epoch + 1)
            self.log_writer.add_scalar(f'Validation/Epoch MSE Loss', np.mean(epoch_eval_losses), self.epoch + 1)
            self.log_writer.add_scalar(f'Validation/Epoch UV PCK',
                                       np.mean(epoch_eval_UV_avg_PCKs, axis=0), self.epoch + 1)
            self.log_writer.add_scalar(f'Validation/Epoch UZ PCK',
                                       np.mean(epoch_eval_UZ_avg_PCKs, axis=0), self.epoch + 1)
            self.log_writer.add_scalar(f'Validation/Epoch AP',
                                       np.mean(epoch_eval_avg_precision_scores, axis=0)[0], self.epoch + 1)
            self.log_writer.add_scalar(f'Validation/Epoch AP 6cm',
                                       np.mean(epoch_eval_avg_precision_scores, axis=0)[1], self.epoch + 1)
            self.log_writer.add_scalar(f'Validation/Epoch AP 10cm',
                                       np.mean(epoch_eval_avg_precision_scores, axis=0)[2], self.epoch + 1)

        # save weight always latest weights and also weights every 2 epochs
        self.save_weights(self.cfg["results_dir"], type='latest')
        if (self.epoch + 1) % 10 == 0:
            self.save_weights(self.cfg["results_dir"], type='epoch')

        return

    def train(self):
        self.init_logger()
        self.init_dataset()
        self.init_model()
        self.init_criterion_and_optimizer()

        for self.epoch in range(self.starting_epoch, self.cfg["epochs"]):
            random.seed(self.cfg['seed'] + self.epoch)
            np.random.seed(self.cfg['seed'] + self.epoch)
            torch.manual_seed(self.cfg['seed'] + self.epoch)

            print(f"Epoch {self.epoch + 1}")
            start_epoch_datetime = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))
            self.model.train()
            self.dataset.train()
            self.run_epoch(phase="train")

            print(f'Epoch {self.epoch + 1}: '
                  f'from {start_epoch_datetime} to {str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f"))}')

            self.scheduler.step(self.epoch + 1)

        self.log_writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='', help='path to training configuration')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint for resuming training')
    parser.add_argument('--results_dir', type=Path, default='./results', help='path to save results')
    parser.add_argument('--device', type=str, default='kinect', help='device type used during recording')
    parser.add_argument('--noise', action='store_true', help='use noise as additional augmentation')
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

    if DEBUG:
        now = str(datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
        args.results_dir = args.results_dir / f'DEBUG_{now}'

    RPETrainer(cfg=cfg, DEBUG=DEBUG)
