import numpy as np
import torch


def PCK_margin(pred_heatmap: torch.Tensor, true_heatmap: torch.Tensor, alpha: float = 0.1, thres: float = 0.5):
    """Computes PCK metric between predicted and ground truth heatmaps

    Parameters
    ----------
    pred_heatmap: torch.Tensor
        Predicted heatmaps.
    true_heatmap: torch.Tensor
        Ground truth heatmaps.
    alpha: float
        Percentage of image to consider as margin error for the prediction.
    thres: float
        Threshold applied to heatmap peak value in order to consider a keypoint as a valid prediction.

    Returns
    -------
    avg_pck: float
        Average PCK metric over all 2D joints.
    pck_scores: np.array
        Array of PCK metric values for each 2D joint.
    """
    try:
        pred_heatmap = pred_heatmap.to('cpu').numpy()
        true_heatmap = true_heatmap.to('cpu').numpy()
    except RuntimeError:
        pred_heatmap = pred_heatmap.detach().to('cpu').numpy()
        true_heatmap = true_heatmap.detach().to('cpu').numpy()

    batch_size, n_kpoints, img_h, img_w = pred_heatmap.shape

    x_margin = alpha * img_w
    y_margin = alpha * img_h

    overall_corrects = np.full(shape=(batch_size, n_kpoints), fill_value=np.nan)

    for b in range(batch_size):
        for n in range(n_kpoints):
            true_kpoint_heatmap = true_heatmap[b][n]
            pred_kpoint_heatmap = pred_heatmap[b][n]

            if true_kpoint_heatmap.max() > thres:
                true_kpoint_idx = np.array(np.unravel_index(np.argmax(true_kpoint_heatmap, axis=None),
                                                            (img_h, img_w)))
                pred_kpoint_idx = np.array(np.unravel_index(np.argmax(pred_kpoint_heatmap, axis=None),
                                                            (img_h, img_w)))

                x_dist = np.abs(true_kpoint_idx[1] - pred_kpoint_idx[1])
                y_dist = np.abs(true_kpoint_idx[0] - pred_kpoint_idx[0])

                is_correct = int(x_dist <= x_margin and y_dist <= y_margin)
                overall_corrects[b][n] = is_correct

    pck_scores = np.nanmean(overall_corrects, axis=0)
    pck_scores[np.isnan(pck_scores)] = 0
    avg_pck = np.mean(pck_scores)

    return avg_pck, pck_scores


def PCK_pixel(pred_heatmap: torch.Tensor, true_heatmap: torch.Tensor,
              pixel_thres: float = 2.5, conf_thres: float = 0.5):
    """Computes PCK metric between predicted and ground truth heatmaps

    Parameters
    ----------
    pred_heatmap: torch.Tensor
        Predicted heatmaps.
    true_heatmap: torch.Tensor
        Ground truth heatmaps.
    pixel_thres: float
        Threshold applied to pixel error on each keypoint.
    conf_thres: float
        Threshold applied to heatmap peak value in order to consider a keypoint as a valid prediction.

    Returns
    -------
    avg_pck: float
        Average PCK metric over all 2D joints.
    pck_scores: np.array
        Array of PCK metric values for each 2D joint.
    """
    try:
        pred_heatmap = pred_heatmap.to('cpu').numpy()
        true_heatmap = true_heatmap.to('cpu').numpy()
    except RuntimeError:
        pred_heatmap = pred_heatmap.detach().to('cpu').numpy()
        true_heatmap = true_heatmap.detach().to('cpu').numpy()

    batch_size, n_kpoints, img_h, img_w = pred_heatmap.shape

    overall_corrects = np.full(shape=(batch_size, n_kpoints), fill_value=np.nan)

    for b in range(batch_size):
        for n in range(n_kpoints):
            true_kpoint_heatmap = true_heatmap[b][n]
            pred_kpoint_heatmap = pred_heatmap[b][n]

            if true_kpoint_heatmap.max() > conf_thres:
                true_kpoint_idx = np.array(np.unravel_index(np.argmax(true_kpoint_heatmap, axis=None),
                                                            (img_h, img_w)))
                pred_kpoint_idx = np.array(np.unravel_index(np.argmax(pred_kpoint_heatmap, axis=None),
                                                            (img_h, img_w)))

                dist = np.sqrt((true_kpoint_idx[0] - pred_kpoint_idx[0])**2 +
                               (true_kpoint_idx[1] - pred_kpoint_idx[1])**2)

                is_correct = int(dist <= pixel_thres)
                overall_corrects[b][n] = is_correct

    pck_scores = np.nanmean(overall_corrects, axis=0)
    pck_scores[np.isnan(pck_scores)] = 0
    avg_pck = np.mean(pck_scores)

    return avg_pck, pck_scores


def joints_3D_precision(intrinsic, pred_heatmap, joints3D_gt_depth, joints3D_gt_Z, z_values, cm_thres=0.1):
    """Computes Average Precision (AP) metric between predicted and ground truth 3D joints

    Parameters
    ----------
    intrinsic: np.array
        Intrinsic camera parameters as a 3x3 matrix.
    pred_heatmap: torch.Tensor
        Predicted heatmaps.
    joints3D_gt_depth: torch.Tensor
        Array of 3D joints coordinates computed with the depth values.
    joints3D_gt_Z: torch.Tensor
        Array of precise ground truth 3D joints coordinate.
    z_values: torch.Tensor
        Array of depth values with the same dimension of input image (HxW)
    cm_thres: float
        Threshold to apply on the distance between prediction and ground truth.

    Returns
    -------
    joints3D_avg_precision_depth: float
        Average precision over all 3D joints computed with the depth values.
    joints3D_precision_depth: np.array
        Array of average precision values for each 3D joint computed with the depth values.
    joints3D_avg_precision_Z: float
        Average precision over all 3D precise ground truth joints.
    joints3D_precision_Z: np.array
        Array of average precision values for each 3D precise ground truth joint.
    """
    # get 2D coordinates from gt and predicted heatmaps
    pred_joints2D = np.ones((pred_heatmap.shape[0], pred_heatmap.shape[1], 3))
    pred_joints2D[:, :, :2] = np.array([np.array(
        [np.unravel_index(np.argmax(pred_heatmap[b, i, :, :].detach().cpu().numpy(),
                                    axis=None), pred_heatmap[b, i, :, :].shape)
         for i in range(pred_heatmap.shape[1])])
        for b in range(pred_heatmap.shape[0])])[..., ::-1]
    # add Z coordinate from GT depth map
    z = np.array([[z_values[b, int(el[1]), int(el[0])] for el in pred_joints2D[b]]
                  for b in range(z_values.shape[0])])
    # convert 2D predicted joints to 3D coordinate multiplying by inverse intrinsic matrix
    inv_intrinsic = torch.inverse(intrinsic).unsqueeze(1).repeat(1, pred_joints2D.shape[1], 1, 1).detach().cpu().numpy()
    pred_joints3D = (inv_intrinsic @ pred_joints2D[:, :, :, None]).squeeze(-1)
    pred_joints3D *= z[:, :, None]
    # compute mean average precision metric and per-joint average precision metric
    joints3D_avg_precision_depth = np.mean(np.sum(
        np.sqrt(np.sum((pred_joints3D - joints3D_gt_depth.detach().cpu().numpy()) ** 2, axis=-1)) < cm_thres,
        axis=-1) / pred_joints3D.shape[1])
    joints3D_precision_depth = np.sum(
        np.sqrt(np.sum((pred_joints3D - joints3D_gt_depth.detach().cpu().numpy()) ** 2,axis=-1)).transpose() < cm_thres,
        axis=-1) / pred_joints3D.shape[0]
    joints3D_avg_precision_Z = np.mean(np.sum(
        np.sqrt(np.sum((pred_joints3D - joints3D_gt_Z.detach().cpu().numpy()) ** 2, axis=-1)) < cm_thres,
        axis=-1) / pred_joints3D.shape[1])
    joints3D_precision_Z = np.sum(
        np.sqrt(np.sum((pred_joints3D - joints3D_gt_Z.detach().cpu().numpy()) ** 2, axis=-1)).transpose() < cm_thres,
        axis=-1) / pred_joints3D.shape[0]

    return joints3D_avg_precision_depth, joints3D_precision_depth, joints3D_avg_precision_Z, joints3D_precision_Z
