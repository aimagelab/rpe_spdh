import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm


class Color:
    """
    Colors enumerator
    """
    BLACK = (0, 0, 0)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)


_KP_NAMES = JOINTS_NAMES = [
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

kp_plot_names = [
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


def kpoint_to_heatmap(kpoint, shape, sigma):
    """Converts a 2D keypoint to a gaussian heatmap

    Parameters
    ----------
    kpoint: np.array
        2D coordinates of keypoint [x, y].
    shape: tuple
        Heatmap dimension (HxW).
    sigma: float
        Variance value of the gaussian.

    Returns
    -------
    heatmap: np.array
        A gaussian heatmap HxW.
    """
    map_h, map_w = shape
    if np.all(kpoint > 0):
        x, y = kpoint
        x *= map_w
        y *= map_h
        xy_grid = np.mgrid[:map_w, :map_h].transpose(2, 1, 0)
        heatmap = np.exp(-np.sum((xy_grid - (x, y)) ** 2, axis=-1) / sigma ** 2)
        heatmap /= (heatmap.max() + np.finfo('float32').eps)
    else:
        heatmap = np.zeros((map_h, map_w))
    return heatmap


def heatmap_from_kpoints_array(kpoints_array, shape, sigma):
    """Converts N 2D keypoints to N gaussian heatmaps

    Parameters
    ----------
    kpoints_array: np.array
        Array of 2D coordinates.
    shape: tuple
        Heatmap dimension (HxW).
    sigma: float
        Variance value of the gaussian.

    Returns
    -------
    heatmaps: np.array
        Array of NxHxW gaussian heatmaps.
    """
    heatmaps = []
    for kpoint in kpoints_array:
        heatmaps.append(kpoint_to_heatmap(kpoint, shape, sigma))
    return np.stack(heatmaps, axis=-1)


def random_blend_grid(true_blends, pred_blends):
    """Stacks predicted and ground truth blended images (heatmap+image) by column.

    Parameters
    ----------
    true_blends: np.array
        Ground truth blended images.
    pred_blends: np.array
        Predicted blended images.

    Returns
    -------
    grid: np.array
        Grid of predicted and ground truth blended images.
    """
    grid = []
    for i in range(0, len(true_blends)):
        grid.append(np.concatenate(true_blends[i], axis=2))
        grid.append(np.concatenate(pred_blends[i], axis=2))
    return grid


def to_colormap(heatmap_tensor, device, cmap = 'jet', cmap_range=(None, None)):
    """Converts a heatmap into an image assigning to the gaussian values a colormap.

    Parameters
    ----------
    heatmap_tensor: torch.Tensor
        Heatmap as a tensor (NxHxW).
    cmap: str
        Colormap to use for heatmap visualization.
    cmap_range: tuple
        Range of values for the colormap.

    Returns
    -------
    output: np.array
        Array of N images representing the heatmaps.
    """
    if not isinstance(heatmap_tensor, np.ndarray):
        try:
            heatmap_tensor = heatmap_tensor.to('cpu').numpy()
        except RuntimeError:
            heatmap_tensor = heatmap_tensor.detach().to('cpu').numpy()

    cmap = cm.ScalarMappable(cmap=cmap)
    cmap.set_clim(vmin=cmap_range[0], vmax=cmap_range[1])

    heatmap_tensor = np.max(heatmap_tensor, axis=1)

    output = []
    batch_size = heatmap_tensor.shape[0]
    for b in range(batch_size):
        rgb = cmap.to_rgba(heatmap_tensor[b])[:, :, :-1]
        output.append(rgb[:, :, ::-1])  # RGB to BGR

    output = np.asarray(output).astype(np.float32)
    output = output.transpose(0, 3, 1, 2)  # (b, h, w, c) -> (b, c, h, w)

    return output


def get_keypoint_barplot(x_data, y_data, metric):
    """Computes a barplot given X and Y data.

    Parameters
    ----------
    x_data: np.array
        X data.
    y_data: np.array
        Y data.
    metric: str
        Metric type for graph title.

    Returns
    -------
    canvas: plot
        Barplot to visualize with matplotlib

    """
    fig, ax = plt.subplots()
    ax.bar(x=x_data, height=y_data)
    ax.set_ylim((0, 100))
    ax.set_xticks(x_data)
    ax.set_xticklabels([kp_plot_names[idx] for idx in range(len(x_data))],
                       fontdict={'rotation': 'vertical'})
    ax.set_title(f'{metric} for each joint')
    # Tweak spacing to prevent clipping of tick-labels
    plt.subplots_adjust(bottom=0.4)
    fig.canvas.draw()
    canvas = np.array(fig.canvas.renderer._renderer)[:, :, :-1]
    plt.close()
    return canvas


def gkern(d, h, w, center, s=2, device='cuda'):
    x = torch.arange(0, w, 1).float().to(device)
    y = torch.arange(0, h, 1).float().to(device)
    y = y.unsqueeze(1)
    z = torch.arange(0, d, 1).float().to(device)
    z = z.unsqueeze(1).unsqueeze(1)

    x0 = center[0]
    y0 = center[1]
    z0 = center[2]

    return torch.exp(-1 * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / s ** 2)


def get_local_maxima_3d(hmap3d, threshold, device='cuda'):
    d = torch.device(device)

    m_f = torch.zeros(hmap3d.shape).to(d)
    m_f[1:, :, :] = hmap3d[:-1, :, :]
    m_b = torch.zeros(hmap3d.shape).to(d)
    m_b[:-1, :, :] = hmap3d[1:, :, :]

    m_u = torch.zeros(hmap3d.shape).to(d)
    m_u[:, 1:, :] = hmap3d[:, :-1, :]
    m_d = torch.zeros(hmap3d.shape).to(d)
    m_d[:, :-1, :] = hmap3d[:, 1:, :]

    m_r = torch.zeros(hmap3d.shape).to(d)
    m_r[:, :, 1:] = hmap3d[:, :, :-1]
    m_l = torch.zeros(hmap3d.shape).to(d)
    m_l[:, :, :-1] = hmap3d[:, :, 1:]

    p = torch.zeros(hmap3d.shape).to(d)
    p[hmap3d >= m_f] = 1
    p[hmap3d >= m_b] += 1
    p[hmap3d >= m_u] += 1
    p[hmap3d >= m_d] += 1
    p[hmap3d >= m_r] += 1
    p[hmap3d >= m_l] += 1

    p[hmap3d >= threshold] += 1
    p[p != 7] = 0

    return torch.tensor(torch.nonzero(p).cpu())


def get_maxima_heatmap3d(heatmap3d, intrinsic):
    try:
        heatmap3d = heatmap3d.to('cpu').numpy()
    except RuntimeError:
        heatmap3d = heatmap3d.detach().to('cpu').numpy()

    batch_size, n_kpoints, img_w, img_h, img_d = heatmap3d.shape

    joints3d_all = np.ones((batch_size, n_kpoints, 3))
    for b in range(batch_size):
        vsize = heatmap3d[b].shape[-3:]
        output_rs = heatmap3d[b].reshape(-1, np.prod(vsize))
        max_index = np.unravel_index(np.argmax(output_rs, axis=1), vsize)
        max_index = np.array(max_index).T
        Z = max_index[:, 2] / ((img_d - 1) / 5.)
        joints2d = np.ones_like(max_index)
        joints2d[:, 0] = max_index[:, 0] * 4
        joints2d[:, 1] = max_index[:, 1] * 4
        joints3d = (np.linalg.inv(intrinsic[b]) @ joints2d[..., None]).squeeze(-1)
        joints3d *= Z[..., None]
        joints3d[:, 1] *= -1
        joints3d_all[b] = joints3d

    return joints3d_all
