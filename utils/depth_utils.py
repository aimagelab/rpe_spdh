import cv2
import numpy as np


def depthmap2normals(depthmap, normalize=True, keep_dims=True):
    """
    Calculate depth normals as
        normals = gF(x,y,z) = (-dF/dx, -dF/dy, 1)
    Args:
        depthmap (np.ndarray): depth map of any dtype, single channel, len(depthmap.shape) == 3
        normalize (bool): if True, normals will be normalized to have unit-magnitude
            Default: True
        keep_dims (bool):
            if True, normals shape will be equals to depthmap shape,
            if False, normals shape will be smaller than depthmap shape.
            Default: True
    Returns:
        Depth normals
    """
    depthmap = np.asarray(depthmap, np.float32)

    if keep_dims is True:
        mask = depthmap != 0
    else:
        mask = depthmap[1:-1, 1:-1] != 0

    if keep_dims is True:
        normals = np.zeros((depthmap.shape[0], depthmap.shape[1], 3), dtype=np.float32)
        normals[1:-1, 1:-1, 0] = - (depthmap[2:, 1:-1] - depthmap[:-2, 1:-1]) / 2
        normals[1:-1, 1:-1, 1] = - (depthmap[1:-1, 2:] - depthmap[1:-1, :-2]) / 2
    else:
        normals = np.zeros((depthmap.shape[0] - 2, depthmap.shape[1] - 2, 3), dtype=np.float32)
        normals[:, :, 0] = - (depthmap[2:, 1:-1] - depthmap[:-2, 1:-1]) / 2
        normals[:, :, 1] = - (depthmap[1:-1, 2:] - depthmap[1:-1, :-2]) / 2
    normals[:, :, 2] = 1

    normals[~mask] = [0, 0, 0]

    if normalize:
        div = np.linalg.norm(normals[mask], ord=2, axis=-1, keepdims=True).repeat(3, axis=-1) + 1e-12
        normals[mask] /= div

    return normals


def depthmap2points(image, fx, fy, cx=None, cy=None):
    """Converts image coordinates to 3D real world coordinates using depth values

    Parameters
    ----------
    image: np.array
        Array of depth values for the whole image.
    fx: float
        Focal of the camera over X axis.
    fy: float
        Focal of the camera over Y axis.
    cx: float
        X coordinate of principal point of the camera.
    cy: float
        Y coordinate of principal point of the camera.

    Returns
    -------
    points: np.array
        Array of XYZ world coordinates.

    """
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, w, h, fx, fy, cx, cy)
    return points


def depthmap2pointcloud(depth, fx, fy, cx=None, cy=None):
    points = depthmap2points(depth, fx, fy, cx, cy)
    points = points.reshape((-1, 3))
    return points


def pointcloud2depthmap(points, img_width, img_height, fx, fy, cx=None, cy=None, rot=None, trans=None):
    if rot is not None or trans is not None:
        raise NotImplementedError

    depthmap = np.zeros((img_height, img_width), dtype=np.float32)
    if len(points) == 0:
        return depthmap

    points = points[np.argsort(points[:, 2])]
    pixels = points2pixels(points, img_width, img_height, fx, fy, cx, cy)
    pixels = pixels.round().astype(np.int32)
    unique_pixels, indexes, counts = np.unique(pixels, return_index=True, return_counts=True, axis=0)

    mask = (unique_pixels[:, 0] >= 0) & (unique_pixels[:, 1] >= 0) & \
           (unique_pixels[:, 0] < img_width) & (unique_pixels[:, 1] < img_height)
    depth_indexes = unique_pixels[mask]
    depthmap[depth_indexes[:, 1], depth_indexes[:, 0]] = points[indexes[mask], 2]

    return depthmap


def pixel2world(x, y, z, img_width, img_height, fx, fy, cx=None, cy=None):
    """Converts image coordinates to 3D real world coordinates using depth values

    Parameters
    ----------
    x: np.array
        Array of X image coordinates.
    y: np.array
        Array of Y image coordinates.
    z: np.array
        Array of depth values for the whole image.
    img_width: int
        Width image dimension.
    img_height: int
        Height image dimension.
    fx: float
        Focal of the camera over X axis.
    fy: float
        Focal of the camera over Y axis.
    cx: float
        X coordinate of principal point of the camera.
    cy: float
        Y coordinate of principal point of the camera.

    Returns
    -------
    w_x: np.array
        Array of X world coordinates.
    w_y: np.array
        Array of Y world coordinates.
    w_z: np.array
        Array of Z world coordinates.

    """
    if cx is None:
        cx = img_width / 2
    if cy is None:
        cy = img_height / 2
    w_x = (x - cx) * z / fx
    w_y = (cy - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def world2pixel(x, y, z, img_width, img_height, fx, fy, cx=None, cy=None):
    if cx is None:
        cx = img_width / 2
    if cy is None:
        cy = img_height / 2
    p_x = x * fx / z + cx
    p_y = cy - y * fy / z
    return p_x, p_y


def points2pixels(points, img_width, img_height, fx, fy, cx=None, cy=None):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy, cx, cy)
    return pixels


def hole_filling(depthmap, kernel_size=5):
    """
    Depth map (small-)hole filling
    Args:
        depthmap (np.ndarray): depth map with dtype np.uint8 (1 or 3 channels) or np.float32 (1 channel)
    Returns:
        np.ndarray: hole-filled image
    """
    orig_shape_len = len(depthmap.shape)
    assert depthmap.dtype == np.uint8 or depthmap.dtype == np.uint16 or \
           (depthmap.dtype == np.float32 and orig_shape_len == 2) or \
           (depthmap.dtype == np.float32 and orig_shape_len == 3 and depthmap.shape[2] == 1)
    assert orig_shape_len == 2 or (orig_shape_len == 3 and depthmap.shape[2] in (1, 3))

    if orig_shape_len == 3:
        depthmap = depthmap[:, :, 0]
    mask = (depthmap > 0).astype(np.uint8)
    mask_filled = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size)))
    points_to_fill = (mask_filled * (depthmap == 0)).astype(np.uint8)
    if depthmap.dtype == np.float32:
        depthmap = cv2.inpaint(depthmap, points_to_fill, 2, cv2.INPAINT_NS)
    else:
        depthmap = cv2.inpaint(depthmap, points_to_fill, 2, cv2.INPAINT_TELEA)
    if orig_shape_len == 3:
        depthmap = np.expand_dims(depthmap, -1)

    return depthmap
