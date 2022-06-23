import numpy as np

from utils.geometry import x_rot, y_rot


def get_camera_position(camera_name, camera_params):
    """Gets camera rotations around X-Y axes and translation information

    Parameters
    ----------
    camera_name: str
        Camera position (central, left or right).
    camera_params: dict
        Camera parameters.

    Returns
    -------
    rot_x: np.array
        Rotation matrix around X axis.
    rot_y: np.array
        Rotation matrix around Y axis.
    tr: list
        Translation vector.
    """
    # camera-to-robot distance vector on the pavement plane
    tr = camera_params[camera_name]['tr_camera_to_base'].copy()
    # tilt angle of the camera with regard to the robot base
    tilt_angle = camera_params[camera_name]['tilt_angle']
    # offset of the camera center with regard to the robot base
    lateral_offset = camera_params[camera_name]['offset']

    rot_x = x_rot(np.radians(tilt_angle))
    hypo = np.sqrt(np.square(tr[0]) + np.square(tr[2]))
    theta = np.radians(90 - np.degrees(np.arccos(np.abs(tr[0]) / hypo)))
    if tr[0] > 0:  # if camera is on the right the angle should be negative (counterclockwise rotation)
        theta *= -1
    tr[0] += lateral_offset  # lateral offset does not affect "theta" angle computation

    # rototranslation of the robot with respect of the camera
    rot_y = y_rot(theta)
    rottr = np.eye(4)
    rottr[:3, :3] = rot_y @ rot_x
    rottr[:3, 3] = tr

    return rottr


def apply_depth_normalization_16bit_image(img, norm_type):
    """Applies normalization over 16bit depth images

    Parameters
    ----------
    img: np.array
        16bit depth image.
    norm_type: str
        Type of normalization (min_max, mean_std supported).

    Returns
    -------
    tmp: np.array
        Normalized and 16bit depth image (zero-centered in the case of mean_std normalization).

    """
    if norm_type == "min_max":
        min_value = 0
        max_value = 5000
        tmp = (img - min_value) / (max_value - min_value)
    elif norm_type == "mean_std":
        tmp = (img - img.mean()) / img.std()
    elif norm_type == "batch_mean_std":
        raise NotImplementedError
    elif norm_type == "dataset_mean_std":
        raise NotImplementedError
    else:
        raise NotImplementedError
    return tmp