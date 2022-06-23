from typing import Union

import numpy as np
import torch


def x_rot(alpha: float, clockwise: bool = False, pytorch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """Computes rotation matrix around X axis

    Parameters
    ----------
    alpha: float
        Rotation angle in radians.
    clockwise: bool
        Default rotation convention is counter-clockwise. In case the `clockwise` flag is set, the sign of `sin(alpha)`
        is reversed to rotate clockwise.
    pytorch: bool
        In case the `pytorch` flag is set, all operation are between torch tensors and a torch.Tensor is returned.
    Returns
    -------
    rot: Union[np.array, torch.Tensor]
        Rotation matrix around X axis.
    """
    if pytorch:
        cx = torch.cos(alpha)
        sx = torch.sin(alpha)
    else:
        cx = np.cos(alpha)
        sx = np.sin(alpha)

    if clockwise:
        sx *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([one, zero, zero], dim=1),
                         torch.stack([zero, cx, -sx], dim=1),
                         torch.stack([zero, sx, cx], dim=1)], dim=0)
    else:
        rot = np.asarray([[1., 0., 0.],
                          [0., cx, -sx],
                          [0., sx, cx]], dtype=np.float32)
    return rot


def y_rot(alpha: float, clockwise: bool = False, pytorch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """Computes rotation matrix around Y axis

    Parameters
    ----------
    alpha: float
        Rotation angle in radians.
    clockwise: bool
        Default rotation convention is counter-clockwise. In case the `clockwise` flag is set, the sign of `sin(alpha)`
        is reversed to rotate clockwise.
    pytorch: bool
        In case the `pytorch` flag is set, all operation are between torch tensors and a torch.Tensor is returned.
    Returns
    -------
    rot: Union[np.array, torch.Tensor]
        Rotation matrix around Y axis.
    """
    if pytorch:
        cy = torch.cos(alpha)
        sy = torch.sin(alpha)
    else:
        cy = np.cos(alpha)
        sy = np.sin(alpha)

    if clockwise:
        sy *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([cy, zero, sy], dim=1),
                         torch.stack([zero, one, zero], dim=1),
                         torch.stack([-sy, zero, cy], dim=1)], dim=0)
    else:
        rot = np.asarray([[cy, 0., sy],
                          [0., 1., 0.],
                          [-sy, 0., cy]], dtype=np.float32)
    return rot


def z_rot(alpha: float, clockwise: bool = False, pytorch: bool = False) -> Union[np.ndarray, torch.Tensor]:
    """Computes rotation matrix around Z axis

    Parameters
    ----------
    alpha: float
        Rotation angle in radians.
    clockwise: bool
        Default rotation convention is counter-clockwise. In case the `clockwise` flag is set, the sign of `sin(alpha)`
        is reversed to rotate clockwise.
    pytorch: bool
        In case the `pytorch` flag is set, all operation are between torch tensors and a torch.Tensor is returned.
    Returns
    -------
    rot: Union[np.array, torch.Tensor]
        Rotation matrix around Z axis.
    """
    if pytorch:
        cz = torch.cos(alpha)
        sz = torch.sin(alpha)
    else:
        cz = np.cos(alpha)
        sz = np.sin(alpha)

    if clockwise:
        sz *= -1

    if pytorch:
        zero = torch.zeros(1)
        one = torch.ones(1)
        rot = torch.cat([torch.stack([cz, -sz, zero], dim=1),
                         torch.stack([sz, cz, zero], dim=1),
                         torch.stack([zero, zero, one], dim=1)], dim=0)
    else:
        rot = np.asarray([[cz, -sz, 0.],
                          [sz, cz, 0.],
                          [0., 0., 1.]], dtype=np.float32)

    return rot