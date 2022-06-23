import typing as tp

import numpy as np


chains_ixs = ([14, 13, 12, 11, 10, 9, 8,
               15,
               1, 2, 3, 4, 5, 6, 7],
              # wrist_l, low_forearm_l, up_forearm_l, low_elbow_l,
              # up_elbow_l, low_shoulder_l, up_shoulder_l,
              # base,
              # wrist_r, low_forearm_r, up_forearm_r, low_elbow_r,
              # up_elbow_r, low_shoulder_r, up_shoulder_r
              [0, 15],
              # base, head
              )


def get_chain_dots(dots: np.ndarray, chain_dots_indexes: tp.List[int]) -> np.ndarray:  # chain of dots
    return dots[chain_dots_indexes]


def get_chains(dots: np.ndarray, arms_chain_ixs: tp.List[int], torso_chain_ixs: tp.List[int]):
    return (get_chain_dots(dots, arms_chain_ixs),
            get_chain_dots(dots, torso_chain_ixs))


def subplot_nodes(dots: np.ndarray, ax, c='red'):
    return ax.scatter3D(dots[:, 0], dots[:, 2], dots[:, 1], c=c)


def subplot_bones(chains: tp.Tuple[np.ndarray, ...], ax, c='greens'):
    return [ax.plot(chain[:, 0], chain[:, 2], chain[:, 1], c=c) for chain in chains]
