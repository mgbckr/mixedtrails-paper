import numpy as np


def norm1_2d(np_array2d):
    """
    Normalize each "row" of a 2 dimensional numpy array using the L1 norm
    """
    return np_array2d / np_array2d.sum(axis=1)[:, np.newaxis]


def norm1_3d(np_array3d):
    """
    Normalize each "row" in each sub array in a 3 dimensional numpy array using the L1 norm
    """
    return np.nan_to_num(np_array3d / np_array3d.sum(axis=2)[:, :, np.newaxis])


def available_destinations(adjacency_matrix, state):
    return np.where(adjacency_matrix[state,] == 1)[0]