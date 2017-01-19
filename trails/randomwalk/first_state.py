import numpy as np


def random(walker, adjacency_matrix, state_properties):
    return np.random.randint(0, len(state_properties))


def same_color(walker, adjacency_matrix, state_properties):
    indices = [i for i, x in enumerate(state_properties) if x == walker]
    return indices[np.random.randint(0, len(indices))]
