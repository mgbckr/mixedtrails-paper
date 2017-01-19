from functools import partial
import numpy as np

# Functions return true when the walker is supposed to keep walking.
# contract:  walk, walker, adjacency_matrix, state_properties


def fixed(n, walk, walker, adjacency_matrix, state_properties):
    """
    Stops the walker when a certain walk length `n` is reached.
    """
    return len(walk) <= n


def init_fixed(n):
    """
    Returns a keep_walking_n function with a fixed "n".
    """
    return partial(fixed, n)


def stop_homo(walk, walker, adjacency_matrix, state_properties):
    """
    Stops the walker when there is no state with the same properties as the walker.
    """
    current_state = walk[-1]
    destinations = adjacency_matrix[current_state, :]
    reachable_properties = np.array(state_properties[destinations > 0])
    return any(reachable_properties == walker)


def init_and(keep_walking_array):
    def keep_walking_and(walk, walker, adjacency_matrix, state_properties):
        return all([keep_walking(walk, walker, adjacency_matrix, state_properties)
                    for keep_walking in keep_walking_array])
    return keep_walking_and
