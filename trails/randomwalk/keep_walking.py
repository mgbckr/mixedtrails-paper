from functools import partial 

def fixed(n, walk, walker, adjacency_matrix, state_properties):
    return len(walk) <= n

def init_fixed(n):
    """
    Returns a keep_walking_n function with a fixed "n".
    """
    return partial(fixed, n)