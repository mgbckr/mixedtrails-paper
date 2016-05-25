import numpy as np
from functools import partial

def random(walk, walker, adjacency_matrix, state_properties):
    """
    Walker chooses destination randomly.
    """
    available_destinations = RandomWalk.available_destinations(adjacency_matrix, walk[-1])
    if len(available_destinations) is 0:
        return None
    return np.random.choice(available_destinations)



def matrix(transition_matrix, walk, walker, adjacency_matrix, state_properties):
    current_state = walk[-1]
    transition_probabilities = transition_matrix[current_state]
    next_state = np.random.choice(len(transition_probabilities), p = transition_probabilities)
    return next_state

def init_matrix(transition_matrix):
    return partial(matrix, transition_matrix)



def grouped_matrix(f_group, grouped_transition_matrix, walk, walker, adjacency_matrix, state_properties):
    group = f_group(walk, walker, adjacency_matrix, state_properties)
    transition_matrix = grouped_transition_matrix[group]
    return matrix(transition_matrix, walk, walker, adjacency_matrix, state_properties)

def init_grouped_matrix(f_group, grouped_transition_matrix):
    return partial(grouped_matrix, f_group, grouped_transition_matrix)
    
    

def homo(walk, walker, adjacency_matrix, state_properties):
    """
    Walker will always prefer own color. Chooses randomly otherwise.
    """
    
    # the group
    group = walker
    
    # the transition
    available_destinations = RandomWalk.available_destinations(adjacency_matrix, walk[-1])
    filtered_destinations = list(filter(lambda d: group == state_properties[d], available_destinations))
    if len(filtered_destinations) is 0:
        return np.random.choice(available_destinations)
    else:
        return np.random.choice(filtered_destinations)

    
    
def homo_weighted(weight, walk, walker, adjecency_matrix, state_properties):
    """
    Walker gives a weight to the ... TODO
    """
    
    # the group
    group = walker

    # the transition
    available_destinations = RandomWalk.available_destinations(adjacency_matrix, walk[-1])
       
    def choose(a,b):
        if a is b:
            return weight
        else:
            1
    weighted_destinations = [choose(state_properties[d], walker) for d in available_destinations]
    
    destination_probabilities = weighted_destinations / sum(weighted_destinations)
    
    random_state_index = \
        np.random.choice(len(weighted_destinations), 1, destination_probabilities)[0]
        
    return available_destinations[random_state_index]
    
    
    
def init_homo_weighted(weight):
    return partial(homo_weighted, weight)
    
    
    

    
def memory(walk, walker, adjacency_matrix, state_properties):
    """
    Walker looks at her history and does a majority vote on the "state" she is in.
    E.g., if most history nodes are red, she will try to go to red.
    If there is a draw, she will choose randomly.
    """
    
    group = transition_group_memory(walk, walker, adjacency_matrix, state_properties)
    
    available_destinations = RandomWalk.available_destinations(adjacency_matrix, walk[-1])
    
    from collections import Counter
    most_common_state_types = Counter([state_properties[state] for state in walk]).most_common(2)
    
    if len(most_common_state_types) is 0 or most_common_state_types[0][1] is most_common_state_types[1][1]:
        return np.random.choice(available_destinations)
    else:
        most_common_state_type = most_common_state_types[0][0]
        filtered_destinations = list(filter(\
            lambda d: most_common_state_type == state_properties[d], \
            available_destinations))
        if len(filtered_destinations) is 0:
            return np.random.choice(available_destinations)
        else:
            return np.random.choice(filtered_destinations)
        
