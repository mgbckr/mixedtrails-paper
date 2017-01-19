import numpy as np
import trails.utils as utils


def expand(grouped_transition_probabilities, number_of_groups_list, group_dimension_index):
    """
    Expands a single grouped_transition_probabilities matrix to cover a cartesian product of groups.
    """
    index1 = np.append(number_of_groups_list, [1])[(group_dimension_index + 1):(len(number_of_groups_list) + 1)]
    index2 = np.append([1], number_of_groups_list)[0:(group_dimension_index + 1)]
#    print(index1)
#    print(index2)
    dim1 = np.prod(index1)
    dim2 = np.prod(index2)

    indexes = np.tile(np.repeat(np.arange(0, len(grouped_transition_probabilities)), dim1), dim2)
#    print("Indices")
#    print(indexes)
#    print(len(indexes))

    return np.array(grouped_transition_probabilities)[indexes]


def random(adjacency_matrix, state_properties):
    n_states = len(state_properties)      
    transition_probabilities = np.ones([n_states, n_states])
    return utils.norm1_2d(transition_probabilities)


def links(adjacency_matrix, state_properties):
    return utils.norm1_2d(adjacency_matrix)


def group_homo(adjacency_matrix, state_properties):
    
    # initialize transition probabilities
    state_classes = np.sort(np.unique(state_properties))
    n_states = len(state_properties)
    group_transition_probabilities = np.zeros([len(state_classes), n_states, n_states])
    
    for state_class_index, group_matrix in enumerate(group_transition_probabilities):
        for state, transition_probabilities in enumerate(group_matrix):
            
            available_destinations = utils.available_destinations(adjacency_matrix, state)
            filtered_destinations = list(filter(
                lambda d: state_classes[state_class_index] == state_properties[d],
                available_destinations))
            
            # if we don't have destinations of the same type, we choose randomly
            if len(filtered_destinations) is 0:
                filtered_destinations = available_destinations
            
            probability = 1 / len(filtered_destinations)
            for d in filtered_destinations:
                transition_probabilities[d] = probability
                
    return group_transition_probabilities
                        

def group_homo_weighted(weight, adjacency_matrix, state_properties):
    
    # initialize transition probabilities
    state_classes = np.sort(np.unique(state_properties))
    n_states = len(state_properties)
    group_transition_probabilities = np.zeros([len(state_classes), n_states, n_states])
            
    for state_class_index, group_matrix in enumerate(group_transition_probabilities):
        for state, transition_probabilities in enumerate(group_matrix):
            
            available_destinations = utils.available_destinations(adjacency_matrix, state)
            
            def choose(a,b):
                if a == b:
                    return weight
                else:
                    return 1
                    
            state_class = state_classes[state_class_index]
            
            weighted_destinations = np.array([choose(state_class, state_properties[d]) for d in available_destinations])
           
            probabilities = weighted_destinations / sum(weighted_destinations)
            
            group_transition_probabilities[state_class_index][state][available_destinations] = probabilities
            
    return group_transition_probabilities