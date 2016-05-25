import numpy as np

def transition_matrix(walks, number_of_states):
    transition_matrix = np.zeros([number_of_states, number_of_states])
    for walk in walks:
        for i in np.arange(1, len(walk[1])):
            src = walk[1][i - 1]
            dst = walk[1][i]
            transition_matrix[src][dst] += 1 
    return transition_matrix

def grouped_transition_matrix(\
        f_group_assignment, \
        number_of_groups, \
        walks, \
        adjacency_matrix, \
        state_properties):
    
    number_of_states = len(state_properties)
    transition_matrix = np.zeros([number_of_groups, number_of_states, number_of_states])
    
    for walk in walks:
        
        walker_properties = walk[0]
        walker_states = walk[1]
        
        for i in np.arange(1, len(walker_states)):
            
            history = walker_states[0:i]
            group = f_group_assignment(\
                history, \
                walker_properties, \
                adjacency_matrix, \
                state_properties)
            
            src = walk[1][i - 1]
            dst = walk[1][i]
      
            transition_matrix[group][src][dst] += 1
        
    return transition_matrix
            
            