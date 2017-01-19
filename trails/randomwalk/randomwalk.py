import trails.randomwalk.first_state as first_state
import trails.randomwalk.keep_walking as keep_walking
import trails.randomwalk.next_state as next_state


class RandomWalk:
    """Wrapper for some random walking"""
    
    def __init__(self, adjacency_matrix, state_properties):
        self.adjacency_matrix = adjacency_matrix
        self.state_properties = state_properties
    
    def walk(self, 
            walker_properties,
            f_next_state=next_state.random,
            f_keep_walking=keep_walking.init_fixed(10),
            f_first_state=first_state.random):

        # initialize container for walks
        walks = [] 

        # do the walking
        for walker in walker_properties:
            
            # set the first state for the walker
            first_state = f_first_state(walker, self.adjacency_matrix, self.state_properties)    
            
            # initialize walk
            walk = [first_state]
            
            # walk the walk
            while f_keep_walking(walk, walker, self.adjacency_matrix, self.state_properties):
                
                # get next state
                next_state = f_next_state(walk, walker, self.adjacency_matrix, self.state_properties)
                
                # quit if we could not determine the next state
                if next_state is None:
                    break
                else:
                    # append the new state to the walk
                    walk.append(next_state)
            
            # log walk
            walks.append((walker, walk))

        # return walks
        return walks
