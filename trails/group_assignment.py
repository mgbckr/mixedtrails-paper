import numpy as np
from functools import partial


def walker(walk, walker, adjacency_matrix, state_properties):
    return walker


def memory(walk, walker, adjacency_matrix, state_properties):
    
    from collections import Counter
    most_common_state_types = Counter([state_properties[state] for state in walk]).most_common(2)

    if len(most_common_state_types) is 0 or \
            (len(most_common_state_types) >= 2 and
             most_common_state_types[0][1] is most_common_state_types[1][1]):
        return 0
    else:
        return most_common_state_types[0][0] + 1


def random(n, walk, walker, adjacency_matrix, state_properties):
    return np.random.choice(np.arange(n))


def init_random(n):
    return partial(random, n)


def proxy_count(number_of_groups_list):
    return np.prod(number_of_groups_list)


def proxy(group_assignment_list, offsets, walk, walker, adjacency_matrix, state_properties):
    groups = [group_assignment(walk, walker, adjacency_matrix, state_properties)
              for group_assignment in group_assignment_list]
    group = sum([g * o for g, o in zip(groups, offsets)])
#    print(offsets)
#    print(groups)
#    print(group)
    return group


def init_proxy(
        group_assignment_list,
        number_of_groups_list):

    offsets = np.cumprod(np.append(number_of_groups_list[1:], [1])[::-1])[::-1]
    return partial(proxy, group_assignment_list, offsets)
