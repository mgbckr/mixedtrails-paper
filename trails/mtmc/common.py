import numpy as np
import scipy.misc
import scipy.special
import scipy.stats as stats
from scipy.sparse import csr_matrix
import math
from sklearn.utils.extmath import cartesian


def calc_transition_counts(
        transitions: np.ndarray,
        group_assignments: np.ndarray,
        n_groups: int,
        n_states: int,
        as_csr=True) \
        -> np.ndarray:

    if as_csr:
        return calc_transition_counts_ascsr(transitions, group_assignments, n_groups, n_states)
    else:
        return calc_transition_counts_asarray(transitions, group_assignments, n_groups, n_states)


def calc_transition_counts_asarray(
        transitions: np.ndarray,
        group_assignments: np.ndarray,
        n_groups: int,
        n_states: int) \
        -> np.ndarray:
    """
    Counts the occurrence of each transition per group.

    Parameters
    ----------
    transitions: ndarray
        Transitions betweens states described by their source and destination state.
        Thus, the shape is: ``(n,2)``,
        where ``n`` is the number of states and
        ``transitions[i,] = [source_state_i, destination_state_i]``.
    group_assignments: ndarray
        Group assignments, i.e., one group index for each transition.
        Thus the shape is ``(n,1)``.
    n_groups: int
        Number of groups.
    n_states: int
        Number of states.

    Notes
    -----
    We assume that groups as well as states states are indexed starting at ``0``.

    """
    counts = np.zeros((n_groups, n_states, n_states))
    for t, group in enumerate(group_assignments):
        counts[group, transitions[t, 0], transitions[t, 1]] += 1
    return counts


def calc_transition_counts_ascsr(
        transitions: np.ndarray,
        group_assignments: np.ndarray,
        n_groups: int,
        n_states: int) \
        -> np.ndarray:
    """
    Counts the occurrence of each transition per group.

    Parameters
    ----------
    transitions: ndarray
        Transitions betweens states described by their source and destination state.
        Thus, the shape is: ``(n,2)``,
        where ``n`` is the number of states and
        ``transitions[i,] = [source_state_i, destination_state_i]``.
    group_assignments: ndarray
        Group assignments, i.e., one group index for each transition.
        Thus the shape is ``(n,1)``.
    n_groups: int
        Number of groups.
    n_states: int
        Number of states.

    Notes
    -----
    We assume that groups as well as states states are indexed starting at ``0``.

    """
    counts = np.empty(n_groups, dtype=np.object)
    for g in range(n_groups):
        selected = transitions[group_assignments == g, :]
        counts[g] = csr_matrix(
            (np.ones(selected.shape[0]), (selected[:, 0], selected[:, 1])),
            shape=(n_states, n_states))
    return counts


def calc_mixed_hypothesis(group_assignment_p, hyp):
    """
    Calculates mixed hypotheses for probabilistic group assignments.
    """
    mixed = np.empty(hyp.shape[0], dtype=np.object)
    for g in range(hyp.shape[0]):
        coeff = (group_assignment_p[:, g].reshape((len(group_assignment_p), 1)) * group_assignment_p).sum(axis=0)
        coeff /= sum(coeff)
        mixed[g] = sum([c * csr_matrix(h) for c, h in zip(coeff, hyp)])
    return mixed


def calc_cartesian_group_assignment_p(group_assignment_p_list):
    cart = np.array(
        [[
           np.prod(values)
           for values in cartesian(line)]
         for line in zip(*group_assignment_p_list)])
    return cart


def calc_cartesian_alpha(alpha, index, n_groups_list):
    if index < 0:
        return np.array([alpha for _ in range(np.prod(n_groups_list))])
    else:
        cart = cartesian([range(n_groups) for n_groups in n_groups_list])
        return np.array([alpha[i] for i in cart[:, index]])


def calc_log_l(
        transitions: np.ndarray,
        group_assignment_p: np.ndarray,
        transition_p: np.ndarray) \
        -> float:
    """
    The log likelihood of the MTMC model.

    Parameters
    ----------
    transitions: ndarray
        Transitions betweens states described by their source and destination state.
        Thus, the shape is: ``(n,2)``,
        where ``n`` is the number of states and
        ``transitions[i,] = [source_state_i, destination_state_i]``.
    group_assignment_p: ndarray
        Group assignment probabilities, i.e.,
        for each transition it holds a probability distribution over groups.
        Thus, the shape is ``(n,g)``,
        where ``n`` is the number of transitions and ``g`` is the number of groups.
    transition_p: ndarray
        The transition probabilities between all states for each group.
        Thus the shape is ``(g,m,m)``,
        where ``g`` is the number of groups and ``m`` is the number of states.
    """

    return sum(
        [scipy.misc.logsumexp(
            [math.log(group_p) + math.log(transition_p[group, src, dst])
             for group, group_p in enumerate(group_assignment_p[t, ]) if group_p != 0])
         for t, (src, dst) in enumerate(transitions)])


def calc_log_prior(
        transition_p: np.ndarray,
        alpha: np.ndarray) \
        -> float:
    """
    The prior probability of the transition probabilities
    given the dirichlet parameters of the prior
    for the MTMC model.

    Parameters
    ----------
    transition_p: ndarray
        The transition probabilities between all states for each group.
        Thus the shape is ``(g,m,m)``,
        where ``g`` is the number of groups and ``m`` is the number of states.
    alpha: ndarray
        The dirichlet prior parameters of the model.
        They have the same dimenstion as the transtition probabilities.
        Thus the shape is ``(g,m,m)``,
        where ``g`` is the number of groups and ``m`` is the number of states.
    """
    return sum(
        [sum(
            [stats.dirichlet.logpdf(src_transition_p, alpha[group, src, ])
             for src, src_transition_p in enumerate(group_transition_p)])
         for group, group_transition_p in enumerate(transition_p)])











