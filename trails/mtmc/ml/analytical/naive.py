import trails.hyptrails as hyptrails
from trails.mtmc.common import *
from scipy.sparse import csr_matrix
import scipy.misc
import math


def log_ml(
        transitions, group_assignment_p, alpha, smoothing=0):
    """
    Calculates the marginal likelihood of the MTMC model analytically.
    This is highly inefficient since it envolves calculating the
    marginal likelihood for all possible group assignments which
    grows exponentially with respect to the number of transitions (``n_groups ** n_transitions``)!

    This particular implementaion takes advantage of the "logsumexp trick",
    which allows to work with log-probabilities hopefully reducing numeric errors.
    However the sums are done recursively with arrays of size ``n_groups``.
    This may cause issues if the values in any array are not on the same scale.

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
    alpha: ndarray
        The dirichlet prior parameters of the model.
        They have the same dimenstion as the transtition probabilities.
        Thus the shape is ``(g,m,m)``,
        where ``g`` is the number of groups and ``m`` is the number of states.
    smoothing: float
        Adds a constant to alpha during calculations.
        Usually, this is used with sparse alpha matrices, i.e.,
        we add the "proto-prior" by setting ``smoothing=1``.
    """

    # derive some constants
    n_transitions = transitions.shape[0]
    n_groups = alpha.shape[0]
    n_states = alpha.shape[1]

    # update alpha
    alpha += smoothing

    # recursive function to go though all group assignments
    def rec(weight, transition, group_assignments):

        # we have assigned a group to every transition,
        # thus we calculate the marginal likelihood
        # for this group assignment
        if transition is n_transitions:

            # calculate transtition counts for this group assignment
            counts = calc_transition_counts_asarray(transitions, group_assignments, n_groups, n_states)

            # weighted marginal likelihood
            return weight + sum(
                [hyptrails.evidence_markov_matrix(
                    n_states,
                    csr_matrix(group_counts),
                    csr_matrix(alpha[group,]),
                    smoothing=0)
                 for group, group_counts in enumerate(counts)])

        # we are still in the process of assigning a group
        # to each transition
        else:
            likelihoods = np.ones(n_groups)
            for group in range(0, n_groups):
                group_assignments[transition] = group
                if group_assignment_p[transition, group] != 0:
                    likelihoods[group] = rec(
                        weight + math.log(group_assignment_p[transition, group]),
                        transition + 1,
                        group_assignments)
            return scipy.misc.logsumexp([l for l in likelihoods if l <= 0])  # likelihoods can never be positive

    return rec(0, 0, np.full(n_transitions, -1, dtype="int8"))

