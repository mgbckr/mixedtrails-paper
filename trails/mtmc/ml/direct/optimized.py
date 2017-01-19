import trails.hyptrails as hyptrails
from trails.mtmc.common import *
from scipy.sparse import csr_matrix
import scipy.misc
import math


def log_ml(
        transitions, group_assignment_p, alpha, smoothing=0, n_samples=100):
    """
    Directly samples the marginal likelihood of the MTMC model
    from the corresponding, analytically derived formula.
    In particular the group assignments are sampled.

    This particular implementation takes advantage of the "logsumexp trick",
    which allows to work with log-probabilities hopefully reducing numeric errors.

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
        They have the same dimension as the transition probabilities.
        Thus the shape is ``(g,m,m)``,
        where ``g`` is the number of groups and ``m`` is the number of states.
    n_samples: int
        Number of Gibbs runs.
        (Default: 100)
    smoothing: float
        Adds a constant to alpha during calculations.
        Usually, this is used with sparse alpha matrices, i.e.,
        we add the "proto-prior" by setting ``smoothing=1``.
    """

    # derive variables
    n_transitions = transitions.shape[0]
    n_groups = alpha.shape[0]
    n_states = alpha[0].shape[0]

    # initialize samples
    samples = np.zeros(n_samples)

    # prepare alpha
    if len(alpha.shape) > 1:
        alpha = np.array([csr_matrix(group_alpha) for group_alpha in alpha])

    # initialize array for current group assignment
    group_assignments = np.empty(n_transitions, dtype="int8")

    # run sampler
    for i in range(0, n_samples):

        # sample group assignments
        for j in range(0, n_transitions):
            group_assignments[j] = int(np.random.multinomial(1, group_assignment_p[j, ]).argmax())

        # calculate transition counts
        for g in range(0, n_groups):
            selected_transitions = transitions[group_assignments == g, :]
            group_counts = csr_matrix(
                (np.ones(selected_transitions.shape[0]), (selected_transitions[:, 0], selected_transitions[:, 1])),
                (n_states, n_states))
            samples[i] += hyptrails.evidence_markov_matrix(
                n_states,
                group_counts,
                alpha[g],
                smoothing=smoothing)

    return scipy.misc.logsumexp(samples) - math.log(n_samples)
