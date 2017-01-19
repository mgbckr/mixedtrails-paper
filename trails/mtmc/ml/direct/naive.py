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
        They have the same dimenstion as the transtition probabilities.
        Thus the shape is ``(g,m,m)``,
        where ``g`` is the number of groups and ``m`` is the number of states.
    n_samples: int
        Number of Gibbs runs.
        (Default: 100)
    """

    # convert (numpy) arrays to array of csr_matrices if necessary
    if len(alpha.shape) > 1:
        alpha = np.array([csr_matrix(a) for a in alpha])

    # derive variables
    n_groups = alpha.shape[0]
    n_states = alpha[0].shape[0]

    # run sampler
    samples = np.empty(n_samples)
    for i in range(n_samples):

        # sample group assignments
        group_assignments = np.array([int(np.random.multinomial(1, p_g).argmax()) for p_g in group_assignment_p])

        # calculate transition counts
        counts = calc_transition_counts_ascsr(transitions, group_assignments, n_groups, n_states)

        # calculate marginal likelihood
        samples[i] = sum(
            [hyptrails.evidence_markov_matrix(
                n_states,
                group_counts,
                alpha[group],
                smoothing=smoothing)
             for group, group_counts in enumerate(counts)])

    return scipy.misc.logsumexp(samples) - math.log(n_samples)
