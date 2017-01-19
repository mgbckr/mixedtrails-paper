import numpy as np

from trails import hyptrails
from scipy.sparse import csr_matrix


def log_ml_counts(
        transition_counts: np.ndarray,
        alpha: np.ndarray,
        smoothing: float=0):

    # convert (numpy) arrays to array of csr_matrices if necessary

    if len(transition_counts.shape) > 1:
        transition_counts = np.array([csr_matrix(counts) for counts in transition_counts])

    if len(alpha.shape) > 1:
        alpha = np.array([csr_matrix(a) for a in alpha])

    # calculate marginal likelihood using standard HypTrails

    n_states = transition_counts[0].shape[0]
    return sum(
        [hyptrails.evidence_markov_matrix(
            n_states,
            group_counts,
            alpha[group, ],
            smoothing=smoothing)
         for group, group_counts in enumerate(transition_counts)])


def log_ml(
        transitions: np.ndarray,
        group_assignment_p: np.ndarray,
        alpha: np.ndarray,
        smoothing: float=0):

    # derive deterministic group assignments
    group_assignments = np.array([np.argmax(p) for p in group_assignment_p])

    # prepare alpha
    if len(alpha.shape) > 1:
        alpha = np.array([csr_matrix(group_alpha) for group_alpha in alpha])

    # calculate transition counts
    n_groups = alpha.shape[0]
    transition_counts = np.empty(n_groups, dtype=np.object)
    for g in range(n_groups):
        t_selected = transitions[group_assignments == g, ]
        transition_counts[g] = csr_matrix(
            (np.ones(t_selected.shape[0]), (t_selected[:, 0], t_selected[:, 1])),
            shape=alpha[0].shape)

    # calculate marginal likelihood
    return log_ml_counts(transition_counts, alpha, smoothing)
