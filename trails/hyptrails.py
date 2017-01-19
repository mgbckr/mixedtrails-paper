# simple HypTrails implementation taking numpy matrices as input


def evidence_markov_matrix(number_of_states, transitions, hyp, smoothing=1):
    from scipy.special import gammaln
    """
    Evidence for a first order markov model using matrices.
    Inspired by https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part4.ipynb
    """
    transitions_prior = transitions + hyp

    evidence = 0
    evidence += gammaln(hyp.sum(axis=1) + number_of_states*smoothing).sum()
    evidence -= gammaln(transitions.sum(axis=1) + hyp.sum(axis=1) + number_of_states*smoothing).sum()
    evidence += gammaln(transitions_prior.data + smoothing).sum()
    
    gamma_smoothing = 0
    if smoothing > 0:
        gamma_smoothing = (len(transitions_prior.data) - len(hyp.data)) * gamma_smoothing
        
    evidence -= gammaln(hyp.data + smoothing).sum() + gamma_smoothing
    
    return evidence
