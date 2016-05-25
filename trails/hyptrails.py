# simple HypTrails implementation taking numpy matrices as input

import math
import numpy as np

def evidence_categorical(n, alpha, smoothing = 1, n_pseudoalpha = 0, pseudoalpha = 1):
    """
    Evidence for a categorical distribution, i.e., a multi-faced die.
    """
    
    numberOfStates = n.size
    sumOfNs = n.sum()
    sumOfAlphas = alpha.sum() + (numberOfStates * smoothing) + n_pseudoalpha * pseudoalpha
        
    def logGamma(n, alpha):
        smoothedAlpha = alpha + smoothing
        return (math.lgamma(smoothedAlpha), math.lgamma(n + smoothedAlpha))
        
    gammaValues = [logGamma(ni,ai) for ni, ai in zip(n, alpha)]
    
    sumOfLogGammaAlphas = sum([i for i,_ in gammaValues])
    sumOfLogGammaNAlphas = sum([i for _,i in gammaValues]) 
    
    return math.lgamma(sumOfAlphas) \
        + sumOfLogGammaNAlphas \
        - sumOfLogGammaAlphas \
        - math.lgamma(sumOfNs + sumOfAlphas)
        
def evidence_markov(transitions, hyp, smoothing = 1, n_pseudoalpha = 0, pseudoalpha = 1):
    """
    Evidence for a first order markov model
    """
    mapped = [evidence_categorical(transitions_i, hyp_i, smoothing, n_pseudoalpha, pseudoalpha) \
        for transitions_i,hyp_i in zip(transitions, hyp)]
    return np.array(mapped).sum()

def evidence_groups(transitions, hyp, smoothing = 1, n_pseudoalpha = 0, pseudoalpha = 1):
    """
    Evidence for a first order markov model with groups. i.e.,
    n and alpha are arrays of transition matrices/alpha parameters, one for each group.
    """
    evidences = np.array([ \
            evidence_markov(transitions_g, hyp_g, smoothing, n_pseudoalpha, pseudoalpha) 
                  for transitions_g, hyp_g in zip(transitions, hyp)])
    print(evidences)
    return evidences.sum(axis = 0)
    
def evidence_groups_single_hypothesis(transitions, hyp, smoothing = 1, n_pseudoalpha = 0, pseudoalpha = 1):
    """
    Evidence for a first order markov model with groups but only one hypothesis. i.e.,
    n is an arrays of transition matrices, one for each group, but there is only one hypothesis for all group.
    """
    return \
        np.array([ \
            evidence_markov(transitions_g, hyp, smoothing, n_pseudoalpha, pseudoalpha) 
                  for transitions_g in transitions]) \
        .sum(axis = 0)
    
def evidence_categorical_kappas(transitions, hyp, kappas, smoothing = 1, n_pseudoalpha = 0, pseudoalpha = 1):
    return np.array([evidence_categorical(transitions, hyp * k, smoothing, n_pseudoalpha, pseudoalpha) \
                     for k in kappas])
    
def evidence_markov_kappas(transitions, hyp, kappas, smoothing = 1, n_pseudoalpha = 0, pseudoalpha = 1):
    return np.array([evidence_markov(transitions, hyp * k, smoothing, n_pseudoalpha, pseudoalpha) \
                     for k in kappas])

def evidence_groups_kappas(transitions, hyp, kappas, smoothing = 1, n_pseudoalpha = 0, pseudoalpha = 1):
    return np.array([evidence_groups(transitions, hyp * k, smoothing, n_pseudoalpha, pseudoalpha) \
                     for k in kappas])

def evidence_groups_single_hypothesis_kappas(transitions, hyp, kappas, smoothing = 1, n_pseudoalpha = 0, pseudoalpha = 1):
    return np.array([evidence_groups_single_hypothesis(transitions, hyp * k, smoothing, n_pseudoalpha, pseudoalpha) \
                     for k in kappas])

