import pickle

import trails.hyptrails as ht
from scipy.sparse import csr_matrix

from collections import OrderedDict


import numpy as np
import trails.mtmc.ml.deterministic.default as deterministic

import sys

# settings

n_states = 4604
ks = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000, 3000000, 10000000, 30000000, 100000000]


# load transitions
with open("tmp/wikispeedia-transitions.p", "rb") as f:
    transitions = pickle.load(f)

# load cosine hypothesis
with open("tmp/wikispeedia-hyp_cos.p", "rb") as f:
    hyp_cos = pickle.load(f)

# load degree hypothesis
with open("tmp/wikispeedia-hyp_deg.p", "rb") as f:
    hyp_deg = pickle.load(f)

experiments = {
    "meta": {
        "ks": ks
    },
    "results": OrderedDict()
}

hyp_links = hyp_cos.copy()
hyp_links.data[:] = 1
sums = hyp_links.sum(axis=1)
sums[sums == 0] = 1
hyp_links = csr_matrix(hyp_links / sums)

transition_counts = csr_matrix((np.ones(transitions.shape[0]), (transitions[:, 0], transitions[:, 1])))

# test link hypothesis
experiments["results"]["links"] = [ht.evidence_markov_matrix(n_states, transition_counts, hyp_links * k,
    smoothing=1) for k in ks]

# test cosine hypothesis
e_cos = [ht.evidence_markov_matrix(n_states, transition_counts, hyp_cos * k,
    smoothing=1) for k in ks]
experiments["results"]["cos"] = e_cos
print("e_cos:               ", e_cos)

# test degree hypothesis
e_deg = [ht.evidence_markov_matrix(n_states, transition_counts, hyp_deg * k,
    smoothing=1) for k in ks]
experiments["results"]["deg"] = e_deg
print("e_deg:               ", e_deg)

# test at1 hypothesis
with open("tmp/wikispeedia-p_gt-strict-at1.p", "rb") as f:
    p_gt = pickle.load(f)

hyp_deg_cos = np.array([hyp_deg, hyp_cos])
e_deg_cos_at1 = [deterministic.log_ml(transitions, p_gt, hyp_deg_cos * k,
    smoothing=1) for k in ks]
experiments["results"]["deg_cos_at1"] = e_deg_cos_at1
print("e_deg_cos_at1:       ", e_deg_cos_at1)


# test at2 hypothesis
with open("tmp/wikispeedia-p_gt-strict-at2.p", "rb") as f:
    p_gt = pickle.load(f)

hyp_deg_cos = np.array([hyp_deg, hyp_cos])
e_deg_cos_at2 = [deterministic.log_ml(transitions, p_gt, hyp_deg_cos * k,
    smoothing=1) for k in ks]
experiments["results"]["deg_cos_at2"] = e_deg_cos_at2
print("e_deg_cos_at2:       ", e_deg_cos_at2)

hyp_deg_deg = np.array([hyp_deg, hyp_deg])
experiments["results"]["deg_deg_at2"] = [deterministic.log_ml(transitions, p_gt, hyp_deg_deg * k,
    smoothing=1) for k in ks]

hyp_cos_cos = np.array([hyp_cos, hyp_cos])
experiments["results"]["cos_cos_at2"] = [deterministic.log_ml(transitions, p_gt, hyp_cos_cos * k,
    smoothing=1) for k in ks]

hyp_cos_deg = np.array([hyp_cos, hyp_deg])
experiments["results"]["cos_deg_at2"] = [deterministic.log_ml(transitions, p_gt, hyp_cos_deg * k,
    smoothing=1) for k in ks]


# test at3 hypothesis
with open("tmp/wikispeedia-p_gt-strict-at3.p", "rb") as f:
    p_gt = pickle.load(f)

hyp_deg_cos = np.array([hyp_deg, hyp_cos])
e_deg_cos_at3 = [deterministic.log_ml(transitions, p_gt, hyp_deg_cos * k,
    smoothing=1) for k in ks]
experiments["results"]["deg_cos_at3"] = e_deg_cos_at3
print("e_deg_cos_at3:       ", e_deg_cos_at3)


# save experiments
with open("out/" + prefix + "wikispeedia-experiments.p", "wb") as f:
    pickle.dump(experiments, f)
