import csv
import urllib
from collections import defaultdict

import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

import pickle

# load sequences
print("Loading sequences ...")
sequences = []
sequence_file = "data/wikispeedia/wikispeedia_paths-and-graph/paths_finished.tsv"
rows = (row for row in open(sequence_file) if not row.startswith('#'))
for line in csv.reader(rows, delimiter='\t'):
    if len(line) == 0:
        continue
    seq = line[3].split(";")
    # for simplicity, let us remove back clicks
    seq = [urllib.parse.unquote(x) for x in seq if x != "<"]
    sequences.append(seq)
print("Number of sequences:", len(sequences))


# load article map
print("Article map ...")
article_map = {}
with open("data/wikispeedia/wikispeedia_paths-and-graph/articles.tsv") as file:
    index = 0
    for line in file:
        stripped = line.strip()
        if stripped.startswith('#') or len(stripped) == 0:
            continue
        article = urllib.parse.unquote(stripped)
        article_map[article] = index
        index += 1
print("Number of articles:", len(article_map))


# map article sequences to index sequences
print("Indexing sequences ...")
index_sequences = [[article_map[article] for article in sequence] for sequence in sequences]


# load shortest path matrix
print("Loading shortest path matrix ...")
shortest_path_matrix = []
with open("data/wikispeedia/wikispeedia_paths-and-graph/shortest-path-distance-matrix.txt") as file:
    index = 0
    for line in file:
        stripped = line.strip()
        if stripped.startswith('#') or len(stripped) == 0:
            continue


        def convert(x):
            if x is "_":
                return -1
            else:
                return int(x)


        shortest_path_matrix.append([convert(x) for x in stripped])
        index += 1

# filter by optimal length == 3
print("Filtering sequences with optimal length 3 ...")
filtered_index_sequences = [s for s in index_sequences if shortest_path_matrix[s[0]][s[-1]] == 3]
print("Filtered sequences:", len(filtered_index_sequences))

# filter 3 to 8 clicks (<=> 4 to 9 articles) (inclusive), i.e.,
print("Filtering sequences with 3-8 clicks ...")
filtered_index_sequences = [s for s in filtered_index_sequences if 4 <= len(s) <= 9]
print("Filtered sequences:", len(filtered_index_sequences))


# calculate adjacency matrix and degrees of articles

# load links
print("Loading links ...")
inlinks = defaultdict(list)
outlinks = defaultdict(list)
adjacency_matrix = dok_matrix((len(article_map), len(article_map)))
with open("data/wikispeedia/wikispeedia_paths-and-graph/links.tsv") as file:
    for line in file:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        link = [article_map[urllib.parse.unquote(x.strip())] for x in line.split("\t")]
        outlinks[link[0]].append(link[1])
        inlinks[link[1]].append(link[0])
        adjacency_matrix[link[0], link[1]] = 1

print("Calculating degrees ...")
degree = dict([(i, len(indeg) + len(outlinks[i])) for i, indeg in inlinks.items()])


# calculate cosine similarity
print("Calculating cosine similarity ...")

article_text = np.empty((len(article_map),), dtype=object)

# add the article text to the array
for article, index in article_map.items():
    article_text[index] = open("data/wikispeedia/plaintext_articles/" + urllib.parse.quote(article) + ".txt").read()

# build tf-idf features
vect = TfidfVectorizer(max_df=0.8, sublinear_tf=True)
X = vect.fit_transform(article_text)

# all-pairs cosine similarity
cos_similarity = X * X.T


# hypotheses
print("Deriving hypotheses ...")

# cosine hypothesis
hyp_cos = cos_similarity.multiply(adjacency_matrix)
hyp_cos_norm = csr_matrix(normalize(hyp_cos, norm='l1', axis=1))

# degree hypothesis
degree_matrix = np.empty(len(article_map))
for k, v in degree.items():
    degree_matrix[k] = v
hyp_deg = adjacency_matrix.multiply(degree_matrix)
hyp_deg_norm = csr_matrix(normalize(hyp_deg, norm='l1', axis=1))

print("Saving hypotheses ...")
with open("tmp/wikispeedia-hyp_cos.p", "wb") as f:
    pickle.dump(hyp_cos_norm, f)
with open("tmp/wikispeedia-hyp_deg.p", "wb") as f:
    pickle.dump(hyp_deg_norm, f)


# deriving transitions
print("Deriving transitions ...")

n_transitions = sum([len(s) - 1 for s in filtered_index_sequences])
transitions = np.zeros((n_transitions, 2), dtype="int16")

index = 0
for seq in filtered_index_sequences:
    transitions_in_sequence = list(zip(seq, seq[1:]))
    for t in transitions_in_sequence:
        transitions[index, ] = t
        index += 1
print("Number of transitions:", n_transitions)

print("Saving transitions and group assignment probabilities ...")
with open("tmp/wikispeedia-transitions.p", "wb") as f:
    pickle.dump(transitions, f)


# deriving different group assignment probabilities
p_gt = np.empty((n_transitions, 2))

# deriving group assignment probabilities (strict - 1)
index = 0
for seq in filtered_index_sequences:

    transitions_in_sequence = list(zip(seq, seq[1:]))

    p_gt[index, ] = [1, 0]
    index += 1

    for t in transitions_in_sequence[1:]:
        p_gt[index, ] = [0, 1]
        index += 1

with open("tmp/wikispeedia-p_gt-strict-at1.p", "wb") as f:
    pickle.dump(p_gt, f)

# deriving group assignment probabilities (strict - 2)
index = 0
for seq in filtered_index_sequences:

    transitions_in_sequence = list(zip(seq, seq[1:]))

    p_gt[index, ] = [1, 0]
    index += 1

    p_gt[index, ] = [1, 0]
    index += 1

    for t in transitions_in_sequence[2:]:
        p_gt[index, ] = [0, 1]
        index += 1

with open("tmp/wikispeedia-p_gt-strict-at2.p", "wb") as f:
    pickle.dump(p_gt, f)


# deriving group assignment probabilities (strict - 3)
index = 0
for seq in filtered_index_sequences:

    transitions_in_sequence = list(zip(seq, seq[1:]))

    p_gt[index, ] = [1, 0]
    index += 1

    p_gt[index, ] = [1, 0]
    index += 1

    p_gt[index, ] = [1, 0]
    index += 1

    for t in transitions_in_sequence[3:]:
        p_gt[index, ] = [0, 1]
        index += 1

with open("tmp/wikispeedia-p_gt-strict-at3.p", "wb") as f:
    pickle.dump(p_gt, f)


