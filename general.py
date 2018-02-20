import numpy as np


def read_brown_clusters(path):
    word_dict = {}

    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip().split()
            word_dict[line[1]] = line[0]

    return word_dict


def expand_embedding_matrix(old_matrix, old_idxs, new_idxs, classes, transpose=False):

    if transpose:
        old_matrix = np.transpose(old_matrix)

    if len(old_matrix.shape) == 2:
        matrix = np.zeros((classes, old_matrix.shape[1]))
    else:
        assert len(old_matrix.shape) == 1
        matrix = np.zeros(classes)

    for cluster, idx in new_idxs.items():
        if cluster in old_idxs:
            matrix[idx] = old_matrix[old_idxs[cluster]]

    if transpose:
        matrix = np.transpose(matrix)

    return matrix


def create_cluster_dict(word2cluster, level):
    assert level > 0

    word2cluster = {word: cluster[:level] for word, cluster in word2cluster.items()}

    clusters = set(word2cluster.values())
    classes = len(clusters) + 1
    cluster2id = {cluster: cid for cid, cluster in enumerate(clusters, 1)}

    word2id = {word: cluster2id[cluster] for word, cluster in word2cluster.items()}

    return word2id, classes
