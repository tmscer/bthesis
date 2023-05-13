import numpy as np
import scipy as scipy


def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


# `h` is called the shrinkage parameter
def cosine_similarity(v1, v2, h=0.0):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + h)


def cosine_distance(v1, v2, h=0.0):
    return 1 - cosine_similarity(v1, v2, h)


def cosine_similarity_matrix(vectors, targets, h=0.0):
    # Major speedups
    sparse_vectors = scipy.sparse.csr_matrix(vectors)
    sparse_targets = scipy.sparse.csr_matrix(targets)
    dot = sparse_vectors.dot(sparse_targets.T).toarray()

    vecs_norm = np.linalg.norm(vectors, axis=1)
    target_norm = np.linalg.norm(targets, axis=1)
    norm = vecs_norm.reshape(-1, 1) * target_norm.reshape(1, -1)

    return dot / (norm + h)


def cosine_distance_matrix(vectors, targets, h=0.0):
    return 1 - cosine_similarity_matrix(vectors, targets, h)


def _test_cosine_distance_matrix():
    test_matrix = np.array([[1, 2], [4, 5], [0, 1]])
    test_target = np.array([[7, 8], [-1, 2]])

    matrix = cosine_distance_matrix(test_matrix, test_target, 0.1)
    print(matrix)

    for i, a in enumerate(test_matrix):
        for j, b in enumerate(test_target):
            baseline = cosine_distance(a, b, 0.1)

            assert matrix[i, j] == baseline, f"{matrix[i, j]} != {baseline}"


_test_cosine_distance_matrix()
