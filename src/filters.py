import numpy as np
from metrics import cosine_distance


def items_of_user(docs, user_item_matrix, user_id):
    return docs.iloc[user_item_matrix[user_id, :] == 1]


def find_closest_vectors(
    vectors, target_vector, k=3, metric=cosine_distance, **metric_kwargs
):
    distances = np.array(
        [metric(vec, target_vector, **metric_kwargs) for vec in vectors]
    )
    indices = np.argpartition(distances, k)
    k_indices = indices[:k]

    return distances[k_indices], k_indices
