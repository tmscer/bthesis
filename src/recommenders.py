import numpy as np
import scipy
from filters import find_closest_vectors
from embeddings import encode
from cache import cache_np
from os import path
from metrics import cosine_distance_matrix, cosine_similarity_matrix


class CacheFiles:
    def __init__(self, root):
        self.root = root

    def title_embeddings_cache(self):
        return path.join(self.root, "title_embeddings.npz")

    def abstract_embeddings_cache(self):
        return path.join(self.root, "abstract_embeddings.npz")

    def collaborative_user_similarities_cache(self):
        return path.join(self.root, "user_similarities.npz")

    def collaborative_item_similarities_cache(self):
        return path.join(self.root, "item_similarities.npz")

    def embedding_item_similarities_cache(self):
        return path.join(self.root, "embedding_item_similarities.npz")


class Recommender:
    def __init__(self, loader):
        self.loader = loader
        self.files = CacheFiles(loader.files.root)

        self.collaborative_user_similarities = None
        self.collaborative_item_similarities = None
        self.embedding_item_similarities = None

        self.title_embeddings = None
        self.abstract_embeddings = None
        self.text_embeddings = None

    def load_title_embeddings(self):
        if self.title_embeddings is None:
            docs = self.loader.load_docs()

            title_embeddings = encode(
                docs["raw.title"], cache_file=self.files.title_embeddings_cache()
            )
            self.title_embeddings = title_embeddings

        return self.title_embeddings

    def load_abstract_embeddings(self):
        if self.abstract_embeddings is None:
            docs = self.loader.load_docs()

            abstract_embeddings = encode(
                docs["raw.abstract"], cache_file=self.files.abstract_embeddings_cache()
            )
            self.abstract_embeddings = abstract_embeddings

        return self.abstract_embeddings

    def load_text_embeddings(self):
        if self.text_embeddings is None:
            title_embeddings = self.load_title_embeddings()
            abstract_embeddings = self.load_abstract_embeddings()

            text_embeddings = np.concatenate(
                [title_embeddings, abstract_embeddings], axis=1
            )
            self.text_embeddings = text_embeddings

        return self.text_embeddings

    def load_collaborative_user_similarities(self):
        if self.collaborative_user_similarities is None:
            user_item_matrix = self.loader.load_user_item_matrix()
            user_similarities = cache_np(
                lambda: cosine_similarity_matrix(user_item_matrix, user_item_matrix),
                self.files.collaborative_user_similarities_cache(),
            )

            self.collaborative_user_similarities = user_similarities

        return self.collaborative_user_similarities

    def load_collaborative_item_similarities(self):
        if self.collaborative_item_similarities is None:
            user_item_matrix = self.loader.load_user_item_matrix()
            item_similarities = cache_np(
                lambda: cosine_similarity_matrix(
                    user_item_matrix.T, user_item_matrix.T
                ),
                self.files.collaborative_item_similarities_cache(),
            )

            self.collaborative_item_similarities = item_similarities

        return self.collaborative_item_similarities

    def load_embedding_item_similarities(self):
        if self.embedding_item_similarities is None:
            text_embeddings = self.load_text_embeddings()
            item_similarities = cache_np(
                lambda: cosine_similarity_matrix(text_embeddings, text_embeddings),
                self.files.embedding_item_similarities_cache(),
            )

            self.embedding_item_similarities = item_similarities

        return self.embedding_item_similarities

    def user_vector(self, user_id):
        return self.loader.load_user_item_matrix()[user_id, :]

    def item_vector(self, item_id):
        return self.loader.load_user_item_matrix()[:, item_id]

    def relevant_items(self, user_id):
        user_vector = self.user_vector(user_id)

        return np.where(user_vector > 0)[0]

    def collaborative_similar_users(self, user_id, k):
        user_item_matrix = self.loader.load_user_item_matrix()
        user_vector = self.user_vector(user_id)

        return find_closest_vectors(user_item_matrix, user_vector, k)

    def collaborative_similar_items(self, item_id, k):
        user_item_matrix = self.loader.load_user_item_matrix()
        item_vector = self.item_vector(item_id)

        return find_closest_vectors(user_item_matrix.T, item_vector, k)

    # When `item_id` is `None`, ratings is predicted for all items
    def collaborative_user_knn(self, user_id, k_users, item_id=None):
        user_similarities = self.load_collaborative_user_similarities()[user_id, :]

        # ignore the closest user which will be the user itself
        similar_users = user_similarities.argsort()[::-1][1: k_users + 1]
        k_user_similarities = user_similarities[similar_users]

        item_scores = np.zeros(self.loader.num_items())
        user_item_matrix = self.loader.load_user_item_matrix()

        for similarity, similar_user in zip(k_user_similarities, similar_users):
            if similar_user == user_id:
                raise Exception("Similar user cannot be the same as the user")

            item_scores += similarity * user_item_matrix[similar_user, :]

        item_scores /= np.sum(k_user_similarities)

        if item_id is None:
            return item_scores

        return item_scores[item_id]

    def collaborative_item_knn(self, user_id, item_id, k_items):
        item_similarities = self.load_collaborative_item_similarities()[item_id, :]

        # ignore the closest item which will be the item itself
        similar_items = item_similarities.argsort()[::-1][1: k_items + 1]
        k_item_similarities = item_similarities[similar_items]

        user_item_matrix = self.loader.load_user_item_matrix()
        score = 0

        for similarity, similar_item in zip(k_item_similarities, similar_items):
            score += similarity * user_item_matrix[user_id, similar_item]

        score /= np.sum(k_item_similarities)

        return score

    def embedding_title_search(self, query, k):
        query_embedding = encode(query)
        title_embeddings = self.load_title_embeddings()

        return find_closest_vectors(title_embeddings, query_embedding, k)

    def embedding_abstract_search(self, query, k):
        query_embedding = encode(query)
        abstract_embeddings = self.load_abstract_embeddings()

        return find_closest_vectors(abstract_embeddings, query_embedding, k)

    def embedding_similar_items(self, item_id, k):
        text_embeddings = self.load_text_embeddings()
        item_vector = text_embeddings[item_id, :]

        return find_closest_vectors(text_embeddings, item_vector, k)

    def embedding_pooled_similar_items(self, user_id, k, pooling="sum"):
        return pooled_vector_search(
            self.load_text_embeddings(),
            self.loader.load_user_item_matrix(),
            user_id,
            k,
            pooling,
        )

    def embedding_weighted_similar_items(self, user_id, k_users, k_items, pooling="mean"):
        user_similarities = self.load_collaborative_user_similarities()[user_id, :]

        # get one more user to account for user with `user_id`
        similar_users = user_similarities.argsort()[::-1][: k_users + 1]

        relevant_items = self.relevant_items(user_id)
        relevant_items_embeddings = self.load_text_embeddings()[relevant_items, :]

        user_item_matrix = self.loader.load_user_item_matrix()
        item_scores = np.zeros(self.loader.num_items())

        pooling = choose_pooling_func(pooling)

        for similar_user in similar_users:
            if similar_user == user_id:
                continue

            candidate_user_items = self.relevant_items(similar_user)
            candidate_user_items_embeddings = self.load_text_embeddings()[
                candidate_user_items, :
            ]

            similarities = cosine_similarity_matrix(
                relevant_items_embeddings, candidate_user_items_embeddings
            )

            weight = pooling(similarities)
            item_scores += weight * user_item_matrix[similar_user, :]

        recommendations = np.argsort(item_scores)[::-1][:k_items]

        return recommendations


def pooled_vector_search(feature_matrix, pooled_matrix, pooled_id, k, pooling):
    pooling = choose_pooling_func(pooling)

    indices = pooled_matrix[pooled_id, :] > 0
    pool_vector = pooling(feature_matrix[indices, :], axis=0)

    return find_closest_vectors(feature_matrix, pool_vector, k)


def choose_pooling_func(pooling):
    if pooling == "sum":
        return np.sum
    elif pooling == "max":
        return np.max
    elif pooling == "mean":
        return np.mean
    elif pooling == "softmax":
        return scipy.special.softmax
    elif callable(pooling):
        return pooling
    else:
        raise ValueError("Unknown pooling " + pooling)
