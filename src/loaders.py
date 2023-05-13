import numpy as np
import pandas as pd
from os import path
from cache import cache_np
from embeddings import encode


def load_user_item_matrix(filename, num_items):
    with open("../datasets/citeulike-a/users.dat", mode="r", encoding="utf-8") as f:
        user_data = f.readlines()
        user_item_matrix = np.zeros((len(user_data), num_items), dtype=np.int32)

        for i, line in enumerate(user_data):
            items_of_user = list(map(int, line.split(" ")))

            user_item_matrix[i, items_of_user] = 1

    return user_item_matrix


class CiteULikeFiles:
    def __init__(self, root):
        self.root = root

    def documents(self):
        return path.join(self.root, "raw-data.csv")

    def user_item_matrix(self):
        return path.join(self.root, "users.dat")

    def tags(self):
        return path.join(self.root, "tags.dat")

    def item_tag_matrix(self):
        return path.join(self.root, "item-tag.dat")

    def vocabulary(self):
        return path.join(self.root, "vocabulary.dat")

    def bag_of_words(self):
        return path.join(self.root, "mult.dat")


class Loader:
    def __init__(self, root):
        self.files = CiteULikeFiles(root)

        self.docs = None
        self.user_item_matrix = None

    def num_items(self):
        return len(self.load_docs())

    def num_users(self):
        return self.load_user_item_matrix().shape[0]

    def load_all(self):
        self.load_docs()
        self.load_user_item_matrix()

    def load_docs(self):
        if self.docs is None:
            docs = pd.read_csv(self.files.documents())
            del docs["title"]

            self.docs = docs

        return self.docs

    def load_user_item_matrix(self):
        if self.user_item_matrix is None:
            num_items = len(self.load_docs())
            user_item_matrix = load_user_item_matrix(
                self.files.user_item_matrix(), num_items
            )
            self.user_item_matrix = user_item_matrix

        return self.user_item_matrix
