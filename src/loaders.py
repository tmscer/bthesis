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
        self.tags = None
        self.tag_to_id = None
        self.item_tag_matrix = None

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

    def load_tags(self):
        if self.tags is None:
            with open(self.files.tags()) as f:
                tags = f.readlines()

            tags = list(map(lambda x: x.strip(), tags))
   
            self.tags = np.array(tags)

        return self.tags

    def get_tag_id(self, tag):
        if self.tag_to_id is None:
            tags = self.load_tags()

            self.tag_to_id = {tag: i for i, tag in enumerate(tags)}

        return self.tag_to_id[tag]

    def get_item_tags(self, item_id):
        item_tag_matrix = self.load_item_tag_matrix()
        item_vector = item_tag_matrix[item_id, :]

        return np.where(item_vector == 1)[0]

    def get_item_tag_names(self, item_id):
        item_tags = self.get_item_tags(item_id)
        tags = self.load_tags()

        return tags[item_tags]

    def load_item_tag_matrix(self):
        if self.item_tag_matrix is None:
            with open(self.files.item_tag_matrix()) as f:
                lines = f.readlines()

            tags = self.load_tags()
            number_of_items = len(lines)
            item_tag_matrix = np.zeros((number_of_items, len(tags)), dtype=np.int32)

            for item_id, line in enumerate(lines):
                numbers = list(map(int, line.split(" ")))

                number_of_tags = numbers[0]
                indices = numbers[1:]

                assert number_of_tags == len(indices)

                item_tag_matrix[item_id, indices] = 1

            self.item_tag_matrix = item_tag_matrix

        return self.item_tag_matrix
