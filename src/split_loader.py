from os import path
from scipy import sparse


class Files:
    def __init__(self, urm_root, icm_root):
        self.urm_root = urm_root
        self.icm_root = icm_root

    def user_rating_matrix(self, split="train"):
        return path.join(self.urm_root, f"URM_{split}.npz")

    def item_content_matrix_tokens_bool(self):
        return path.join(self.icm_root, "ICM_tokens_bool.npz")

    def item_content_matrix_tokens_tf_idf(self):
        return path.join(self.icm_root, "ICM_tokens_TFIDF.npz")


class Loader:
    def __init__(self, urm_root, icm_root):
        self.files = Files(urm_root, icm_root)

        self.urm = {
            "train": None,
            "validation": None,
            "test": None,
        }
        self.icm_bool = None
        self.icm_tf_idf = None

    # sparse.save_npz("yourmatrix.npz", your_matrix)
    # your_matrix_back = sparse.load_npz("yourmatrix.npz")
    def load_user_rating_matrix(self, split="train"):
        if split not in self.urm:
            raise ValueError(f"Unknown split {split}")

        if self.urm[split] is None:
            self.urm[split] = sparse.load_npz(self.files.user_rating_matrix(split))

        return self.urm[split]

    def load_item_content_matrix_tokens_bool(self):
        if self.icm_bool is None:
            self.icm_bool = sparse.load_npz(
                self.files.item_content_matrix_tokens_bool()
            )

        return self.icm_bool

    def load_item_content_matrix_tokens_tf_idf(self):
        if self.icm_tf_idf is None:
            self.icm_tf_idf = sparse.load_npz(
                self.files.item_content_matrix_tokens_tf_idf()
            )

        return self.icm_tf_idf
