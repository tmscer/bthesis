from sentence_transformers import SentenceTransformer
from cache import cache_np


encoder = SentenceTransformer("all-mpnet-base-v2")


# The only way to speed this up is to use a GPU or cache the results
def encode(text, cache_file=None):
    return cache_np(lambda: encoder.encode(text), cache_file)
