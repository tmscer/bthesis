import numpy as np


def cache_np(computation, cache_file=None):
    if cache_file is None:
        return computation()

    try:
        output = np.load(cache_file)["arr_0"]
    except FileNotFoundError:
        output = computation()
        np.savez_compressed(cache_file, output)

    return output
