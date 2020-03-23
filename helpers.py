import numpy as np


def set_seed(seed):
    if seed:
        np.random.seed(seed)


def draw(p):
    return np.random.uniform() < p
