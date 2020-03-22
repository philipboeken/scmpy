import numpy as np
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def draw(p):
    return np.random.uniform() < p
