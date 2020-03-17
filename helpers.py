import math as m
import numpy as np
import scipy.stats as stats
import sys
import os


def draw(p):
    return np.random.uniform() < p


def id_f(x):
    return x


def linear_f(x):
    coef = np.random.uniform(0.5, 1.5)
    coef *= np.random.choice([-1, 1])
    c = np.random.uniform(-2, 2)
    return coef*x + c


def multilinear_f(self, *args):
    return sum(linear_f(x) for x in args)


def semilinear_f(*args):
    return multilinear_f(nonlinear_f(x) for x in args)


def nonlinear_f(x):
    a = np.random.uniform(0, 2)
    return m.exp(-x**2/2) * m.sin(a * x)
