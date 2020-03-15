import math as m
import numpy as np
from graph import Graph
from helpers import *


class Simulator:
    def __init__(self, p, q, eps, eta, N, acyclic, surgical, seed, dep):
        self.p = p
        self.q = q
        self.eps = eps
        self.eta = eta
        self.N = N
        self.acyclic = acyclic
        self.surgical = surgical
        self.dep = dep
        self.graph = Graph()

        np.random.seed(seed)

    def linear_function(self):
        coef = np.random.uniform(0.5, 1.5)
        coef *= np.random.choice([-1, 1])
        return (lambda x: coef*x)

    def nonlinear_function(self):
        a = np.random.uniform(0, 4)
        return (lambda x: m.exp(-x**2/2) * m.sin(a * x))

    def get_dependence(self):
        dep = np.random.choice(
            ['linear', 'gaussian']
        ) if self.dep == 'mixed' else self.dep
        if dep == 'linear':
            return self.linear_function()
        return self.nonlinear_function()

    def init_graph(self):
        for _ in range(self.p):
            self.graph.add_system_variable()
        for _ in range(self.q):
            self.graph.add_context_variable()
        self.connect_context_variables()

    def connect_context_variables(self):
        for context_var in self.graph.context_variables():
            system_var = self.graph.system_variable_of_lowest_order()
            self.graph.add_directed_edge(context_var, system_var)

    def add_bidirected_edges(self):
        for edge in self.graph.open_edges():
            if draw(self.eps):
                self.graph.add_bidirected_edge(*edge)

    def add_directed_edges(self):
        for edge in self.graph.open_edges():
            if draw(self.eta):
                self.graph.add_directed_edge(*edge, random=True)

    def simulate(self):
        self.init_graph()
        self.add_bidirected_edges()
        self.add_directed_edges()

        self.noise = np.random.normal(size=(1, self.p))
        self.context = np.random.normal(size=(1, self.q))

    def saveTo(self, outdir):
        self.graph.saveDotFile(outdir)
