import math as m
import numpy as np
from scm import SCM, f
from helpers import *


class SCMSimulator:
    def __init__(self, p, q, eps, eta, N, acyclic, surgical, seed):
        self.p = p
        self.q = q
        self.eps = eps
        self.eta = eta
        self.N = N
        self.acyclic = acyclic
        self.surgical = surgical
        self.scm = SCM()

        np.random.seed(seed)

    def init_graph(self):
        for _ in range(self.p):
            snode = self.scm.add_system_variable()
            enode = self.scm.add_exogenous_variable()
            self.scm.H.add_directed_edge(enode, snode)
        for _ in range(self.q):
            self.scm.add_context_variable()
        self.connect_context_variables()

    def connect_context_variables(self):
        for context_var in self.scm.K:
            self.scm.connect_context_variable(context_var)

    def add_bidirected_edges(self):
        for edge in self.scm.H.open_edges():
            if draw(self.eps):
                self.scm.H.add_bidirected_edge(*edge)

    def add_directed_edges(self):
        for edge in self.scm.H.open_edges():
            if draw(self.eta):
                self.scm.H.add_directed_edge(*edge, random=True)

    def add_mappings(self):
        for node in self.scm.I:
            self.scm.add_map(f(
                node,
                node.parents('system'),
                semilinear_f,
                node.parents('confounder'),
                linear_f,
                node.parents('context'),
                id_f,
                node.parents('exogenous'),
                id_f
            ))

    def simulate(self):
        self.init_graph()
        self.add_bidirected_edges()
        self.add_directed_edges()
        self.add_mappings()

        self.noise = np.random.normal(size=(1, self.p))
        self.context = np.random.normal(size=(1, self.q))

    def saveTo(self, outdir):
        self.scm.saveGraph(outdir)
