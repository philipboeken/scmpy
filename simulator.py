import math as m
import numpy as np
from scm import SCM
from helpers import *


class SCMSimulator:
    def __init__(self, p, q, eps, eta, N, acyclic, surgical, seed, dep):
        self.p = p
        self.q = q
        self.eps = eps
        self.eta = eta
        self.N = N
        self.acyclic = acyclic
        self.surgical = surgical
        self.dep = dep
        self.scm = SCM()

        np.random.seed(seed)

    def linear_function(self, x):
        coef = np.random.uniform(0.5, 1.5)
        coef *= np.random.choice([-1, 1])
        return (lambda x: coef*x)

    def semilinear_function(self, domain):
        return lambda *args: sum([self.nonlinear_function for x in domain].map(*args))

    def nonlinear_function(self, x):
        a = np.random.uniform(0, 2)
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
            domain = []
            sys_vars = node.sys_vars()
            domain += list(sys_vars)

            conf_vars = node.confounders()
            context_vars = node.context_vars()
            ex_map = node.exogenous_var()
            self.scm.add_mapping(node, domain, map)

            

    def simulate(self):
        self.init_graph()
        self.add_bidirected_edges()
        self.add_directed_edges()
        self.add_mappings()

        self.noise = np.random.normal(size=(1, self.p))
        self.context = np.random.normal(size=(1, self.q))

    def saveTo(self, outdir):
        self.scm.saveGraph(outdir)
