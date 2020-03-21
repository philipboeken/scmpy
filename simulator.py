from itertools import combinations
from scipy.stats import bernoulli
from pandas import DataFrame
import numpy.random as rd
from helpers import *
import numpy as np
from scm import *
import math as m
import random


class SCMGenerator:
    def __init__(self, p, q, eps, eta, acyclic, surgical, seed):
        self.p = p
        self.q = q
        self.eps = eps
        self.eta = eta
        self.acyclic = acyclic
        self.surgical = surgical
        np.random.seed(seed)
        self.scm = SCM()

    def init_graph(self):
        for _ in range(self.p):
            system = self.scm.add_node('system')
            exo = self.scm.add_node('exogenous')
            self.scm.H.add_directed_edge(exo, system)
        for _ in range(self.q):
            self.scm.add_node('context')
        self.connect_context_nodes()

    def connect_context_nodes(self):
        for context_var in self.scm.context:
            self.scm.connect_context_node(context_var)

    def add_latent_confs(self):
        for node1, node2 in combinations(self.scm.system, 2):
            if draw(self.eps):
                conf = self.scm.add_node('latconf')
                self.scm.H.add_directed_edge(conf, node1)
                self.scm.H.add_directed_edge(conf, node2)

    def add_directed_edges(self):
        for node1, node2 in combinations(self.scm.system, 2):
            if draw(self.eta) and not self.scm.H.has_edge_between(node1, node2):
                self.scm.H.add_directed_edge(node1, node2, random=True)

    def add_mappings(self):
        for node in self.scm.system:
            self.scm.add_map(f(
                codomain=node,
                context_map=linear_f(
                    sorted(node.parents('context'), key=id), [0, 1]),
                system_map=nonlinear_f(
                    sorted(node.parents('system'), key=id)),
                exo_map=linear_f(
                    sorted(node.parents('exogenous'), key=id)),
                latconf_map=linear_f(
                    sorted(node.parents('latconf'), key=id))
            ))

    def generate_scm(self):
        self.init_graph()
        self.add_directed_edges()
        self.add_latent_confs()
        self.add_mappings()
        return self.scm


class RandVar:
    distributions = {
        'normal': rd.normal,
        'bernoulli': bernoulli.rvs
    }

    def __init__(self, node, dist):
        self.node = node
        self.dist = self.distributions[dist]
        self.value = 0

    def draw(self):
        self.value = self.dist()


class SCMSimulator:
    def __init__(self, scm):
        self.scm = scm
        self.system = set(RandVar(node, 'normal') for node in scm.system)
        self.context = set(RandVar(node, 'bernoulli') for node in scm.context)
        self.exogenous = set(RandVar(node, 'normal') for node in scm.exogenous)
        self.latconf = set(RandVar(node, 'normal') for node in scm.latconf)

    def randvar(self, node):
        return next(filter(
            lambda rv: rv.node == node,
            self.randvars(type=node.type)
        ))

    def randvars(self, type=None, sort=False, nodes=None):
        if type:
            randvars = getattr(self, type)
        elif isinstance(nodes, list):
            randvars = [self.randvar(node) for node in nodes]
        else:
            randvars = self.system | self.context | self.exogenous | self.latconf
        if sort:
            return sorted(randvars, key=lambda rv: rv.node.name)
        return randvars

    def names(self):
        return [rv.node.name for rv in self.randvars(sort=True)]

    def state(self):
        return [rv.value for rv in self.randvars(sort=True)]

    def simulate(self, N):
        sample = self.sample(N)
        self.data = DataFrame(sample)
        for context in self.context:
            context.value = 1
            sample = self.sample(N)
            self.data = self.data.append(sample, ignore_index=True)
            context.value = 0
        self.data.columns = self.names()

    def sample(self, N):
        sample = []
        for _ in range(N):
            self.randomize_state(target='exogenous')
            self.randomize_state(target='latconf')
            if self.scm.is_acyclic():
                SCMSolver.iterate(self)
            else:
                self.randomize_state(target='system')
                self.scm.solve(self)
            sample += [self.state()]
        return sample

    def randomize_state(self, target):
        for randvar in self.randvars(target):
            randvar.draw()

    def save_to(self, outdir):
        self.data.to_csv(f'{outdir}/sim-samples.csv')


class SCMSolver:
    @staticmethod
    def iterate(scm):
        updates = scm.scm.F.copy()
        while len(updates) > 0:
            f = random.sample(updates, 1)[0]
            if not f.depends_on(updates):
                SCMSolver.apply(scm, f)
                updates.discard(f)

    @staticmethod
    def apply(scm, f):
        target = next(filter(
            lambda rv: rv.node == f.codomain,
            scm.randvars()
        ))
        system = [rv.value for rv in scm.randvars(nodes=f.system_map.domain)]
        latconfs = [rv.value for rv in scm.randvars(
            nodes=f.latconf_map.domain)]
        context = [rv.value for rv in scm.randvars(
            nodes=f.context_map.domain)]
        exogenous = [rv.value for rv in scm.randvars(nodes=f.exo_map.domain)]
        target.value = f.evaluate(system, latconfs, context, exogenous)

    @staticmethod
    def solve(scm):
        print('TODO')
