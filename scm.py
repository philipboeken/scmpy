from itertools import combinations
from graph import Graph
from helpers import *
import numpy as np
import numpy.random as rd
import math as m


class linear_f:
    def __init__(self, domain, coef=[]):
        self.domain = domain
        if coef:
            self.coef = coef
        else:
            self.coef = rd.uniform(0.5, 1.5, len(domain) + 1).tolist()
            self.coef = list(map(lambda x: x * rd.choice([-1, 1]), self.coef))

    def __call__(self, x):
        x = list(x)
        return self.coef[0] + sum(x[i]*self.coef[i+1] for i in len(x))

    def __str__(self):
        if not self.domain:
            return ''
        ran = range(len(self.domain))
        vals = [f'{self.coef[i+1]:.1f} * {self.domain[i]}' for i in ran]
        if self.coef[0]:
            return f'{self.coef[0]:.1f} + ' + ' + '.join(vals)
        return ' + '.join(vals)


class nonlinear_f:
    def __init__(self, domain, a=[], semilinear=True):
        self.domain = domain
        self.semilinear = semilinear
        if a:
            self.a = a
        else:
            self.a = rd.uniform(0.5, 1.5, len(domain)).tolist()

    def __call__(self, x):
        x = list(x)
        if self.semilinear:
            return sum([self.f(self.a[i], x[i]) for i in len(x)])
        return np.prod([self.f(self.a[i], x[i]) for i in len(x)])

    def __str__(self):
        if not self.domain:
            return ''
        ran = range(len(self.domain))
        vals = [
            f'e^(-{self.domain[i]}^2) * sin({self.a[i]:.1f}*{self.domain[i]})' for i in ran]
        if self.semilinear:
            return ' + '.join(vals)
        return ' * '.join(vals)

    def f(self, a, x):
        return m.exp(-x**2) * m.sin(a * x)


class f:
    def __init__(
        self,
        codomain,
        system_map,
        confs_map,
        context_map,
        exo_map
    ):
        self.codomain = codomain
        self.system_map = system_map
        self.confs_map = confs_map
        self.context_map = context_map
        self.exo_map = exo_map

    def __str__(self):
        maps = [
            self.context_map,
            self.system_map,
            self.exo_map,
            self.confs_map
        ]
        s = ' + '.join([str(m) for m in maps if str(m)])
        return f'{self.codomain} = {s}'

    def evaluate(self, system, confounders, context, exogenous):
        return (self.system_map(system)
                + self.confs_map(confounders)
                + self.context_map(context)
                + self.exo_map(exogenous))


class SCM:
    def __init__(self):
        self.I = set()
        self.K = set()
        self.J = set()
        self.H = Graph()
        self.F = set()

    def add_system_variable(self):
        node = self.H.add_node(type='system')
        self.I.add(node)
        return node

    def add_context_variable(self):
        node = self.H.add_node(type='context')
        self.K.add(node)
        return node

    def add_exogenous_variable(self):
        node = self.H.add_node(type='exogenous')
        self.J.add(node)
        return node

    def add_map(self, f):
        self.F.add(f)

    def connect_context_variable(self, cnode, snode=None):
        snode = snode if snode else self.system_variable_of_lowest_order()
        self.H.add_directed_edge(cnode, snode)

    def system_variable_of_lowest_order(self):
        return sorted(self.I, key=lambda x: x.order())[0]

    def mapsOutput(self):
        s = ''
        for f in sorted(self.F, key=lambda f: f.codomain.name):
            s += f'{str(f)}\n'
        return s
