from itertools import combinations
import numpy.random as rd
from graph import Graph
import numpy as np
import math as m


class id_f:
    def __init__(self, domain):
        self.domain = domain

    def __call__(self, x):
        return 0.6 * sum(list(x))

    def __str__(self):
        if not self.domain:
            return ''
        vals = [f'{self.domain[i]}' for i in range(len(self.domain))]
        return ' + '.join(vals)


class linear_f:
    def __init__(self, domain, coef=[], bias=True):
        self.domain = domain
        if coef:
            self.coef = coef
        else:
            self.coef = [rd.uniform(0.5, 1.5)] if domain and bias else [0]
            self.coef += rd.uniform(0.5, 1.5, len(domain)).tolist()
            self.coef = list(map(lambda x: x * rd.choice([-1, 1]), self.coef))

    def __call__(self, x):
        x = [1] + list(x)
        return sum(x[i]*self.coef[i] for i in range(len(x)))

    def __str__(self):
        if not self.domain:
            return ''
        ran = range(len(self.domain))
        vals = [f'{self.coef[i+1]:.1f} * {self.domain[i]}' for i in ran]
        s = f'{self.coef[0]:.1f} + ' + ' + '.join(vals).replace('1.0 * ', '')
        return s.replace('0.0 + ', '')


class nonlinear_f:
    def __init__(self, domain, a=[], additive=True):
        self.domain = domain
        self.additive = additive
        if a:
            self.a = a
        else:
            self.a = rd.uniform(0.5, 1.5, len(domain)).tolist()

    def __call__(self, x):
        x = list(x)
        if self.additive:
            return sum([self.__f(self.a[i], x[i]) for i in range(len(x))])
        return np.prod([self.__f(self.a[i], x[i]) for i in range(len(x))])

    def __str__(self):
        if not self.domain:
            return ''
        ran = range(len(self.domain))
        vals = [
            f'e^(-{self.domain[i]}^2) * sin({self.a[i]:.1f}*{self.domain[i]})' for i in ran]
        if self.additive:
            return ' + '.join(vals)
        return ' * '.join(vals)

    def __f(self, a, x):
        # return m.exp(-x**2) * m.sin(a * x)
        return m.sin(8*x)


class f:
    def __init__(
        self,
        codomain,
        system_map,
        latconf_map,
        context_map,
        exo_map,
        surgical
    ):
        self.codomain = codomain
        self.system_map = system_map
        self.latconf_map = latconf_map
        self.context_map = context_map
        self.exo_map = exo_map
        self.surgical = surgical

    def __str__(self):
        maps = [
            self.context_map,
            self.system_map,
            self.exo_map,
            self.latconf_map
        ]
        s = ' + '.join([str(m) for m in maps if str(m)])
        return f'{self.codomain} = {s}'

    def domain(self):
        return set(self.system_map.domain).union(
            set(self.latconf_map.domain),
            set(self.context_map.domain),
            set(self.exo_map.domain)
        )

    def evaluate(self, system, latconfs, context, exogenous):
        if self.surgical:
            if self.context_map(context):
                return self.context_map(context)
            else:
                return (self.system_map(system)
                        + self.latconf_map(latconfs)
                        + self.exo_map(exogenous))
        return (self.system_map(system)
                + self.latconf_map(latconfs)
                + 3 * self.context_map(context)
                + self.exo_map(exogenous))

    def depends_on(self, fs):
        nodes = set(map(lambda f: f.codomain, fs))
        return self.domain() & nodes != set()


class SCM:
    options = {
        'system': {
            'name': 'X',
            'shape': 'oval',
            'augmented': False
        },
        'context': {
            'name': 'C',
            'shape': 'box',
            'augmented': False
        },
        'exogenous': {
            'name': 'E',
            'shape': 'circle',
            'augmented': True
        },
        'latconf': {
            'name': 'L',
            'shape': 'circle',
            'augmented': True
        }
    }

    def __init__(self):
        self.system = set()  # system
        self.context = set()  # context
        self.exogenous = set()  # exogenous (noise)
        self.latconf = set()  # latent confounders
        self.H = Graph()  # graph
        self.F = set()  # mappings

    def get_options(self, type, option=None):
        if option:
            return self.options[type][option]
        return self.options[type]

    def nodes(self, type, sort=False):
        nodes = getattr(self, type)
        if sort:
            return sorted(nodes, key=lambda node: node.name)
        return nodes

    def is_acyclic(self):
        return self.H.is_acyclic('system')

    def add_node(self, type):
        node = self.H.add_node(
            type=type,
            **self.get_options(type)
        )
        self.nodes(type).add(node)
        return node

    def add_map(self, f):
        self.F.add(f)

    def connect_context_node(self, context, system=None):
        system = system if system else self.system_node_of_lowest_order()
        self.H.add_directed_edge(context, system)

    def system_node_of_lowest_order(self):
        return sorted(self.system, key=lambda x: (x.order(), x.name))[0]

    def maps_output(self):
        maps = sorted(self.F, key=lambda f: f.codomain.name)
        maps = [str(map) for map in maps]
        return '\n'.join(maps)

    def save_to(self, outdir):
        f = open(f'{outdir}/sim-graph.dot', 'w+')
        f.write(self.H.dot_output(augmented=False))
        f.close()

        f = open(f'{outdir}/sim-graph-augmented.dot', 'w+')
        f.write(self.H.dot_output(augmented=True))
        f.close()

        f = open(f'{outdir}/sim-maps.txt', 'w+')
        f.write(self.maps_output())
        f.close()

        self.H.adjacency_matrix('system', sort=True).to_csv(f'{outdir}/sim-edge.csv', sep='\t')

        self.H.ancestral_matrix('system').to_csv(f'{outdir}/sim-arel.csv', sep='\t')
