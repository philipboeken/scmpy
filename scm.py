from itertools import combinations
from graph import Graph


class f:
    def __init__(self, codomain, domain, mapping):
        self.codomain = codomain
        self.domain = domain
        self.mapping = mapping


class SCM:
    def __init__(self):
        self.I = set()
        self.K = set()
        self.J = set()
        self.H = Graph()
        self.F = set()

    def add_system_variable(self):
        node = self.H.add_node_of_type('system')
        self.I.add(node)
        return node

    def add_context_variable(self):
        node = self.H.add_node_of_type('context')
        self.K.add(node)
        return node

    def add_exogenous_variable(self):
        node = self.H.add_node_of_type('exogenous')
        self.J.add(node)
        return node

    def connect_context_variable(self, cnode, snode=None):
        snode = snode if snode else self.system_variable_of_lowest_order()
        self.H.add_directed_edge(cnode, snode)

    def system_variable_of_lowest_order(self):
        return sorted(self.I, key=lambda x: x.order())[0]

    def saveGraph(self, outdir):
        self.H.saveDotFile(outdir)
