from itertools import combinations
from graph import Graph


class f:
    def __init__(
        self,
        codomain,
        system_domain,
        system_map,
        confounders_domain,
        confounders_map,
        context_domain,
        context_map,
        exogenous_domain,
        exogenous_map
    ):
        self.codomain = codomain
        self.system_domain = set(system_domain)
        self.system_map = system_map
        self.confounders_domain = set(confounders_domain)
        self.confounders_map = confounders_map
        self.context_domain = set(context_domain)
        self.context_map = context_map
        self.exogenous_domain = set(exogenous_domain)
        self.exogenous_map = exogenous_map

    def evaluate(self, system, confounders, context, exogenous):
        return (self.system_map(*system)
                + self.confounders_map(*confounders)
                + self.context_map(*context)
                + self.exogenous_map(*exogenous))


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

    def add_map(self, f):
        self.F.add(f)

    def connect_context_variable(self, cnode, snode=None):
        snode = snode if snode else self.system_variable_of_lowest_order()
        self.H.add_directed_edge(cnode, snode)

    def system_variable_of_lowest_order(self):
        return sorted(self.I, key=lambda x: x.order())[0]

    def saveGraph(self, outdir):
        self.H.saveDotFile(outdir)
