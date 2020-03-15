from itertools import combinations
from helpers import *


class Edge:
    heads = {
        'arrow': 'normal',
        'open': 'odot',
        'none': 'none'
    }

    def __init__(self, node1, node2, head1, head2):
        self.data = {
            (node1, self.heads[head1]),
            (node2, self.heads[head2])
        }

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.data == other.data

    def __hash__(self):
        return hash((data for data in self.data))

    def nodes(self):
        return {c[0] for c in self.data}

    def is_between(self, node1, node2):
        return self.nodes() == {node1, node2}

    def dot_output(self):
        c1, c2 = sorted(self.data, key=lambda x: x[1])
        if c1[1] == c2[1]:
            return f'{c1[0]}->{c2[0]}[dir="both", arrowhead="{c1[1]}"];\n'
        return f'{c1[0]}->{c2[0]}[arrowtail="{c1[1]}", arrowhead="{c2[1]}"];\n'


class Node:
    names = {'system': 'S', 'context': 'C', 'confounder': 'A'}
    shape = {'system': 'S', 'context': 'C', 'confounder': 'A'}

    def __init__(self, id, type):
        self.id = id
        self.type = type
        self.name = self.names[type] + str(id)
        self.edges = set()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.id == other.id

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def is_system_variable(self):
        return self.type == 'system'

    def is_context_variable(self):
        return self.type == 'context'

    def is_confounder(self):
        return self.type == 'confounder'

    def adjacent_nodes(self):
        nodes = {}
        for edge in self.edges:
            nodes = nodes.union({node for node in edge.nodes()})
        return nodes.difference({self})

    def order(self):
        return len(self.edges)

    def dot_output(self):
        if self.is_system_variable():
            return f'{self.name}[label="{self.name}"];\n'
        elif self.is_context_variable():
            return f'{self.name}[label="{self.name}", shape=rectangle];\n'
        else:
            return '{}->{}[dir="both"];\n'.format(
                *{node.name for node in self.adjacent_nodes()}
            )


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    def system_variables(self):
        return {v for v in self.nodes if v.is_system_variable()}

    def context_variables(self):
        return {v for v in self.nodes if v.is_context_variable()}

    def confounders(self):
        return {v for v in self.nodes if v.is_confounder()}

    def system_variable_of_lowest_order(self):
        return sorted(self.system_variables(), key=lambda x: x.order())[0]

    def open_edges(self):
        edges = set()
        for node1, node2 in combinations(self.system_variables(), 2):
            if not self.has_edge_between(node1, node2):
                edges.add((node1, node2))
        return edges

    def add_node(self, id, type):
        node = Node(id, type)
        self.nodes.add(node)

    def add_edge(self, edge):
        self.edges.add(edge)
        for node in edge.nodes():
            node.edges.add(edge)

    def add_directed_edge(self, node1, node2, random=False):
        if random and draw(0.5):
            edge = Edge(node2, node1, 'none', 'arrow')
        else:
            edge = Edge(node1, node2, 'none', 'arrow')
        self.add_edge(edge)

    def add_bidirected_edge(self, node1, node2):
        edge = Edge(node1, node2, 'arrow', 'arrow')
        self.edges.add(edge)

    def add_system_variable(self):
        vars = self.system_variables()
        id = max([v.id for v in vars]) + 1 if vars else 0
        self.add_node(id, 'system')

    def add_context_variable(self):
        vars = self.context_variables()
        id = max([v.id for v in vars]) + 1 if vars else 0
        self.add_node(id, 'context')

    def add_confounder(self):
        vars = self.confounders()
        id = max([v.id for v in vars]) + 1 if vars else 0
        self.add_node(id, 'confounder')

    def has_edge_between(self, node1, node2):
        for edge in self.edges:
            if edge.is_between(node1, node2):
                return True
        return False

    def saveDotFile(self, outdir):
        f = open('{}/sim-graph.dot'.format(outdir), 'w+')
        f.write('digraph G {\n')
        for node in sorted(self.context_variables(), key=id):
            f.write(node.dot_output())
        for node in sorted(self.system_variables(), key=id):
            f.write(node.dot_output())
        for edge in self.edges:
            f.write(edge.dot_output())
        f.write('}\n')
