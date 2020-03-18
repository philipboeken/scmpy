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
        data = sorted(self.data, key=lambda d: d[0].name)
        return hash(tuple(map(lambda d: d[0].name, data)))

    def nodes(self, type=None, name=None):
        for c in self.data:
            if type and not c[0].type == type:
                continue
            if name and not c[0].name == name:
                continue
            yield c[0]

    def is_between(self, node1, node2):
        return self.nodes() == {node1, node2}

    def has_node_of_type(self, type):
        return len(list(self.nodes(type=type))) > 0

    def dot_output(self):
        c1, c2 = sorted(self.data, key=lambda x: x[1])
        if c1[1] == c2[1]:
            return f'{c1[0]}->{c2[0]}[dir="both", arrowhead="{c1[1]}"];\n'
        return f'{c1[0]}->{c2[0]}[arrowtail="{c1[1]}", arrowhead="{c2[1]}"];\n'


class Node:
    names = {'system': 'S', 'context': 'C',
             'confounder': 'A', 'exogenous': 'E'}

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

    def has_type(self, type):
        return self.type == type

    def is_parent_of(self, node):
        options = {
            Edge(self, node, 'arrow', 'arrow'),
            Edge(self, node, 'open', 'arrow'),
            Edge(self, node, 'none', 'arrow')
        }
        return len(options.intersection(self.edges)) > 0

    def adjacent_nodes(self, type=None):
        nodes = set()
        for edge in self.edges:
            nodes = nodes.union({node for node in edge.nodes()})
        nodes = nodes.difference({self})
        if type:
            nodes = filter(lambda node: node.has_type(type), nodes)
        return nodes

    def parents(self, type=None):
        return filter(
            lambda node: node.is_parent_of(self),
            self.adjacent_nodes(type)
        )

    def children(self, type=None):
        return filter(
            lambda node: self.is_parent_of(node),
            self.adjacent_nodes(type)
        )

    def order(self):
        return len(self.edges)

    def dot_output(self):
        if self.has_type('system'):
            return f'{self.name}[label="{self.name}"];\n'
        elif self.has_type('context'):
            return f'{self.name}[label="{self.name}", shape=rectangle];\n'
        else:
            return '{}->{}[dir="both"];\n'.format(
                *{node.name for node in self.adjacent_nodes()}
            )


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    def get_nodes(self, type=None, name=None):
        for node in self.nodes:
            if type and not node.type == type:
                continue
            if name and not node.name == name:
                continue
            yield node

    def get_edges(self, augmented=False):
        if augmented:
            return self.edges
        edges = set()
        for edge in self.edges:
            if edge.has_node_of_type('confounder'):
                confounder = next(iter(edge.nodes(type='confounder')))
                edges.add(Edge(*confounder.children(), 'arrow', 'arrow'))
            else:
                edges.add(edge)
        return edges

    def open_edges(self):
        edges = set()
        for node1, node2 in combinations(self.get_nodes(type='system'), 2):
            if not self.has_edge_between(node1, node2):
                edges.add((node1, node2))
        return edges

    def add_node(self, type, id=None):
        ids = [v.id for v in self.get_nodes(type=type)]
        id = max(ids) + 1 if ids else 0
        node = Node(id, type)
        self.nodes.add(node)
        return node

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
        conf = self.add_node(type='confounder')
        self.add_edge(Edge(conf, node1, 'none', 'arrow'))
        self.add_edge(Edge(conf, node2, 'none', 'arrow'))

    def has_edge_between(self, node1, node2):
        for edge in self.edges:
            if edge.is_between(node1, node2):
                return True
        return False

    def dotOutput(self, augmented=False):
        out = 'digraph G {\n'
        for node in sorted(self.get_nodes(type='context'), key=id):
            out += node.dot_output()
        for node in sorted(self.get_nodes(type='system'), key=id):
            out += node.dot_output()
        for edge in self.get_edges(augmented):
            out += edge.dot_output()
        out += '}\n'
        return out
