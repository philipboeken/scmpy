from scipy.linalg import expm
from pandas import DataFrame
from helpers import draw
from numpy import diag


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

    def augmented(self):
        return any(map(lambda node: node.augmented, self.nodes()))

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
    def __init__(self, id, type, name, shape, augmented):
        self.id = id
        self.type = type
        self.name = name + str(id)
        self.edges = set()
        self.shape = shape
        self.augmented = augmented

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.name == other.name

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
        return len(options & self.edges) > 0

    def adjacent_nodes(self, type=None):
        nodes = set()
        for edge in self.edges:
            nodes = nodes | {node for node in edge.nodes()}
        nodes = nodes - {self}
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

    def dot_output(self, augmented=False):
        if not augmented and self.augmented:
            if len(list(self.children())) == 2:
                return '{}->{}[dir="both"];\n'.format(
                    *{node.name for node in self.adjacent_nodes()}
                )
            return ''
        return f'{self.name}[label="{self.name}", shape={self.shape}];\n'


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()

    def get_nodes(self, type=None, name=None, sort=False):
        nodes = self.nodes
        if type:
            nodes = [node for node in nodes if node.type == type]
        if name:
            nodes = [node for node in nodes if node.name == name]
        if sort:
            return sorted(nodes, key=lambda node: node.name)
        return nodes

    def get_edges(self, augmented=False):
        if augmented:
            return self.edges
        edges = set()
        for edge in self.edges:
            if not edge.augmented():
                edges.add(edge)
        return edges

    def adjacency_matrix(self, type):
        matrix = DataFrame(
            0,
            index=self.get_nodes(type),
            columns=self.get_nodes(type)
        )
        for node in self.get_nodes(type):
            for child in node.children(type=type):
                matrix[node][child] = 1
            for parent in node.parents(type=type):
                matrix[parent][node] = 1
        return matrix

    def is_acyclic(self, type):
        exp = expm(self.adjacency_matrix(type=type).to_numpy())
        return abs(sum(diag(exp))) - len(self.get_nodes(type=type)) < 1e-10

    def add_node(self, type, name, shape, augmented=False, id=None):
        ids = [v.id for v in self.get_nodes(type=type)]
        id = max(ids) + 1 if ids else 0
        node = Node(id, type, name, shape, augmented)
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

    def has_edge_between(self, node1, node2):
        for edge in self.edges:
            if edge.is_between(node1, node2):
                return True
        return False

    def dot_output(self, augmented=False):
        out = 'digraph G {\n'
        for node in self.get_nodes(sort=True):
            out += node.dot_output(augmented)
        for edge in self.get_edges(augmented):
            out += edge.dot_output()
        out += '}\n'
        return out
