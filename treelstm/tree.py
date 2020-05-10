from enum import Enum

class TreeType(Enum):
    DEPENDENCY = 1
    CONSTITUENCY = 2

# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def __repr__(self):
        assert self.num_children == len(self.children)
        if self.children:
            return '{0}: {1}'.format(self.idx, str(self.children))
        else:
            return str(self.idx)

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if hasattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def leaf_size(self):
        if hasattr(self, '_leaf_size'):
            return self._leaf_size
        count = 0
        if self.num_children == 0:
            return 1
        for child in self.children:
            count += child.leaf_size()
        self._leaf_size = count
        return self._leaf_size

    def construct_graph(self, add_vertices, add_edge):
        if add_vertices is not None:
            add_vertices(self.size())
        for child in self.children:
            child.construct_graph(None, add_edge)
            add_edge(self.idx, child.idx)


