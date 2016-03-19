class Node:
    def __init__(self, name):
        self.name = name
        self.left_acc = 0
        self.right_acc = 0
        self.merge_dist = 0
        self.weight = 1

        self.parent = None
        self.left = None
        self.right = None

    def set_left(self, node):
        self.left = node
        node.parent = self

    def set_right(self, node):
        self.right = node
        node.parent = self

    def to_root(self):
        parent = self.parent
        current = self
        while parent != None:
            is_left = parent.has_left(current)
            yield parent, is_left
            current = parent
            parent = parent.parent

    def has_right(self, node):
        if self.right is None:
            return False
        elif self.right is node:
            return True
        else:
            return self.right.has_right(node) or self.right.has_left(node)

    def has_left(self, node):
        if self.left is None:
            return False
        elif self.left is node:
            return True
        else:
            return self.left.has_right(node) or self.left.has_left(node)

    def add_acc_left(self, val):
        max_influence = self.left.max_influence()
        self.left_acc = min(self.left_acc + val, max_influence)

    def add_acc_right(self, val):
        max_influence = self.right.max_influence()
        self.right_acc = min(self.right_acc + val, max_influence)

    def max_influence(self):
        if self.parent == None:
            return 0

        return self.weight / self.parent.weight

class MergeEngine:
    def __init__(self, N):
        self.nodes = [Node(i) for i in range(N)]
        self.latest = N

    def create(self):
        latest = self.latest
        new_node = Node(latest)
        self.nodes.append(new_node)
        self.latest += 1
        return new_node

    def merge(self, a, b, dist):
        root = self.create()
        root.set_left(self.nodes[a])
        root.set_right(self.nodes[b])
        root.merge_dist = dist
        root.weight = root.left.weight + root.right.weight
