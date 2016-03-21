from sklearn.neighbors import BallTree
from fastcluster import linkage
import math

class Node:
    def __init__(self, name):
        self.name = name
        self.left_acc = 0
        self.right_acc = 0
        self.merge_dist = 0
        self.weight = 1

        self._sum_acc_to_root = None

        self.parent = None
        self.left = None
        self.right = None

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return self.left is None and self.right is None

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
        # max_influence = self.left.max_influence()
        # self.left_acc = min(self.left_acc + val, max_influence)
        self.left_acc += val

    def add_acc_right(self, val):
        # max_influence = self.right.max_influence()
        # self.right_acc = min(self.right_acc + val, max_influence)
        self.right_acc += val

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


class HierarchicalVoronoidFilling:

    def __init__(self, X):
        self.X = X
        self.merge_hist = self.prepare(X)
        self.sigmoid_x_is_one = None

    def prepare(self, X):
        merge_hist = linkage(X, method='ward', metric='euclidean', preserve_input=True)
        return merge_hist

    def sigmoid(self, x):
        # sigmoid rebased
        # sigmoid(0) = 0
        # sigmoid(sigmoid_x_is_one) ~= 1
        return 2 / (1 + math.exp(-x)) - 1

    def scale(self, x, weight):
        return self.sigmoid_x_is_one / weight * x

    def sum_acc_sigmoid(self, node):
        # left_acc and right_acc
        if node.is_leaf():
            return node.left_acc + node.right_acc

        acc = [node.left_acc, node.right_acc]
        weight = [node.left.max_influence(), node.right.max_influence()]
        scaled_x = list(map(lambda aw: self.scale(aw[0], aw[1]),
                            zip(acc, weight)))
        sigmoid_left, sigmoid_right = list(map(self.sigmoid, scaled_x))
        result = (sigmoid_left * weight[0] + sigmoid_right * weight[1]) / sum(weight)
        # print('result:', result)
        return result

    def sum_acc_to_root(self, node):
        if node.is_root():
            node._sum_acc_to_root = self.sum_acc_sigmoid(node)

        if node._sum_acc_to_root is None:
            node._sum_acc_to_root = self.sum_acc_sigmoid(node) \
                                    + self.sum_acc_to_root(node.parent)

        return node._sum_acc_to_root

    def run(self, seeding, c=0.03, sigmoid_x_is_one=1e-9):
        # sigmoid(x) must be as close as 'precision' to 1 to be considered as 1
        # this the better the precision, the steeper the curve
        # that means it requires less points to gain high score in a given area (voronoid)
        sigmoid_x_is_one = math.log(int(2 * (1 / sigmoid_x_is_one)) - 1)
        self.sigmoid_x_is_one = sigmoid_x_is_one

        # print('sigmoid_x_is_one:', sigmoid_x_is_one)

        # create merge tree
        tree = MergeEngine(len(self.X))
        for a, b, merge_dist, _ in self.merge_hist:
            a, b = int(a), int(b)
            tree.merge(a, b, merge_dist)
            # print('merge_dist:', merge_dist)

        def influence(dist, level):
            if dist == 0:
                return 1

            return c * 1 / math.pow(dist, 1)

        ball_tree = BallTree(self.X)
        _, seeding_indexes = ball_tree.query(seeding)
        for idx, in seeding_indexes:
            node = tree.nodes[idx]
            # print('node:', node.name)

            # special case, leaf nodes
            node.left_acc = 0.5
            node.right_acc = 0.5

            acc_dist = 0
            for level, (parent, is_left) in enumerate(node.to_root()):
                acc_dist += parent.merge_dist
                # print('parents:', parent.name, 'dist:', parent.merge_dist)
                # add influence to the accumulator of its parent and its fore-parents
                addition = influence(acc_dist, level)
                # print('add:', addition)
                if is_left:
                    parent.add_acc_left(addition)
                else:
                    parent.add_acc_right(addition)

        # for node in tree.nodes:
        #     print('node:', node.name, 'acc:', node.acc)

        # root = tree.nodes[-1]
        # print('root name:', root.name)
        # print('root left_acc:', root.left_acc)
        # print('max inf_left:', root.left.max_influence())
        # print('root right_acc:', root.right_acc)
        # print('max inf_right:', root.right.max_influence())
        # print('root merge_dist:', root.merge_dist)
        # print('root weight:', root.weight)

        badness = len(self.X)
        for i in range(len(self.X)):
            acc = min(self.sum_acc_to_root(tree.nodes[i]), 1)
            # print('acc:', acc)
            badness -= acc

        return badness / len(self.X)