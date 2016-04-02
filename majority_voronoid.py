from fastcluster import linkage
from collections import Counter
import math

class Node:
    def __init__(self, name):
        self.name = name
        self.cnt = 1
        self.seeding_counter = Counter()
        self.parent = self

    def is_root(self):
        return self.parent is self

    def set_seed(self, label):
        self.seeding_counter[label] = 1

class DisjointSet:
    def __init__(self, sets_cnt):
        self.cnt = sets_cnt
        self.nodes = [Node(i) for i in range(2 * sets_cnt - 1)]
        self.highest = sets_cnt

    def parent(self, node):
        assert isinstance(node, Node)

        if node.parent is not node:
            node.parent = self.parent(node.parent)
        return node.parent

    def join(self, a_name, b_name):
        assert isinstance(a_name, int)
        assert isinstance(b_name, int)

        a = self.nodes[a_name]
        b = self.nodes[b_name]

        a = self.parent(a)
        b = self.parent(b)

        if a == b:
            raise Exception('joining the same set')

        c = self.nodes[self.highest]
        self.highest += 1

        c.cnt = a.cnt + b.cnt
        c.seeding_counter = a.seeding_counter + b.seeding_counter

        a.parent = c
        b.parent = c


    @staticmethod
    def sigmoid(x):
        # sigmoid rebased
        # sigmoid(0) = 0
        # sigmoid(sigmoid_x_is_one) ~= 1
        return 2 / (1 + math.exp(-x)) - 1

    @staticmethod
    def scale(x, weight, sigmoid_x_is_one):
        return sigmoid_x_is_one / weight * x

    @staticmethod
    def node_goodness(node, sigmoid_x_is_one):
        # maximum goodness is node.cnt
        assert isinstance(node, Node)

        if len(node.seeding_counter) == 0:
            return 0

        seeding_cnt = sum(cnt for _, cnt in node.seeding_counter.items())
        major_label, major_cnt = node.seeding_counter.most_common(1).pop()
        # print('major_label:', major_label)
        # print('major_cnt:', major_cnt, 'seeding:', seeding_cnt, '/', node.cnt)
        scaling_factor = DisjointSet.sigmoid(
                DisjointSet.scale(seeding_cnt, node.cnt, sigmoid_x_is_one))
        # scaling_factor = 1

        score = major_cnt / seeding_cnt * node.cnt * scaling_factor
        # print('score:', score, '/', node.cnt)
        return score

    def goodness(self, sigmoid_x_is_one):
        sum_goodness = 0
        for node in self.nodes:
            if not node.is_root():
                continue

            sum_goodness += DisjointSet.node_goodness(node, sigmoid_x_is_one)

        return sum_goodness / self.cnt


    def badness(self, sigmoid_x_is_one):
        bad = 1 - self.goodness(sigmoid_x_is_one)
        # print('badness:', bad)
        return bad

class MajorityVoronoid:
    def __init__(self, X):
        self.X = X
        self.merge_hist = self.prepare(X)

    def prepare(self, X):
        merge_hist = linkage(X, method='ward', metric='euclidean', preserve_input=True)
        return merge_hist

    def run(self, seeding_y, sigmoid_x_is_one):
        # sigmoid(x) must be as close as 'precision' to 1 to be considered as 1
        # this the better the precision, the steeper the curve
        # that means it requires less points to gain high score in a given area (voronoid)
        sigmoid_x_is_one = math.log(int(2 * (1 / sigmoid_x_is_one)) - 1)
        # print('sigmoid_x_is_one:', sigmoid_x_is_one)
        self.sigmoid_x_is_one = sigmoid_x_is_one

        disjoint_set = DisjointSet(len(self.X))

        # set seeding
        for i, label in enumerate(seeding_y):
            if label is None:
                continue
            disjoint_set.nodes[i].set_seed(label)

        best_badness = disjoint_set.badness(sigmoid_x_is_one)

        # merge nodes to create a tree
        # step by step
        for i, (a, b, merge_dist, _) in enumerate(self.merge_hist):
            clusters_cnt = len(self.X) - i - 1
            a, b = int(a), int(b)
            disjoint_set.join(a, b)
            best_badness = min(disjoint_set.badness(sigmoid_x_is_one),
                               best_badness)

        return best_badness

