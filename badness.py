from sklearn.neighbors import BallTree
from numpy import inner
from fastcluster import linkage
from mergeengine import MergeEngine

def rmsd_nearest_from_centroids(seeding, centroids):
    # root mean squared distance from each centroids to its closest seeding
    ball_tree = BallTree(seeding)
    dist, idx = ball_tree.query(centroids)

    # root mean squared distance
    sum_sqdist = sum(d[0] ** 2 for d in dist)
    mean = sum_sqdist / len(centroids)
    return mean ** 0.5

def md_nearest_from_centroids(seeding, centroids):
    # mean distance
    ball_tree = BallTree(seeding)
    dist, idx = ball_tree.query(centroids)
    sum_dist = sum(d[0] for d in dist)
    mean = sum_dist / len(centroids)
    return mean

def md_weighted_nearest_from_centroids(seeding, centroids, weights):
    assert len(centroids) == len(weights)

    sum_weight = sum(weights)

    ball_tree = BallTree(seeding)
    dist, idx = ball_tree.query(centroids)
    sum_weighted_dist = sum(d[0] * weight for d, weight in zip(dist, weights))
    mean = sum_weighted_dist / sum_weight
    return mean

def voronoid_filling(seeding, centroids, weights):
    assert len(centroids) == len(weights)

    ball_tree = BallTree(centroids)
    _, indexes = ball_tree.query(seeding)
    filled_centroids = set()

    sum_weights = sum(weights)
    badness = sum_weights
    for idx, in indexes:
        if idx in filled_centroids:
            continue

        filled_centroids.add(idx)
        badness -= weights[idx]

    return badness / sum_weights


class HierarchicalVoronoidFilling:

    def __init__(self, X):
        self.X = X
        self.merge_hist = self.prepare(X)

    def prepare(self, X):
        merge_hist = linkage(X, method='ward', metric='euclidean', preserve_input=True)
        return merge_hist

    def run(self, seeding, c=0.03):
        # create merge tree
        tree = MergeEngine(len(self.X))
        for a, b, merge_dist, _ in self.merge_hist:
            a, b = int(a), int(b)
            tree.merge(a, b, merge_dist)
            # print('merge_dist:', merge_dist)

        def influence(dist, level):
            if dist == 0:
                return 1

            return 1 / dist * c

        ball_tree = BallTree(self.X)
        _, seeding_indexes = ball_tree.query(seeding)
        for idx, in seeding_indexes:
            node = tree.nodes[idx]
            # print('node:', node.name)
            # special case, leaf nodes
            node.left_acc = 0.5
            node.right_acc = 0.5

            for level, (parent, is_left) in enumerate(node.to_root()):
                # print('parents:', parent.name, 'dist:', parent.merge_dist)
                # add influence to the accumulator of its parent and its fore-parents
                addition = influence(parent.merge_dist, level)
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
            acc = min(tree.nodes[i].sum_acc_to_root(), 1)
            # print('acc:', acc)
            badness -= acc

        return badness / len(self.X)