from fastcluster import linkage
from disjoint import DisjointSet
from sklearn.cluster import AgglomerativeClustering as SkAgglomerativeClustering
from sklearn.neighbors import BallTree
import numpy as np

class AgglomerativeClustering(SkAgglomerativeClustering):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = None
        self.cluster_centers_ = None

    def fit(self, X, *args):
        super().fit(X, *args)
        self.X = X
        self.cluster_centers_ = self.get_centroids()

    def get_centroids(self):
        clusters_cnt = max(self.labels_) + 1
        centroids = [None for i in range(clusters_cnt)]
        cluster_member_cnt = [0 for i in range(clusters_cnt)]
        for i, x in enumerate(self.X):
            belongs = self.labels_[i]
            cluster_member_cnt[belongs] += 1
            if centroids[belongs] is None:
                centroids[belongs] = x
            else:
                centroids[belongs] += x
        for i, centroid in enumerate(centroids):
            centroids[i] = centroid / cluster_member_cnt[i]
        return centroids

    def predict(self, X):
        centroids = self.cluster_centers_

        ball_tree = BallTree(centroids)
        _, indexes = ball_tree.query(X)

        result = []
        for idx, in indexes:
            result.append(self.labels_[idx])

        return result

class AgglomerativeClusteringMaxMergeDist:
    def __init__(self):
        self.X = None
        self.cluster_centers_ = None
        self.max_merge_dist = None

    def get_centroids(self, X, belong_to):
        clusters_cnt = max(belong_to) + 1
        acc_centroids_by_group = [np.zeros(X[0].shape) for i in range(clusters_cnt)]
        cluster_member_cnt = [0 for i in range(clusters_cnt)]
        for i, x in enumerate(X):
            belongs = belong_to[i]
            cluster_member_cnt[belongs] += 1
            acc_centroids_by_group[belongs] += x

        centroids = [acc_centroid / member_cnt
                     for acc_centroid, member_cnt
                     in zip(acc_centroids_by_group, cluster_member_cnt)]

        return centroids, cluster_member_cnt

    def fit(self, X, max_merge_dist):
        self.X = X
        self.max_merge_dist = max_merge_dist

        merge_hist = linkage(X, method='ward', metric='euclidean', preserve_input=True)

        disjoint = DisjointSet(len(X))

        # _, _, merge_dists, _ = list(zip(*merge_hist))
        # print('merge_dists:', merge_dists)

        for a, b, merge_dist, _ in merge_hist:
            if merge_dist > max_merge_dist:
                break

            a, b = int(a), int(b)
            disjoint.join(a, b)

        belong_to = [disjoint.parent(i) for i in range(len(X))]

        # rename the cluster name to be 0 -> cluster_count - 1
        cluster_map = {}
        cluster_name = 0
        belong_to_renamed = []
        for each in belong_to:
            if not each in cluster_map:
                cluster_map[each] = cluster_name
                cluster_name += 1
            belong_to_renamed.append(cluster_map[each])

        # print('belong_to_renamed:', belong_to_renamed)

        centroids, cluster_member_cnt = self.get_centroids(X, belong_to_renamed)
        self.cluster_centers_ = centroids

        print('centroids:', centroids)
        print('cluster_member_cnt:', cluster_member_cnt)

        return centroids, cluster_member_cnt