from sklearn.cluster import KMeans
from dividable_clustering import DividableClustering
from collections import Counter
from l_method import agglomerative_l_method
from agglomerative_clustering import AgglomerativeClustering
import math
import numpy as np

'''
Kmeans Mocking Nested Ratio (with bounds)
An attemp to tighten the lower bound
kmeans with sub-groups (using l-method + agglomerative clustering)
— tighter lower bound — ignore the multi label seeded groups
and use the majority instead
'''

class Group:
    def __init__(self, name):
        self.name = name
        self.cnt = 0
        self.seeding_counter = Counter()
        self.X = []
        self.y_seed = []
        self.clustering_model = None

    def seeding_cnt(self):
        return sum(cnt for _, cnt in self.seeding_counter.items())

    def add(self, x, y=None):
        self.cnt += 1
        self.X.append(x)
        self.y_seed.append(y)

        if y is not None:
            self.seeding_counter[y] += 1

    def major(self):
        # returns label, cnt
        return self.seeding_counter.most_common(1).pop()

    def get_seeds(self, X, y_seed):
        seeds = list(filter(lambda xy: xy[1] is not None,
                            zip(X, y_seed)))
        return seeds

    def seeds(self):
        return self.get_seeds(self.X, self.y_seed)

    def has_collision(self, X, y_seed, model = None):
        # seeded group is said to be
        seeds = self.get_seeds(X, y_seed)

        # no seeds, no collision
        if len(seeds) == 0:
            return False

        seed_x, seed_y = list(zip(*seeds))

        if model is None:
            seed_groups = [0 for i in range(len(seed_x))]
        else:
            seed_groups = model.predict(seed_x)

        y_by_label = {}
        for label, y in zip(seed_groups, seed_y):
            if not label in y_by_label:
                y_by_label[label] = y
            elif y_by_label[label] != y:
                return True

        return False

    def cluster(self):
        l_method = agglomerative_l_method(self.X)

        # first tier clustering, using agglomerative clustering
        self.clustering_model = DividableClustering()
        self.clustering_model.fit(self.X, l_method.labels_)

        # second tier, using kmeans
        for suspect_label in range(self.clustering_model.latest_label):
            ind_X = self.clustering_model.get_X_with_idx(suspect_label)
            y_seed = []
            X = []
            for x, idx in ind_X:
                X.append(x)
                y_seed.append(self.y_seed[idx])

            # no collision in this sub-group
            if not self.has_collision(X, y_seed):
                continue

            # there is collisions in this sub-group
            low_cnt = 2
            high_cnt = len(X)
            last_possible_labels = None
            while low_cnt <= high_cnt:
                # 1/4 biased binary search
                cluster_cnt = int((high_cnt - low_cnt) * 1/4 + low_cnt)
                kmeans = KMeans(cluster_cnt)
                kmeans.fit(X)

                if not self.has_collision(X, y_seed, kmeans):
                    last_possible_labels = kmeans.labels_
                    high_cnt = cluster_cnt - 1
                else:
                    low_cnt = cluster_cnt + 1

            print('split sub_clusters_cnt:', cluster_cnt, 'cnt:', len(X), 'main cnt:', self.cnt)
            self.clustering_model.split(suspect_label, last_possible_labels)

        self.clustering_model.relabel()


class KmeansMockingNestedSplit:

    def __init__(self, clusters_cnt, X):
        self.kmeans = KMeans(clusters_cnt)
        self.clusters_cnt = clusters_cnt
        self.X = list(X)
        self.groups = []

        self.prepare(X)

    def grouping_result(self):
        return self.kmeans.labels_

    def prepare(self, X):
        self.kmeans.fit(X)

    def score(self, group):
        assert isinstance(group, Group)

        # short circuit
        seeding_cnt = group.seeding_cnt()
        if seeding_cnt == 0:
            # this depends on the actual implementation of ssl-kmeans
            return 0, 0

        # cluster the sub-clusters
        group.cluster()

        count_by_label = Counter(group.clustering_model.predict_nn(group.X))

        # print('count by label:', count_by_label)

        # seeded group is said to be
        seeds = list(filter(lambda xy: xy[1] is not None,
                            zip(group.X, group.y_seed)))
        seed_x, seed_y = list(zip(*seeds))
        seeded_labels = group.clustering_model.predict_nn(seed_x)

        # print('seeded labels:', seeded_labels)
        # print('seeded labels:', group.clustering_model.predict_centroids(seed_x))

        # no of points in seeded groups (certain groups)
        certain_cnt = sum(count_by_label[l] for l in set(seeded_labels))
        uncertain_cnt = group.cnt - certain_cnt

        # count the major_y clusters members
        major_y, _ = group.major()
        major_cnt = 0
        added = set()
        for label, y in zip(seeded_labels, seed_y):
            if y == major_y and label not in added:
                # don't count twice !
                added.add(label)
                major_cnt += count_by_label[label]

        lower_bound = major_cnt
        upper_bound = major_cnt + uncertain_cnt

        # print('group:', group.name, 'bound:', lower_bound, '/', upper_bound, '/', group.cnt)

        return lower_bound, upper_bound

    def goodness(self):
        sum_lower_bound = 0
        sum_upper_bound = 0
        for group in self.groups:
            lower_bound, upper_bound = self.score(group)
            sum_lower_bound += lower_bound
            sum_upper_bound += upper_bound

        normal_lower_bound = sum_lower_bound / len(self.X)
        normal_upper_bound = sum_upper_bound / len(self.X)

        return normal_lower_bound, normal_upper_bound

    def badness(self):
        good_lower_bound, good_upper_bound = self.goodness()
        bad_lower_bound = 1 - good_upper_bound
        bad_upper_bound = 1 - good_lower_bound
        return bad_lower_bound, bad_upper_bound

    def run(self, seeding_y):
        self.groups = [Group(i) for i in range(self.clusters_cnt)]

        for x, group, label in zip(self.X, self.grouping_result(), seeding_y):
            self.groups[group].add(x, label)

        # return self.badness()
        return self.goodness()