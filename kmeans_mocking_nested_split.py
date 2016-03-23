from sklearn.cluster import KMeans
from dividable_clustering import DividableClustering
from collections import Counter
from l_method import agglomerative_l_method
from agglomerative_clustering import AgglomerativeClustering
import math

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

    def has_collision(self, X, y_seed, model = None):
        # seeded group is said to be
        seeds = list(filter(lambda xy: xy[1] is not None,
                            zip(X, y_seed)))

        # no seeds, no collision
        if len(seeds) == 0:
            return False

        seed_x, seed_y = list(zip(*seeds))

        if model is None:
            seed_groups = [0 for i in range(len(seed_x))]
        else:
            seed_groups = model.predict(seed_x)

        label_by_group = {}
        for cluster, label in zip(seed_groups, seed_y):
            if not cluster in label_by_group:
                label_by_group[cluster] = label
            elif label_by_group[cluster] != label:
                return True

        return False

    def cluster(self):
        l_method = agglomerative_l_method(self.X)
        suggest_cluster_cnt = len(l_method.cluster_centers_)
        # print('sub_clusters_cnt:', self.sub_clusters_cnt, 'cnt:', self.cnt)

        agg = AgglomerativeClustering(suggest_cluster_cnt)
        agg.fit(self.X)

        # first tier clustering, using agglomerative clustering
        self.clustering_model = DividableClustering()
        self.clustering_model.fit(self.X, agg.labels_)

        # second tier, using kmeans
        for suspect_label in range(suggest_cluster_cnt):
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
            cluster_cnt = 2
            while True:
                kmeans = KMeans(cluster_cnt)
                kmeans.fit(X)

                if not self.has_collision(X, y_seed, kmeans):
                    self.clustering_model.split(suspect_label, kmeans.labels_)
                    break

                print('split sub_clusters_cnt:', cluster_cnt, 'cnt:', len(X), 'main cnt:', self.cnt)
                cluster_cnt += 1

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

        count_by_group = Counter(group.clustering_model.predict(group.X))

        # seeded group is said to be
        seeds = list(filter(lambda xy: xy[1] is not None,
                            zip(group.X, group.y_seed)))

        seed_x, seed_y = list(zip(*seeds))
        seed_groups = group.clustering_model.predict(seed_x)

        # no of points in seeded groups (certain groups)
        certain_cnt = sum(count_by_group[g] for g in set(seed_groups))
        uncertain_cnt = group.cnt - certain_cnt

        label_cnt_by_group = {}
        for g, label in zip(seed_groups, seed_y):
            if not g in label_cnt_by_group:
                label_cnt_by_group[g] = Counter()
            label_cnt_by_group[g][label] += 1

        # print('label_cnt_by_group:', label_cnt_by_group)

        major_label, _ = group.major()

        def sum_seeds_in_group(g):
            return sum(seed_cnt for _, seed_cnt in label_cnt_by_group[g].items())

        major_cnt = 0
        for g in set(seed_groups):
            major_cnt += count_by_group[g] * label_cnt_by_group[g][major_label] / sum_seeds_in_group(g)

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