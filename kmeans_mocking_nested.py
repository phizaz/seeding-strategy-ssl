from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from collections import Counter
from l_method import agglomerative_l_method
from agglomerative_clustering import AgglomerativeClustering
import math

'''
Kmeans Mocking Nested (output with bounds)
kmeans with sub-groups (using l-method + agglomerative clustering or kmeans)
test result is that using agglomerative clustering will have the looser lower bound
but, better upper bound and conformation

result: very good upper bound for iris dataset -- loose lower bound
not at a usable level for other datasets
'''

class Group:
    def __init__(self, name):
        self.name = name
        self.cnt = 0
        self.seeding_counter = Counter()
        self.X = []
        self.y_seed = []
        self.sub_clusters_cnt = None
        self.sub_centroids = []
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

    def has_collision(self):
        # seeded group is said to be
        seeds = list(filter(lambda xy: xy[1] is not None,
                            zip(self.X, self.y_seed)))

        seed_x, seed_y = list(zip(*seeds))
        seed_groups = self.clustering_model.predict(seed_x)

        label_by_group = {}
        for cluster, label in zip(seed_groups, seed_y):
            if not cluster in label_by_group:
                label_by_group[cluster] = label
            elif label_by_group[cluster] != label:
                return True

        return False

    def cluster(self):
        l_method = agglomerative_l_method(self.X)
        self.sub_clusters_cnt = len(l_method.cluster_centers_)
        # print('sub_clusters_cnt:', self.sub_clusters_cnt, 'cnt:', self.cnt)

        while True:
            self.clustering_model = KMeans(self.sub_clusters_cnt)
            # self.clustering_model = AgglomerativeClustering(self.sub_clusters_cnt)
            self.clustering_model.fit(self.X)

            if self.has_collision():
                # increase sub-cluster count until there is no ambiguous seeding
                self.sub_clusters_cnt += 1
                print('sub_clusters_cnt:', self.sub_clusters_cnt, 'seed cnt:', self.seeding_cnt(), 'cnt:', self.cnt)
            else:
                break

class KmeansMockingNested:

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

        count_by_group = Counter(group.clustering_model.labels_)

        # seeded group is said to be
        seeds = list(filter(lambda xy: xy[1] is not None,
                            zip(group.X, group.y_seed)))

        seed_x, seed_y = list(zip(*seeds))
        seed_groups = group.clustering_model.predict(seed_x)

        # no of points in seeded groups (certain groups)
        certain_cnt = sum(count_by_group[cluster] for cluster in set(seed_groups))
        uncertain_cnt = group.cnt - certain_cnt

        groups_by_label = {}
        for cluster, label in zip(seed_groups, seed_y):
            if not label in groups_by_label:
                groups_by_label[label] = set()

            groups_by_label[label].add(cluster)

        major_label, major_cnt = group.major()
        major_groups = groups_by_label[major_label]
        major_cnt = sum(count_by_group[g] for g in major_groups)

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