from collections import Counter
from l_method import agglomerative_l_method
from dividable_clustering import DividableClustering

'''
Cluster Mocking Nested Ratio (with bounds)
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
        self.sub_clusters_cnt = None
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

    def cluster(self):
        l_method = agglomerative_l_method(self.X)
        self.sub_clusters_cnt = len(l_method.cluster_centers_)
        # print('sub_clusters_cnt:', self.sub_clusters_cnt, 'cnt:', self.cnt)

        self.clustering_model = DividableClustering()
        self.clustering_model.fit(self.X, l_method.labels_)

class ClusterMockingNestedRatio:
    def __init__(self, X, labels):
        self.clusters_cnt = max(labels) + 1
        self.X = X
        self.labels = labels
        self.groups = []

    def grouping_result(self):
        return self.labels

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

        # no of points in seeded groups (certain groups)
        certain_cnt = sum(count_by_label[l] for l in set(seeded_labels))
        uncertain_cnt = group.cnt - certain_cnt

        label_cnt_by_label = {}
        for label, y in zip(seeded_labels, seed_y):
            if not label in label_cnt_by_label:
                label_cnt_by_label[label] = Counter()
            label_cnt_by_label[label][y] += 1

        # print('label_cnt_by_group:', label_cnt_by_label)

        major_label, _ = group.major()

        def sum_seeds_in_label(label):
            return sum(seed_cnt for _, seed_cnt in label_cnt_by_label[label].items())

        major_cnt = 0
        for label in set(seeded_labels):
            major_cnt += count_by_label[label] * label_cnt_by_label[label][major_label] / sum_seeds_in_label(label)

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

        return self.goodness()