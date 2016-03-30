from sklearn.cluster import KMeans
from collections import Counter
import math

'''
Kmeans Mocking
given the number of SSL clusters and use the same clustering algorithm (kmeans)
to cluster and predict based on majority of seeding points on each voronoid area
to prevent the over-confident problem, we use sigmoid function as a certainty function,
in other words, the very first points will have higher contributer compare to
the consecutive ones.

result: no good, it is impossible to adjust the sigmoid parameter to match every test data
because: we ignore the actual data distribution inside a given voronoid area
'''

class Group:
    def __init__(self, name):
        self.name = name
        self.cnt = 0
        self.seeding_counter = Counter()
        self.X = []

    def seeding_cnt(self):
        return sum(cnt for _, cnt in self.seeding_counter.items())

    def add(self, x, y = None):
        self.cnt += 1
        self.X.append(x)

        if y is not None:
            self.seeding_counter[y] += 1

    def major(self):
        # returns label, cnt
        return self.seeding_counter.most_common(1).pop()

class KmeansMocking:

    def __init__(self, clusters_cnt, X):
        self.kmeans = KMeans(clusters_cnt)
        self.clusters_cnt = clusters_cnt
        self.X = X
        self.groups = []
        self.sigmoid_x_is_one = None

        self.prepare(X)

    def grouping_result(self):
        return self.kmeans.labels_

    def prepare(self, X):
        self.kmeans.fit(X)

    def score(self, group):
        assert isinstance(group, Group)

        seeding_cnt = group.seeding_cnt()
        if seeding_cnt == 0:
            return 0

        major_label, major_cnt = group.major()
        scaling_factor = 1

        score = major_cnt / seeding_cnt * group.cnt * scaling_factor
        return score

    def goodness(self):
        sum_goodness = 0
        for group in self.groups:
            sum_goodness += self.score(group)

        return sum_goodness / len(self.X)

    def badness(self):
        return 1 - self.goodness()

    def run(self, seeding_y):
        self.groups = [Group(i) for i in range(self.clusters_cnt)]

        for x, group, label in zip(self.X, self.grouping_result(), seeding_y):
            self.groups[group].add(x, label)

        return self.badness()



