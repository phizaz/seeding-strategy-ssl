from sklearn.cluster import KMeans
from collections import Counter
import math

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

    def sigmoid(self, x):
        # sigmoid rebased
        # sigmoid(0) = 0
        # sigmoid(sigmoid_x_is_one) ~= 1
        return 2 / (1 + math.exp(-x)) - 1

    def scale(self, x, weight):
        return self.sigmoid_x_is_one / weight * x

    def score(self, group):
        assert isinstance(group, Group)

        seeding_cnt = group.seeding_cnt()
        if seeding_cnt == 0:
            return 0

        major_label, major_cnt = group.major()
        scaling_factor = self.sigmoid(
            self.scale(seeding_cnt, group.cnt)
        )
        # this will make this algorithm more discrete,
        # works better for pendigits
        scaling_factor = 1

        # if seeding_cnt == 1:
        #     print('scaling:', scaling_factor, 'seeding_cnt:', seeding_cnt, '/', group.cnt)

        score = major_cnt / seeding_cnt * group.cnt * scaling_factor
        return score

    def goodness(self):
        sum_goodness = 0
        for group in self.groups:
            sum_goodness += self.score(group)

        return sum_goodness / len(self.X)

    def badness(self):
        return 1 - self.goodness()

    def run(self, seeding_y, sigmoid_x_is_one=1e-9):
        # sigmoid(x) must be as close as 'precision' to 1 to be considered as 1
        # this the better the precision, the steeper the curve
        # that means it requires less points to gain high score in a given area (voronoid)
        sigmoid_x_is_one = math.log(int(2 * (1 / sigmoid_x_is_one)) - 1)
        self.sigmoid_x_is_one = sigmoid_x_is_one

        self.groups = [Group(i) for i in range(self.clusters_cnt)]

        for x, group, label in zip(self.X, self.grouping_result(), seeding_y):
            self.groups[group].add(x, label)

        return self.badness()



