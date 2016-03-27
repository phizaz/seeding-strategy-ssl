from fastcluster import linkage
from disjoint import DisjointSet
from collections import deque
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import BallTree
import time
from itertools import islice
import numpy
import math
from collections import Counter

class Result:
    def __init__(self, labels, centers):
        self.labels_ = labels
        self.cluster_centers_ = centers

    def predict(self, X):
        ball_tree = BallTree()
        ball_tree.fit(self.cluster_centers_)

        _, indexes = ball_tree.query(X)
        result = []
        for idx, in indexes:
            result.append(self.labels_[idx])

        return result

def f_creator(coef, intercept):
    def f(x):
        return intercept + coef * x

    return f

def best_fit_line(x, y):
    coef, intercept = numpy.polyfit(x, y, 1)
    return coef, intercept

def plot(X, fn):
    return [fn(x) for x in X]

def single_cluster(coef_a, coef_b, rthreshold=0.01):
    # determine if there is only one cluster
    # if the two slopes are different so slightly enough
    # nevertheless, this will fail if not counting the bigger picture as well!!

    # we use arctan instead of the slope
    # because slopes don't act in a uniform way
    # but radians do
    angle_a = math.atan2(coef_a, 1)
    angle_b = math.atan2(coef_b, 1)
    # relative difference of the absolute mean of the two
    avg = abs(angle_a + angle_b) / 2
    # print('avg:', avg)
    # print('coef_a:', coef_a, 'angle_a:', angle_a)
    # print('coef_b:', coef_b, 'angle_b:', angle_b)
    relative_difference = abs(angle_a - angle_b) / avg
    # print('relative_difference:', relative_difference)
    return relative_difference <= rthreshold

def l_method(num_groups, merge_dist):
    element_cnt = len(num_groups)

    # short circuit, since the l-method doesn't work with the number of elements below 4
    if element_cnt < 4:
        return 1

    # now we have some level of confidence that O(n) is not attainable
    # this l_method is gonna be slow... n * 2 * O(MSE)

    x_left = num_groups[:2]
    y_left = merge_dist[:2]

    # we use 'deque' data structure here to attain the efficient 'popleft'
    x_right = deque(num_groups[2:])
    y_right = deque(merge_dist[2:])

    min_score = float('inf')
    min_c = None

    # this is for determining single cluster problem
    # min_coef_left = 0
    # min_coef_right = 0

    for left_cnt in range(2, element_cnt - 2 + 1):
        # get best fit lines
        coef_left, intercept_left = best_fit_line(x_left, y_left)
        coef_right, intercept_right = best_fit_line(x_right, y_right)

        fn_left = f_creator(coef_left, intercept_left)
        fn_right = f_creator(coef_right, intercept_right)

        y_pred_left = plot(x_left, fn_left)
        y_pred_right = plot(x_right, fn_right)

        # calculate the error on each line
        mseA = mean_squared_error(y_left, y_pred_left)
        mseB = mean_squared_error(y_right, y_pred_right)

        # calculate the error on both line cumulatively
        A = left_cnt / element_cnt * mseA
        B = (element_cnt - left_cnt) / element_cnt * mseB
        score = A + B

        x_left.append(num_groups[left_cnt])
        y_left.append(merge_dist[left_cnt])

        x_right.popleft()
        y_right.popleft()

        if score < min_score:
            # find the best pair of best fit lines (that has the lowest mse)

            # left_cnt is not the number of clusters
            # since the first num_group begins with 2
            min_c, min_score = left_cnt + 1, score

            # for determining single class problem
            # min_coef_left, min_coef_right = coef_left, coef_right

    return min_c

    # this won't work for the moment
    # if single_cluster(min_coef_left, min_coef_right, 0.01):
    #     # two lines are too close in slope to each other (1% tolerance)
    #     # considered to be a single line
    #     # thus, single cluster
    #     return 1
    # else:
    #     return min_c

def refined_l_method(num_groups, merge_dist):
    element_cnt = cutoff = last_knee = current_knee = len(num_groups)

    # short circuit, since the l-method doesn't work with the number of elements below 4
    if element_cnt < 4:
        return 1

    while True:
        last_knee = current_knee
        # print('cutoff:', cutoff)
        current_knee = l_method(num_groups[:cutoff], merge_dist[:cutoff])
        # print('current_knee:', current_knee)

        # you can keep this number high (* 2), and no problem with that
        # just make sure that the cutoff tends to go down every time
        # but, we use 2 here according to the paper
        cutoff = current_knee * 3
        if current_knee >= last_knee:
            break

    return current_knee

def get_centroids(X, belong_to):
    clusters_cnt = max(belong_to) + 1
    centroids = [numpy.zeros(X[0].shape) for i in range(clusters_cnt)]
    cluster_member_cnt = [0 for i in range(clusters_cnt)]
    for i, x in enumerate(X):
        belongs = belong_to[i]
        cluster_member_cnt[belongs] += 1
        centroids[belongs] += x
    for i, centroid in enumerate(centroids):
        centroids[i] = centroid / cluster_member_cnt[i]
    return centroids

def agglomerative_l_method(X):
    # library: fastcluster
    merge_hist = linkage(X, method='ward', metric='euclidean', preserve_input=True)

    # reorder to be x [2->N]
    num_groups = [i for i in range(2, len(X) + 1)]
    merge_dist = list(reversed([each[2] for each in merge_hist]))

    cluster_count = refined_l_method(num_groups, merge_dist)

    # print('refined_l_method time:', end_time - start_time)
    # print('cluster_count:', cluster_count)

    # make clusters by merging them according to merge_hist
    disjoint = DisjointSet(len(X))
    for a, b, _, _ in islice(merge_hist, 0, len(X) - cluster_count):
        a, b = int(a), int(b)
        disjoint.join(a, b)

    # get cluster name for each instance
    belong_to = [disjoint.parent(i) for i in range(len(X))]
    # print('belong_to:', belong_to)
    # counter = Counter(belong_to)
    # print('belong_to:', counter)

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

    centroids = get_centroids(X, belong_to_renamed)
    # print('centroids:', centroids)

    return Result(belong_to_renamed, centroids)

def recursive_agglomerative_l_method(X):
    raise Exception('out of service !')
    # won't give any disrable output for the moment

    def recursion(X):
        belong_to = agglomerative_l_method(X)
        num_clusters = max(belong_to) + 1

        if num_clusters == 1:
            return belong_to, num_clusters

        new_belong_to = [None for i in range(len(belong_to))]
        next_cluster_name = 0
        for cluster in range(num_clusters):
            next_X = []
            for belong, x in zip(belong_to, X):
                if belong == cluster:
                    next_X.append(x)
            sub_belong, sub_num_clusters = recursion(next_X)
            sub_belong_itr = 0
            for i, belong in enumerate(belong_to):
                if belong == cluster:
                    new_belong_to[i] = sub_belong[sub_belong_itr] + next_cluster_name
                    sub_belong_itr += 1
            next_cluster_name += sub_num_clusters
        return new_belong_to, next_cluster_name

    belong_to, clusters = recursion(X)
    # print('belong_to:', belong_to)
    # print('clusters:', clusters)
    centroids = get_centroids(X, belong_to)
    # print('centroids:', centroids)
    return Result(belong_to, centroids)
