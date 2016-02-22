from fastcluster import linkage
from disjoint import DisjointSet
from collections import deque
import time
from itertools import islice
import numpy

def f_creator(coef, intercept):
    def f(x):
        return intercept + coef * x

    return f


def best_fit_line(x, y):
    # regression = LinearRegression()
    # regression.fit(x, y)
    # coef = regression.coef_[0]
    # intercept = regression.intercept_

    coef, intercept = numpy.polyfit(x, y, 1)
    return coef, intercept


def mean_squared_error(X, Y):
    # start_time = time.time()
    coef, intercept = best_fit_line(X, Y)
    # end_time = time.time()
    # print('best_fit time:', end_time - start_time)

    f = f_creator(coef, intercept)

    sum = 0
    for arr_x, real_y in zip(X, Y):
        x = arr_x
        y = f(x)
        sum += (real_y - y) ** 2
    mse = sum / len(Y)
    return mse


def l_method(num_groups, merge_dist):
    # print(num_groups)
    # print(merge_dist)

    b = len(num_groups) + 1

    # start_time = time.time()
    x_left = num_groups[:2]
    y_left = merge_dist[:2]
    # we use 'deque' data structure here to attain the efficient 'popleft'
    x_right = deque(num_groups[2:])
    y_right = deque(merge_dist[2:])
    # end_time = time.time()
    # print('list preparation time:', end_time - start_time)

    min_score = float('inf')
    min_c = None
    for c in range(3, b - 2):
        # start_time = time.time()
        mseA = mean_squared_error(x_left, y_left)
        mseB = mean_squared_error(x_right, y_right)
        # end_time = time.time()

        # if c % 13 == 0:
        #     print('c:', c)
        #     print('mean_squared_time:', end_time - start_time)
        A = (c - 1) / (b - 1) * mseA
        B = (b - c) / (b - 1) * mseB
        score = A + B

        if score < min_score:
            # print('score:', score)
            # print('c:', c)
            # print('A:', A)
            # print('B:', B)
            # print('mseA:', mseA)
            # print('mseB:', mseB)
            min_c, min_score = c, score

        # start_time = time.time()
        x_left.append(num_groups[c - 1])
        y_left.append(merge_dist[c - 1])

        x_right.popleft()
        y_right.popleft()
        # end_time = time.time()
        # print('list manipulation time:', end_time - start_time)

    return min_c


def refined_l_method(num_groups, merge_dist):
    cutoff = last_knee = current_knee = len(num_groups)
    while True:
        last_knee = current_knee
        print('cutoff:', cutoff)
        current_knee = l_method(num_groups[:cutoff], merge_dist[:cutoff])
        print('current_knee:', current_knee)
        # this actually modified from original version *2 -> *3
        # tested to have better results
        # you can keep this number high, and no problem with that
        # just make sure that the cutoff tends to go down every time
        cutoff = current_knee * 3
        if current_knee >= last_knee:
            break
    return current_knee


def agglomerative_l_method(x):
    # library: fastcluster
    merge_hist = linkage(x, method='ward', metric='euclidean', preserve_input=True)
    # for each in merge_hist:
    #     a, b, dist, cnt = each
    #     a, b, cnt = int(a), int(b), int(cnt)
    # print(a, b, cnt, dist)

    # if abs(c - 0) < .000001:
    #     print('point:', x[a], x[b])

    # reorder to be x [2->N]
    num_groups = [i for i in range(2, len(x) + 1)]
    merge_dist = list(reversed([each[2] for each in merge_hist]))
    # print('num_groups:', num_groups[:100])
    # print('merge_dist:', merge_dist[:100])
    # print(num_groups)
    # print(merge_dist)

    start_time = time.time()
    cluster_count = refined_l_method(num_groups, merge_dist)
    end_time = time.time()

    print('refined_l_method time:', end_time - start_time)

    print('cluster_count:', cluster_count)

    # make clusters by merging them according to merge_hist
    disjoint = DisjointSet(len(x))
    for a, b, _, _ in islice(merge_hist, 0, len(x) - cluster_count):
        a, b = int(a), int(b)
        disjoint.join(a, b)

    # get cluster name for each instance
    belong_to = [disjoint.parent(i) for i in range(len(x))]
    # print('belong_to:', belong_to)

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

    return belong_to_renamed