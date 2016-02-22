import time

import numpy

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from fastcluster import linkage
from collections import deque
from plottools import scatter2d

def kmeans(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']

        kmeans = KMeans(*args, **margs)
        kmeans.fit(x)

        return inst.set('model', kmeans).set('prediction', kmeans.labels_)

    return fn

def knn(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')
        if not 'y' in inst:
            raise Exception('no y')

        x = inst['x']
        y = inst['y']

        knn = KNeighborsClassifier(*args, **margs)
        knn.fit(x, y)

        return inst.set('model', knn)

    return fn

def linear_regression(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')
        if not 'y' in inst:
            raise Exception('no y')

        x = inst['x']
        y = inst['y']

        regression = LinearRegression(*args, **margs)
        regression.fit(x, y)
        #
        # print('coef:', regression.coef_)
        # print('intercept:', regression.intercept_)

        return inst.set('model', regression)

    return fn

def agglomerative(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']

        agg = AgglomerativeClustering(*args, **margs)
        agg.fit(x)

        return inst.set('model', agg).set('prediction', agg.labels_)

    return fn

def agglomerative_fast(*args, **margs):
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
            cutoff = current_knee * 3
            if current_knee >= last_knee:
                break
        return current_knee

    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']

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
        min_c = refined_l_method(num_groups, merge_dist)
        end_time = time.time()

        print('refined_l_method time:', end_time - start_time)

        print('min_c:', min_c)

    return fn