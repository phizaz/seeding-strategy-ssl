from numpy import inner
import numpy as np
import math
from sklearn.neighbors.kde import KernelDensity
from sklearn.grid_search import GridSearchCV
import time
from fast_climb_approx import create_fast_climb_kdtree
from random import shuffle

def kernel(x):
    d = len(x)
    return ((2 * math.pi) ** (-d / 2)) * math.exp(-inner(x, x) / 2)

def create_density_fn(X, bandwidth=.2, author='me'):
    if author == 'me':
        # based on the paper
        N = len(X)
        d = len(X[0])
        hd = bandwidth ** d

        def density(x):
            s = sum(kernel( (x - each_x) / bandwidth ) for each_x in X)
            return 1 / (N * hd) * s
        return density

    elif author == 'scikit':
        # faster !
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth, rtol=1e-6).fit(X)
        def scikit(x):
            return math.exp(kde.score([x]))

        return scikit

def create_gradienter(X, bandwidth=.2):
    d = len(X[0])
    hd2 = bandwidth ** (d + 2)
    N = len(X)

    def gradient(x):
        s = sum( kernel((x - each_x) / bandwidth) * (each_x - x) for each_x in X )
        return 1 / (hd2 * N) * s

    return gradient

def get_bandwidth(X, mode='cv'):
    def cv_bandwidth(X):
        # exhaustive search
        params = {
            'bandwidth': np.linspace(1e-4, 1.0, 1000)
        }
        grid = GridSearchCV(
                KernelDensity(rtol=1e-6),
                params,
                cv=10,
                n_jobs=-1)
        grid.fit(X)
        print('best_params:', grid.best_params_)
        return grid.best_params_['bandwidth']

    return cv_bandwidth(X)

def create_hill_climber(dataset, fast=True, ret_histroy=False):
    X = dataset.X
    # bandwidth = get_bandwidth(X, mode='cv')
    bandwidth = dataset.bandwidth
    density = create_density_fn(X, bandwidth, 'scikit')

    def normal_climber(rate=.01):
        # it is not as good should not be used!
        gradient = create_gradienter(X, bandwidth)

        def climb(x):
            g = gradient(x)
            # print('gradient:', g)
            size = inner(g, g) ** .5
            return x + rate * (g / size)

        def climb_till_end(x):
            current = x
            current_dense = density(current)
            if ret_histroy:
                history = [current]
            # print('start:', current)
            while True:
                next = climb(current)

                if ret_histroy:
                    history.append(next)
                # print('next:', next)

                next_dense = density(next)
                d = abs(next_dense - current_dense) / current_dense * 100
                # d = diff(current, next)

                current, current_dense = next, next_dense
                # current = next

                if d < 0.01:
                    # density increament less than 0.1%
                    break

            if ret_histroy:
                return current, history
            else:
                return current

        return climb_till_end

    def fast_climb_till_end(x, approx=None):

        if approx:
            # this is a future feature
            # for approximate gaussian processing
            # using kdtree to calculate the effect of far away points
            # and points in the same region (small) to have
            # the same averaged kernel weight
            fast_climb = create_fast_climb_kdtree(X, kernel, approx)
        else:
            def fast_climb(x):
                # denclue 2.0 paper
                start_time = time.process_time()
                computed_kernel = [kernel((x - each_x) / bandwidth) for each_x in X]
                s1 = sum(each_kernel * each_x for each_kernel, each_x in zip(computed_kernel, X))
                s2 = sum(computed_kernel)
                end_time = time.process_time()
                print('fast_climb:', end_time - start_time)
                return s1 / s2

        current = x
        current_dense = density(current)

        if ret_histroy:
            history = [current]

        while True:
            next = fast_climb(current)

            if ret_histroy:
                history.append(next)
            # print('next:', next)

            # start_time = time.process_time()
            next_dense = density(next)
            d = abs(next_dense - current_dense) / current_dense * 100
            # d = diff(current, next)
            # end_time = time.process_time()
            # print('diff:', end_time - start_time)

            current, current_dense = next, next_dense
            # current = next

            if d < 0.00001:
                # density increament less than 0.1%
                break

        if ret_histroy:
            return current, history
        else:
            return current

    if fast:
        return fast_climb_till_end
    else:
        return normal_climber

def denclue(dataset, sample_rate=.1):
    # sample is the number of points (ratio) to be climbing to the summit
    # most cases 10% is more than enough, since climbing is a very expensive process
    hill_climber = create_hill_climber(dataset)
    X = dataset.X[:]
    shuffle(X)
    sample_cnt = int(len(X) * sample_rate)

    centroids = set()
    for x in X[:sample_cnt]:
        summit = hill_climber(x)
        rounded_summit = tuple(map(lambda x: round(x, 4), summit))
        centroids.add(rounded_summit)

    return list(centroids)
