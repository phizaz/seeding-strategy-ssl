from numpy import inner
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity
from sklearn.grid_search import GridSearchCV

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
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
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
            'bandwidth': np.linspace(0.01, 1.0, 300)
        }
        grid = GridSearchCV(KernelDensity(), params, cv=10, n_jobs=-1)
        grid.fit(X)
        print('best_params:', grid.best_params_)
        return grid.best_params_['bandwidth']

    return cv_bandwidth(X)

def create_hill_climber(X, rate=.01):
    bandwidth = get_bandwidth(X, mode='cv')
    gradient = create_gradienter(X, bandwidth)
    density = create_density_fn(X, bandwidth, 'scikit')

    def diff(f, t):
        dif = f - t
        dist = inner(dif, dif) ** .5
        base = inner(f, f) ** .5
        return dist / base * 100

    def climb(x):
        g = gradient(x)
        # print('gradient:', g)
        size = inner(g, g) ** .5
        return x + rate * ( g / size )

    def fast_climb(x):
        # denclue 2.0 paper
        s1 = sum(kernel((x - each_x) / bandwidth) * each_x for each_x in X)
        s2 = sum(kernel((x - each_x) / bandwidth) for each_x in X)
        return s1 / s2


    def climb_till_end(x):
        current = x
        current_dense = density(current)
        history = []
        history.append(current)
        # print('start:', current)
        while True:
            next = fast_climb(current)
            next_dense = density(next)

            history.append(next)
            # print('next:', next)

            d = abs(next_dense - current_dense) / current_dense * 100

            current, current_dense = next, next_dense

            if d < 0.00001:
                # density increament less than 0.1%
                break

        return current, history

    return climb_till_end



