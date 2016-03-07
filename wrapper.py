from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.linear_model import LinearRegression
import numpy as np
import l_method
from badness import *
from pipe import Pipe
from pipetools import *
from util import *

def kmeans(n_clusters=8, n_init=10):
    def fn(inst):
        x = requires('x', inst)

        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init)
        kmeans.fit(x)

        return inst.set('model', kmeans)\
            .set('prediction', kmeans.labels_)\
            .set('centroids', kmeans.cluster_centers_)

    return fn

def knn(*args, **margs):
    def fn(inst):
        x, y = requires(['x', 'y'], inst)

        print('len x:', len(x))
        print('len y:', len(y))

        knn = KNeighborsClassifier(*args, **margs)
        knn.fit(x, y)

        print('knn x:', x)

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

def agglomerative_l_method():
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']
        model = l_method.agglomerative_l_method(x)
        return inst.set('prediction', model.labels_)\
            .set('centroids', model.cluster_centers_)

    return fn

def recursive_agglomerative_l_method():
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']
        clusters = l_method.recursive_agglomerative_l_method(x)
        return inst.set('prediction', clusters)

    return fn

def kernel_density_estimation(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']

        kde = KernelDensity(*args, **margs)
        kde.fit(x)

        log_pdf = kde.score_samples(x)
        pdf = np.exp(log_pdf)

        return inst.set('model', kde).set('pdf', pdf)

    return fn

def badness_agglomeratvie_l_method(prepare=False):
    def prepare_fn(inst):
        # get good centroids
        x, y = requires(['x', 'y'], inst)

        # get the 'good' centroids
        result = Pipe() \
            .x(x) \
            .y(y) \
            .pipe(agglomerative_l_method()) \
            .connect(stop())
        if not 'centroids' in result:
            raise Exception('no centroids in pipe')

        return inst.set('good_centroids', result['centroids'])

    def fn(inst):
        x, y_seed, good_centroids = requires(['x', 'y_seed', 'good_centroids'], inst)

        # build seeding list
        seeding = []
        for i, each in enumerate(y_seed):
            if each is None:
                continue
            seeding.append(x[i])

        if len(seeding) == len(x):
            raise Exception('you probably seed with 100%')

        badness = {
            'rmsd': rmsd_nearest_from_centroids(seeding, good_centroids),
            'md': md_nearest_from_centroids(seeding, good_centroids),
        }

        print('badness:', badness)

        return inst.set('badness', badness)

    if prepare:
        return prepare_fn
    else:
        return fn

def good_K_for_KNN():
    # use Dataset class instead, it will do this automatically for you see dataset.py
    def fn(inst):
        x, y, x_test, y_test = \
            requires(['x', 'y', 'x_test', 'y_test'], inst)

        goodK = good_K_for_KNN_with_testdata(x, y, x_test, y_test)
        return inst.set('goodK', goodK)
    
    return fn